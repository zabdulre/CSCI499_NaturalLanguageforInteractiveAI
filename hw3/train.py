import json

import numpy as np
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score

import utils
from model import EncoderDecoder
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    train_loader = None
    val_loader = None

    # Load JSON file into a dictionary
    file = open(args.in_data_fn)
    rawData = json.load(file)

    # split the data into a train set and validation set
    trainRawData = rawData["train"]
    validateRawData = rawData["valid_seen"]

    # preprocess both sets and tokenize
    vocab_to_index, index_to_vocab, maxLength, maxEpisodeLength = utils.build_tokenizer_table(trainRawData)

    unkValue = vocab_to_index["<unk>"]
    startValue = vocab_to_index["<start>"]
    endValue = vocab_to_index["<end>"]
    padValue = vocab_to_index["<pad>"]
    sepValue = vocab_to_index["<sep>"]

    action_to_index, label_to_index, target_to_index, index_to_target = build_output_tables(trainRawData, padValue)

    # setup some utility variables

    train_tokens, train_labels = __processTrainingSet__(trainRawData, vocab_to_index, action_to_index, target_to_index,
                                                        startValue, endValue,
                                                        unkValue, padValue, sepValue, maxLength, maxEpisodeLength)
    train_processed = torch.utils.data.TensorDataset(torch.from_numpy(train_tokens[0:2000]), torch.from_numpy(train_labels[0:2000]))

    val_tokens, val_labels = __processTrainingSet__(validateRawData, vocab_to_index, action_to_index, target_to_index,
                                                    startValue, endValue,
                                                    unkValue, padValue, sepValue, maxLength, maxEpisodeLength)
    val_processed = torch.utils.data.TensorDataset(torch.from_numpy(val_tokens), torch.from_numpy(val_labels))

    train_loader = torch.utils.data.DataLoader(train_processed, shuffle=True, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_processed,
                                             shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader, {"vocab": vocab_to_index, "action": action_to_index, "target": target_to_index,
                                      "maxLength": maxLength, "maxEpisodeLength": maxEpisodeLength, "pad": padValue}


def __processTrainingSet__(trainRawData, vocab_to_index, action_to_index, target_to_index, startValue, endValue,
                           unkValue, padValue, sepValue, maxLength, maxEpisodeLength):
    # loop through each example
    processedTrainData = np.zeros((len(trainRawData), maxLength), dtype=np.int32)
    processedLabelData = np.zeros((len(trainRawData), maxEpisodeLength, 2), dtype=np.int32)

    taskIndex = 0

    for episode in trainRawData:
        wordIndex = 0
        episodeIndex = 0
        processedTrainData[taskIndex][0] = startValue
        # processedLabelData[taskIndex][0][0] = startValue
        # processedLabelData[taskIndex][0][1] = startValue
        # episodeIndex += 1
        for instance in episode:
            labels = instance[1]
            sentences = instance[0]
            sentences = preprocess_string(sentences)

            # tokenize the sentence
            for word in sentences.split():
                processedTrainData[taskIndex][wordIndex] = vocab_to_index.get(word, unkValue)
                wordIndex += 1
                if wordIndex >= maxLength:
                    break
            # tokenize the labels
            processedLabelData[taskIndex][episodeIndex][0] = action_to_index.get(labels[0])
            processedLabelData[taskIndex][episodeIndex][1] = target_to_index.get(labels[1])
            episodeIndex += 1
            wordIndex += 1
            if wordIndex >= maxLength:
                break
            processedTrainData[taskIndex][wordIndex] = sepValue

        if wordIndex < maxLength:
            wordIndex += 1
            if wordIndex < maxLength:
                processedTrainData[taskIndex][wordIndex] = endValue
            for i in range(maxLength - wordIndex):
                processedTrainData[taskIndex][wordIndex] = padValue
                wordIndex += 1
        if episodeIndex < maxEpisodeLength:
            #    episodeIndex += 1
            #      if episodeIndex < maxEpisodeLength:
            #            processedLabelData[taskIndex][0][0] = endValue
            #             processedLabelData[taskIndex][0][1] = endValue
            for i in range(maxEpisodeLength - episodeIndex):
                processedLabelData[taskIndex][episodeIndex][0] = padValue
                processedLabelData[taskIndex][episodeIndex][1] = padValue
                episodeIndex += 1
        taskIndex += 1

    return processedTrainData, processedLabelData


def setup_model(args, maps, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    vocab = maps["vocab"]
    model = EncoderDecoder(args.batch_size, maps["maxLength"], len(maps["vocab"]), args.embedding_size,
                           maps['maxEpisodeLength'], args.decoder_size, len(maps['action']), len(maps['target']),
                           args.encoder_size, vocab['<end>'], vocab['<pad>'])
    return model


def setup_optimizer(args, model, maps):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss(ignore_index=maps['pad'])
    target_criterion = torch.nn.CrossEntropyLoss(ignore_index=maps['pad'])
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
        args,
        model,
        loader,
        optimizer,
        actionCriterion,
        targetCriterion,
        device,
        training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    global prefix_em
    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actionOutputs, targetOutputs = model(inputs, labels)

        actionLoss = 0
        targetLoss = 0

        actionLabels, targetLabels = torch.split(labels, (1, 1), dim=2)
        actionLoss = actionCriterion(actionOutputs.swapaxes(1, 2), actionLabels.squeeze().long())
        targetLoss = targetCriterion(targetOutputs.swapaxes(1, 2), targetLabels.squeeze().long())

        loss = actionLoss + targetLoss
        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        # em = output == labels
        # prefix_em = prefix_em(output, labels)
        # acc = 0.0

        # logging
        epoch_loss += targetLoss.item() + actionLoss.item()
        epoch_acc += 0.0

    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, actionCriterion, targetCriterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            actionCriterion,
            targetCriterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, actionCriterion, targetCriterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            actionCriterion,
            targetCriterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                actionCriterion,
                targetCriterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    actionCriterion, targetCriterion, optimizer = setup_optimizer(args, model, maps)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            actionCriterion, targetCriterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, actionCriterion, targetCriterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="size of each batch in loader"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=32, help="size of the encoder embeddings"
    )
    parser.add_argument(
        "--decoder_size", type=int, default=32, help="size of the decoder hidden size"
    )
    parser.add_argument(
        "--encoder_size", type=int, default=32, help="size of the decoder hidden size"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, type=int, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
