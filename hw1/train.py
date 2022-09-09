import json

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
import model as Model

import utils
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
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
    vocab_to_index, index_to_vocab, maxLength = utils.build_tokenizer_table(trainRawData)
    action_to_index, label_to_index, target_to_index, index_to_target = build_output_tables(trainRawData)

    # setup some utility variables
    unkValue = vocab_to_index["<unk>"]
    startValue = vocab_to_index["<start>"]
    endValue = vocab_to_index["<end>"]
    padValue = vocab_to_index["<pad>"]

    train_tokens, train_labels = __processTrainingSet__(trainRawData, vocab_to_index, action_to_index, target_to_index,
                                                        startValue, endValue,
                                                        unkValue, padValue, maxLength)
    train_processed = torch.utils.data.TensorDataset(torch.from_numpy(train_tokens), torch.from_numpy(train_labels))

    val_tokens, val_labels = __processTrainingSet__(validateRawData, vocab_to_index, action_to_index, target_to_index,
                                                    startValue, endValue,
                                                    unkValue, padValue, maxLength)
    val_processed = torch.utils.data.TensorDataset(torch.from_numpy(val_tokens), torch.from_numpy(val_labels))

    train_loader = torch.utils.data.DataLoader(train_processed, shuffle=True, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_processed,
                                             shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader, {"vocab": index_to_vocab, "action": action_to_index, "target": target_to_index,
                                      "maxLength": maxLength}


def numberOfExamples(trainRawData):
    i = int(0)
    for episode in trainRawData:
        for instance in episode:
            i += 1

    return i


def __processTrainingSet__(trainRawData, vocab_to_index, action_to_index, target_to_index, startValue, endValue,
                           unkValue, padValue, maxLength):
    # loop through each example
    processedTrainData = np.zeros((numberOfExamples(trainRawData), maxLength), dtype=np.int32)
    processedLabelData = np.zeros((numberOfExamples(trainRawData), 2), dtype=np.int32)

    taskIndex = 0
    wordIndex = 0
    for episode in trainRawData:
        for instance in episode:
            wordIndex = 0
            labels = instance[1]
            sentences = instance[0]
            sentences = preprocess_string(sentences)
            processedTrainData[taskIndex][wordIndex] = startValue
            wordIndex += 1

            # tokenize the sentence
            for word in sentences.split():
                processedTrainData[taskIndex][wordIndex] = vocab_to_index.get(word, unkValue)
                wordIndex += 1
                if wordIndex >= maxLength:
                    break

            if wordIndex < maxLength:
                for i in range(maxLength - wordIndex):
                    processedTrainData[taskIndex][wordIndex] = padValue
                    wordIndex += 1

            # tokenize the labels
            processedLabelData[taskIndex][0] = action_to_index.get(labels[0])
            processedLabelData[taskIndex][1] = target_to_index.get(labels[1])
            taskIndex += 1

    return processedTrainData, processedLabelData


def setup_model(args, maps, devices):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = Model.LanguageModel(devices, len(maps["vocab"]), maps["maxLength"], len(maps["action"]), len(maps["target"]), args.embedding_dim, args)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
        args,
        model,
        loader,
        optimizer,
        action_criterion,
        target_criterion,
        device,
        training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out  = model(inputs)
        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out, labels[:, 0].long())
        target_loss = target_criterion(targets_out, labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
        args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    action = {"trainingLoss":[], "trainingAcc":[], "valLoss":[], "valAcc":[]}
    target = {"trainingLoss":[], "trainingAcc":[], "valLoss":[], "valAcc":[]}
    x = []
    valx = []
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )

            valx.append(epoch)
            action["valLoss"].append(val_action_loss)
            action["valAcc"].append(val_action_acc)
            target["valLoss"].append(val_target_loss)
            target["valAcc"].append(val_target_acc)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
        action["trainingLoss"].append(train_action_loss)
        action["trainingAcc"].append(train_action_acc)
        target["trainingLoss"].append(train_action_loss)
        target["trainingAcc"].append(train_action_acc)

        x.append(epoch)

    plt.title("Action: loss performance")
    plt.plot(x, action["trainingLoss"], label = "action training loss")
    plt.plot(valx, action["valLoss"], label = "action validation set loss")
    plt.legend()
    plt.show()

    plt.title("Target: loss performance")
    plt.plot(x, target["trainingLoss"], label = "target training loss")
    plt.plot(valx, target["valLoss"], label = "target validation loss")
    plt.legend()
    plt.show()

    plt.title("Action: Accuracy performance")
    plt.plot(x, action["trainingAcc"], label = "action training accuracy")
    plt.plot(valx, action["valAcc"], label = "action validation set accuracy")
    plt.legend()
    plt.show()

    plt.title("Target: Accuracy Performance")
    plt.plot(x, target["trainingAcc"], label = "target training accuracy")
    plt.plot(valx, target["valAcc"], label = "target validation accuracy")
    plt.legend()
    plt.show()

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
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
    parser.add_argument("--hidden_size", type=int, default=10, help="size of the model's hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="The learning rate for the model")
    parser.add_argument("--embedding_dim", type=int, default=10, help="The embedding dimension of the model")
    args = parser.parse_args()

    main(args)
