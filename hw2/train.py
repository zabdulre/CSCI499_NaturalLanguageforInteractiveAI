import argparse
import os

import numpy as np
import tqdm
import torch
from sklearn.metrics import accuracy_score
from torch.utils import data

from eval_utils import downstream_validation
import utils
import data_utils
from model import LanguageModel


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #
    lenContext = args.len_context
    lenContextLeft = lenContext // 2  # The model will look one extra to the right of the word if the context length is odd
    padding_token = vocab_to_index['<pad>']
    end_token = vocab_to_index['<end>']
    # listOfContext = np.zeros((numberOfExamples(encoded_sentences), args.vocab_size), dtype=np.int32)
    # listOfWords = np.zeros((numberOfExamples(encoded_sentences), 1), dtype=np.int32)
    exampleNumber = -1
    contextArr = []
    wordArr = []

    for sentence, length in zip(encoded_sentences, lens):
        for word, i in zip(sentence, range(length[0])):
            index = 0
            first = True
            for j in range(lenContext + 1):  # plus one so that we can skip the word itself
                if (j - lenContextLeft) == 0:
                    continue
                elif j + i < lenContextLeft:
                    break
                    # contextArr[exampleNumber].append(padding_token)
                    # index += 1
                elif (i + (lenContext - lenContextLeft) + 1) >= length[0]:
                    break
                    # contextArr[exampleNumber].append(padding_token)
                    # index += 1
                else:
                    if first:
                        contextArr.append([])
                        wordArr.append([word])
                        exampleNumber += 1
                    first = False
                    contextArr[exampleNumber].append(sentence[j + i - lenContextLeft])
                    index += 1
            if word == end_token:
                break

    # listOfContext = [np.array(i) for i in contextArr]
    listOfContext = np.array(contextArr)
    listOfWords = np.array(wordArr)
    # for a given sentence loop through the words
    # for each word, add the surrounding context to one list
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(listOfWords), torch.from_numpy(listOfContext))
    # dataset, _ = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.005), (len(dataset) - int(len(dataset) * 0.005))],
    # generator=torch.Generator().manual_seed(21))

    tSize = int(0.8 * len(dataset))
    vSize = len(dataset) - tSize
    train_processed, val_processed = torch.utils.data.random_split(dataset, [tSize, vSize],
                                                                   generator=torch.Generator().manual_seed(21))

    train_loader = torch.utils.data.DataLoader(train_processed, shuffle=True, batch_size=args.batch_size,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_processed,
                                             shuffle=True, batch_size=args.batch_size, drop_last=True)
    return train_loader, val_loader, index_to_vocab


def numberOfExamples(encoded_sentences):
    return encoded_sentences.shape[0] * encoded_sentences.shape[1]


def setup_model(args, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = LanguageModel(device, args.embedding_dim, args.vocab_size, args.len_context)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return criterion, optimizer


def train_epoch(
        args,
        model,
        loader,
        optimizer,
        criterion,
        device,
        i2v,
        training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []
    acc = 0
    trainIndex = 0
    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        trainIndex += 1
        newLabels = torch.zeros((args.batch_size, args.vocab_size), dtype=torch.int32)

        index = 0
        for i in labels:
            for j in i:
                newLabels[index][j] = 1
            index += 1

        labels = newLabels
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).float()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze(), labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        # preds = torch.as_tensor(pred_logits > 0.0, dtype=torch.int32).squeeze()
        preds = torch.topk(pred_logits, args.len_context, dim=2).indices.squeeze()
        acc += iou_accuracy(preds.cpu(), labels.cpu(), args)

    # acc = accuracy_score(pred_labels, target_labels)
    acc /= trainIndex
    epoch_loss /= len(loader)

    return epoch_loss, acc


def iou_accuracy(preds, labels, args):
    intersection = 0
    index = 0
    for i in preds:
        for j in i:
            if labels[index][j] == 1:
                intersection += 1
        index += 1
    return intersection / (args.len_context * args.batch_size)


def validate(args, model, loader, optimizer, criterion, device, i2v):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            i2v,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            i2v
        )
        print(f"train loss : {train_loss} | train acc: {train_acc}")

        # do it after training
        # save word vectors
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        print("saving word vec to ", word_vec_file)
        utils.save_word2vec_format(word_vec_file, model, i2v)

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
                i2v
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

        # evaluate learned embeddings on a downstream task
        downstream_validation(word_vec_file, external_val_analogies)

        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.outputs_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128
        , help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=3, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument(
        "--len_context", type=int, default=5, help="How much to the left, right the model should look"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate used for the optimizer"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=5, help="Embedding dimension used as input to the fully connected layer"
    )
    args = parser.parse_args()
    main(args)
