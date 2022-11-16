# IMPLEMENT YOUR MODEL CLASS HERE
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, vocabSize, embeddingSize, encoderHIddenSize, batchSize, padIdx):
        super().__init__()
        self.embeddingSize = int(embeddingSize / 2) * 2  # round up to the next even number
        self.encoderHiddenSize = encoderHIddenSize
        self.batchSize = batchSize
        self.embedding = nn.Embedding(vocabSize, self.embeddingSize, padding_idx=padIdx)
        self.lstm = nn.LSTM(self.embeddingSize, encoderHIddenSize, batch_first=True)

    def forward(self, word, h, c):
        if h is None or c is None:
            h = torch.zeros((1, word.size(0), self.encoderHiddenSize))
            c = torch.zeros((1, word.size(0), self.encoderHiddenSize))

        embedding = self.embedding(word)
        # embedding = torch.unsqueeze(embedding, 0)
        output, (x, y) = self.lstm(embedding, (h, c))
        return output, (x, y)


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, batch_size, maxLength, maxEpisodeLength, decoderHiddenSize, numberOfActions,
                 numberOfTargets, vocabSize, padIdx):
        super().__init__()
        self.maxEpisodeLength = maxEpisodeLength
        self.batchSize = batch_size
        self.embeddingSize = numberOfTargets + numberOfActions  # round up to the next even number
        self.decoderHiddenSize = decoderHiddenSize
        self.lstm = nn.LSTM(self.embeddingSize, decoderHiddenSize, batch_first=True)
        self.fcAction = nn.Linear(self.decoderHiddenSize, numberOfActions)
        self.fcTarget = nn.Linear(self.decoderHiddenSize, numberOfTargets)

    """
        actions  = []
        objects = []
        for batchette in x:
            prevOutput = batchette
            prevOutputState = torch.zeros((1, 1, self.decoderHiddenSize))
            prevCellState = torch.zeros((1, 1, self.decoderHiddenSize))
            actionToAdd = []
            objectToAdd = []
            #At reach time step, use input from array. If array runs out, then you're done
            for i in range(self.maxEpisodeLength):
                latentState, (prevOutputState, prevCellState) = self.lstm(prevOutput, (prevOutputState, prevCellState))
                actionToAdd.append(self.fcAction(latentState))
                objectToAdd.append(self.fcObject(latentState))
            actions.append(actionToAdd)
            objectToAdd.append(objectToAdd)
        return np.array(actions), np.array(objects)
    """

    # For the first word, the encoder hidden state is given as the first argument
    def forward(self, prevAction, prevTarget, prevH, prevC):
        prevEmbedded = torch.unsqueeze(torch.cat((prevAction, prevTarget), 1).to(torch.float), 1)

        output, (x, y) = self.lstm(prevEmbedded, (prevH, prevC))

        action = self.fcAction(prevH)
        target = self.fcTarget(prevH)

        return action, target, (x, y)


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, batch_size, max_length, vocabSize, embeddingSize, maxEpisodeLength, decoderHiddenSize,
                 numberOfActions, numberOfTargets, encoderHiddenSize, endToken, padToken):
        super().__init__()
        self.endToken = endToken
        self.padToken = padToken
        self.maxEpisodeLength = maxEpisodeLength
        self.batchSize = batch_size
        self.numberOfActions = numberOfActions
        self.numberOfTargets = numberOfTargets
        self.encoder = Encoder(vocabSize, embeddingSize, encoderHiddenSize, batch_size, padToken)
        self.decoder = Decoder(batch_size, max_length, maxEpisodeLength, encoderHiddenSize,
                               numberOfActions, numberOfTargets, vocabSize, padToken)

    # Must be given an array of words chosen, this can be used for teacher forcing or for greedy
    # Input is of shape (batch size, length of words), label is (batch size, max length of labels, 2)
    def forward(self, input, labels=None, teacherForcing=True):
        encoding = self.getEncoding(input)

        if labels is None:
            decoding = self.getGreedyDecoding(encoding)
        else:
            if teacherForcing:
                decoding = self.getTeacherForcedDecoding(encoding, labels)
            else:
                decoding = self.getGreedyDecoding(encoding)

        return decoding

    def getEncoding(self, input):
        encoding = None
        prevH = None
        prevC = None

        encoding, (prevH, prevC) = self.encoder.forward(input, prevH, prevC)  # encode the whole thing in one go

        """
        come back to feed in input sentence by sentence for attention
        for batchette in input:
            for word in batchette:
                encoding, (prevH, prevC) = self.encoder.forward(word, prevH, prevC)
                if word.item() == self.endToken:
                    break
        """
        return prevH, prevC

    def getGreedyDecoding(self, encoding):
        prevH = encoding[0]
        prevC = encoding[1]
        shapeOfActions = (self.batchSize, 1, 1)
        shapeOfTargets = (self.batchSize, 1, 1)
        currentAction = torch.nn.functional.one_hot(torch.zeros(shapeOfActions).to(torch.int64), self.numberOfActions).squeeze()
        currentTarget = torch.nn.functional.one_hot(torch.zeros(shapeOfTargets).to(torch.int64), self.numberOfTargets).squeeze()
        previousPredictedAction = None
        previousPredictedTarget = None
        for i in range(self.maxEpisodeLength):
            predictedAction, predictedTarget, (prevH, prevC)= self.decoder.forward(currentAction, currentTarget, prevH, prevC)

            currentAction = torch.nn.functional.one_hot(torch.max(predictedAction, 2, True)[1], self.numberOfActions).squeeze().squeeze()
            currentTarget = torch.nn.functional.one_hot(torch.max(predictedAction, 2, True)[1], self.numberOfTargets).squeeze().squeeze()

            if previousPredictedAction is not None:
                previousPredictedAction = torch.cat((previousPredictedAction, predictedAction), 0)
            else:
                previousPredictedAction = predictedAction

            if previousPredictedTarget is not None:
                previousPredictedTarget = torch.cat((previousPredictedTarget, predictedTarget), 0)
            else:
                previousPredictedTarget = predictedTarget

        return torch.swapaxes(previousPredictedAction, 0, 1), torch.swapaxes(previousPredictedTarget, 0, 1)

    # encoder shouldnt be input it should be hidden state
    # need to put a start, stop, and pad token in the action and target, make sure it doesnt coincide
    # make it batchwise
    def getTeacherForcedDecoding(self, encoding, labels):
        prevH = encoding[0]
        prevC = encoding[1]
        actions, targets = torch.split(labels, [1, 1], 2)
        actions = torch.split(actions, 1, 1)
        targets = torch.split(targets, 1, 1)
        currentAction = torch.nn.functional.one_hot(torch.zeros_like(actions[0]).to(torch.int64), self.numberOfActions).squeeze()
        currentTarget = torch.nn.functional.one_hot(torch.zeros_like(targets[0]).to(torch.int64), self.numberOfTargets).squeeze()

        previousPredictedAction = None
        previousPredictedTarget = None
        for i in range(self.maxEpisodeLength):
            predictedAction, predictedTarget, (prevH, prevC)= self.decoder.forward(currentAction, currentTarget, prevH, prevC)

            currentAction = torch.nn.functional.one_hot(actions[i].to(torch.int64), self.numberOfActions).squeeze()
            currentTarget = torch.nn.functional.one_hot(targets[i].to(torch.int64), self.numberOfTargets).squeeze()

            if previousPredictedAction is not None:
                previousPredictedAction = torch.cat((previousPredictedAction, predictedAction), 0)
            else:
                previousPredictedAction = predictedAction

            if previousPredictedTarget is not None:
                previousPredictedTarget = torch.cat((previousPredictedTarget, predictedTarget), 0)
            else:
                previousPredictedTarget = predictedTarget

        return torch.swapaxes(previousPredictedAction, 0, 1), torch.swapaxes(previousPredictedTarget, 0, 1)
        """
        allPredictedActions = None
        allPredictedTarget = None
        for batchette, partialEncodingH, partialEncodingC in zip(labels, encoding[0].squeeze(), encoding[1].squeeze()):
            prevAction = torch.zeros(1)
            prevTarget = torch.zeros(1)
            prevH = partialEncodingH.unsqueeze(0)
            prevC = partialEncodingC.unsqueeze(0)
            previousPredictedAction = None
            previousPredictedTarget = None
            for label in batchette:
                predictedAction, predictedTarget, (prevH, prevC) = self.decoder.forward(torch.nn.functional.one_hot(prevAction.to(torch.int64), self.numberOfActions),
                                                                                        torch.nn.functional.one_hot(prevTarget.to(torch.int64), self.numberOfTargets), prevH, prevC)

                if (predictedAction is not None) and (previousPredictedAction is not None):
                    previousPredictedAction = torch.cat((predictedAction, previousPredictedAction), 0)
                else:
                    previousPredictedAction = predictedAction

                if (predictedTarget is not None) and (previousPredictedTarget is not None):
                    previousPredictedTarget = torch.cat((predictedTarget, previousPredictedTarget), 0)
                else:
                    previousPredictedTarget = predictedTarget

                prevAction = label[0]
                prevTarget = label[1]

                if (prevAction == self.padToken) or (prevTarget == self.padToken):
                    actionPaddingTensor = torch.full(
                        (self.maxEpisodeLength - previousPredictedAction.size(0), self.numberOfActions), self.padToken)
                    targetPaddingTensor = torch.full(
                        (self.maxEpisodeLength - previousPredictedTarget.size(0), self.numberOfTargets), self.padToken)
                    previousPredictedAction = torch.cat((actionPaddingTensor, previousPredictedAction), 0)
                    previousPredictedTarget = torch.cat((targetPaddingTensor, previousPredictedTarget), 0)
                    break

            if allPredictedTarget is None:
                allPredictedTarget = previousPredictedTarget.unsqueeze(0)
                allPredictedActions = previousPredictedAction.unsqueeze(0)
            else:
                allPredictedActions = torch.cat((allPredictedActions, previousPredictedAction.unsqueeze(0)), 0)
                allPredictedTarget = torch.cat((allPredictedTarget, previousPredictedTarget.unsqueeze(0)), 0)

        return allPredictedActions, allPredictedTarget
        """
