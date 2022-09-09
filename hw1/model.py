# IMPLEMENT YOUR MODEL CLASS HERE
import torch


class LanguageModel(torch.nn.Module):
    def __init__(
            self,
            device,
            vocab_size,
            input_len,
            actions,
            targets,
            embedding_dim,
            args
    ):
        super(LanguageModel, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.hidden_size = args.hidden_size

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.LSTM = torch.nn.LSTM(embedding_dim * input_len, self.hidden_size, 1, dropout=0.0, batch_first=True)

        self.fcAction = torch.nn.Linear(self.hidden_size, actions)
        self.fcTarget = torch.nn.Linear(self.hidden_size, targets)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        embedded = self.embedding(x).view(-1, 1, seq_len*self.embedding_dim)

        _, (lstm1, _ ) = self.LSTM(embedded)

        out = (self.fcAction(lstm1).squeeze(), self.fcTarget(lstm1).squeeze())

        return out
