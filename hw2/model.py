import torch


class LanguageModel(torch.nn.Module):
    def __init__(
            self,
            device,
            embedding_dim,
            vocab_size
    ):
        super(LanguageModel, self).__init__()
        self.device = device

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc = torch.nn.Linear(embedding_dim, vocab_size)  # might have to tweak so that ouput has a 0 and a 1 bit

    def forward(self, x):
        #  batch_size, seq_len = x.size(0), x.size(1)

        embedded = self.embedding(x)

        out = self.fc(embedded)

        return out
