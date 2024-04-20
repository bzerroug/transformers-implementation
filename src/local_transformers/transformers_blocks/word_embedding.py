import math

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, dimension):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, dimension)

    def forward(self, x):
        return self.word_embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, dimension, max_seq_length=2000):
        super(PositionalEmbedding, self).__init__()

        positional_encoding = torch.zeros(max_seq_length, dimension)
        for pos in range(max_seq_length):
            for i in range(dimension):
                if i % 2 == 0:
                    pe = math.sin(pos / 1000 ** (2 * i / dimension))
                else:
                    pe = math.cos(pos / 1000 ** (2 * i / dimension))
                positional_encoding[pos, i] = pe

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        return x + self.positional_encoding[: x.size(1), :]
