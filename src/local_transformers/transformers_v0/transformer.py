from ..transcoders import Encoder, Decoder
import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, num_heads, num_layers, factor=2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_vocab_size, embedding_dim, num_heads, num_layers, factor)
        self.decoder = Decoder(output_vocab_size, embedding_dim, num_heads, num_layers, factor)

    def get_mask_output(self, output):
        _, output_len = output.size()
        return torch.tril(torch.ones((output_len, output_len)), diagonal=1).bool()
    

    def forward(self, input, output):
        encoder = self.encoder(input) 
        mask = self.get_mask_output(output)
        return self.decoder(output, encoder, mask=mask) 
