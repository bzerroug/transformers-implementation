from torch import nn
from transformers_blocks import (
    Embedding,
    FeedForward,
    MultiHeadSelfAttentionQKV,
    PositionalEmbedding,
)


class TransformerEncodingBloc(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super(TransformerEncodingBloc, self).__init__()
        # Mutli Head Attention with its notmalization and its dropout
        self.attention = MultiHeadSelfAttentionQKV(embedding_dim, num_heads)
        self.normalization_mha = nn.LayerNorm(embedding_dim)
        self.dropout_mha = nn.Dropout(0.2)

        # Feed Forward with its notmalization and its dropout
        self.feed_forward = FeedForward(embedding_dim, factor)
        self.normalization_ff = nn.LayerNorm(embedding_dim)
        self.dropout_ff = nn.Dropout(0.2)

    def forward(self, query, key, value):
        # Multi Head Attention
        mha = self.attention(query, key, value)
        mha_residuals = mha + value
        mha_residuals_norm = self.normalization_mha(mha_residuals)
        mha_residuals_norm_dropout = self.dropout_mha(mha_residuals_norm)

        # Feed Forward
        ff = self.feed_forward(mha_residuals_norm_dropout)
        ff_residuals = ff + mha_residuals_norm_dropout
        ff_residuals_norm = self.normalization_ff(ff_residuals)
        ff_residuals_norm_dropout = self.dropout_ff(ff_residuals_norm)

        return ff_residuals_norm_dropout


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, factor):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncodingBloc(embedding_dim, num_heads, factor)
                for i in range(num_layers)
            ]
        )

    def forward(self, X):
        encoded = self.embedding(X)
        output = self.positional_embedding(encoded)
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output, output, output)

        return output
