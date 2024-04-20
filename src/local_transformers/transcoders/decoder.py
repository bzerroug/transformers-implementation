from torch import nn
from ..transformers_blocks import (
    Embedding,
    FeedForward,
    MultiHeadSelfAttentionQKV,
    PositionalEmbedding,
)


class TransformerDecodingBloc(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super(TransformerDecodingBloc, self).__init__()
        # Mutli Head Self Attention with its notmalization and its dropout
        self.self_attention = MultiHeadSelfAttentionQKV(embedding_dim, num_heads)
        self.normalization_mhsa = nn.LayerNorm(embedding_dim)
        self.dropout_mhsa = nn.Dropout(0.2)

        # Mutli Head cross Attention with its notmalization and its dropout
        self.cross_attention = MultiHeadSelfAttentionQKV(embedding_dim, num_heads)
        self.normalization_mhca = nn.LayerNorm(embedding_dim)
        self.dropout_mhca = nn.Dropout(0.2)

        # Feed Forward with its notmalization and its dropout
        self.feed_forward = FeedForward(embedding_dim, factor)
        self.normalization_ff = nn.LayerNorm(embedding_dim)
        self.dropout_ff = nn.Dropout(0.2)

    def forward(self, x, encoder, mask):
        # Mutli Head Self Attention
        mhsa = self.self_attention(x, x, x, mask)
        mhsa_residuals = mhsa + x
        mhsa_residuals_norm = self.normalization_mhsa(mhsa_residuals)
        mhsa_residuals_norm_dropout = self.dropout_mhsa(mhsa_residuals_norm)

        # Mutli Head Cross Attention
        query = mhsa_residuals_norm_dropout
        mhca = self.cross_attention(
            query, encoder, encoder
        )  # Query belongs to self attention and Key and Value from encoder
        mhca_residuals = mhca + query
        mhca_residuals_norm = self.normalization_mhca(mhca_residuals)
        mhca_residuals_norm_dropout = self.dropout_mhca(mhca_residuals_norm)

        # Feed Forward with its notmalization and its dropout
        ff = self.feed_forward(mhca_residuals_norm_dropout)
        ff_residuals = ff + mhca_residuals_norm_dropout
        ff_residuals_norm = self.normalization_ff(ff_residuals)
        ff_residuals_norm_dropout = self.dropout_ff(ff_residuals_norm)

        return ff_residuals_norm_dropout


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, factor=1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecodingBloc(embedding_dim, num_heads, factor)
                for i in range(num_layers)
            ]
        )
        self.linear_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, encoder_out, mask=None):
        encoded = self.embedding(x)
        output = self.positional_embedding(encoded)
        output = self.dropout(output)
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output, encoder_out, mask)

        output = self.linear_out(output)

        return nn.Softmax(dim=2)(output)
