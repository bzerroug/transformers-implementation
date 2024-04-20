import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class OneHeadSelfAttentionQKV(nn.Module):
    def __init__(self, k, low_dim):
        super().__init__()
        # Check if input is divisible by number of heads
        self.k = k
        self.low_dim = low_dim
        # 1. Define linear transformations to reduce dimensionnalité of input
        # biais = False because we want only weights
        self.to_reduce_dim = nn.Linear(k, low_dim, bias=False)
        # 2. Define linear transformations to key, queries and values
        # biais = False because we want only weights
        self.to_queries = nn.Linear(low_dim, low_dim, bias=False)
        self.to_keys = nn.Linear(low_dim, low_dim, bias=False)
        self.to_values = nn.Linear(low_dim, low_dim, bias=False)

    def forward(self, Q, K, V):
        # 3. Reduce dimensionnalité of input
        low_dim_Q = self.to_reduce_dim(Q)
        low_dim_K = self.to_reduce_dim(K)
        low_dim_V = self.to_reduce_dim(V)

        # 4. Apply the linear transformation associated to every input to obtain the key, query and value
        query = self.to_queries(low_dim_Q)
        key = self.to_keys(low_dim_K)
        value = self.to_values(low_dim_V)

        # 5. Compute the raw weights w′ij=𝐪iT𝐤j and normalize them
        weights_raw = torch.bmm(query, key.transpose(1, 2))
        weights_raw_normalized = torch.div(
            weights_raw, torch.sqrt(torch.tensor(self.low_dim))
        )

        # 6. We apply the Softmax function to the similarity dimension (batch dim x input dim x sim dim)
        weights = nn.Softmax(dim=2)(weights_raw_normalized)

        # 7. Multiply weights of self attention to the values
        return torch.bmm(weights, value)


class MultiHeadSelfAttentionQKV(nn.Module):
    # 8.Define a head number that is divisible from the input
    def __init__(self, k, heads=4):
        super().__init__()
        # Check if input is divisible by number of heads
        assert k % heads == 0

        self.k = k
        self.heads = heads

        # 9. Instantiate OneHeadSelfAttention multiple times to have MultiHeadSelfAttention
        self.list_heads = []
        for head in range(self.heads):
            self.list_heads.append(OneHeadSelfAttentionQKV(k, k // heads))

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)

    def forward(self, Q, K, V):
        # 10. Get all heads elements
        list_to_concat = []
        for one_head in self.list_heads:
            list_to_concat.append((one_head(Q, K, V),))

        # 11. Concatenate all the heads
        multi_heads = sum(list_to_concat, ())
        concatenated = torch.cat(multi_heads, dim=2)

        # 12. Linear transformation
        return self.unifyheads(concatenated)


class MultHeadsSelfAttentionOptQKV(nn.Module):
    # 1.Define a head number that is divisible from the input
    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        # Check if input is divisible by number of heads
        assert k % heads == 0

        self.k = k
        self.heads = heads

        # 2. Define linear transformations to key, queries and values for each head
        # biais = False because we want only weights
        self.to_queries = nn.Linear(k, k, bias=False)
        self.to_keys = nn.Linear(k, k, bias=False)
        self.to_values = nn.Linear(k, k, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(k, k)

    def forward(self, Q, K, V):
        b, t, k = Q.size()  # as the training will be done by batch

        # 3. Apply the linear transformation associated to every input to obtain the key, query and value
        query = self.to_queries(Q)
        key = self.to_keys(K)
        value = self.to_values(V)

        s = self.k // self.heads  # number of elements per head
        h = self.heads

        # 4. Reshape the matrix of key, query and value to have them in different heads.
        queries = query.view(b, t, h, s)
        keys = key.view(b, t, h, s)
        values = value.view(b, t, h, s)

        # 5. Merge heads and batch because it's the same operation for each head
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # 6. Compute the raw weights w′ij=𝐪iT𝐤j and normalize them
        weights_raw = torch.bmm(queries, keys.transpose(1, 2))
        weights_raw_normalized = torch.div(weights_raw, torch.sqrt(torch.tensor(k)))

        # 7. We apply the Softmax function to the similarity dimension (batch dim x input dim x sim dim)
        weights = nn.Softmax(dim=2)(weights_raw_normalized)

        # 8. Multiply weights of self attention to the values
        self_attentions = torch.bmm(weights, values).view(b, h, t, s)

        # 9. Reshape in order to concatenatre heads and have b x t x k
        self_attention_formatted = (
            self_attentions.transpose(1, 2).contiguous().view(b, t, s * h)
        )

        # 10. Apply the unifyheads an return it
        return self.unifyheads(self_attention_formatted)
