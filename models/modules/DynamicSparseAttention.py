from torch import nn
import torch


# 动态稀疏注意力机制
class DynamicSparseAttention(nn.Module):
    def __init__(self, embed_size, heads, sparsity):
        super(DynamicSparseAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.sparsity = sparsity
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        if sparsity < 0 or sparsity >= 1:
            raise ValueError("sparsity must be between 0 and 1")

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, value_len, self.heads, self.embed_size // self.heads)
        keys = keys.view(N, key_len, self.heads, self.embed_size // self.heads)
        queries = queries.view(N, query_len, self.heads, self.embed_size // self.heads)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.nn.functional.softmax(energy, dim=3)

        k = int((1 - self.sparsity) * attention.shape[-1])
        if k <= 0:
            raise ValueError("k value is non-positive, check sparsity and attention dimensions.")
        topk, _ = attention.topk(k, dim=-1)
        min_topk = topk[:, :, :, -1].unsqueeze(-1).expand_as(attention)
        sparse_attention = torch.where(attention < min_topk, torch.zeros_like(attention), attention)

        out = torch.einsum("nhql,nlhd->nqhd", [sparse_attention, values]).reshape(
            N, query_len, self.embed_size
        )
        out = self.fc_out(out)
        return out
