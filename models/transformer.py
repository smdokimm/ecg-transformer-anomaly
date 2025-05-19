import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


def get_sinusoidal_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.q_linear = MyLinear(d_model, d_model)
        self.k_linear = MyLinear(d_model, d_model)
        self.v_linear = MyLinear(d_model, d_model)
        self.out_linear = MyLinear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, L, D = query.size()
        Q = self.q_linear(query).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_linear(key).view   (B, L, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_linear(value).view (B, L, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.out_linear(out)

class TransformerEncoderLayerUnbundled(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1  = nn.Dropout(0.0) 
        self.norm1     = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            MyLinear(d_model, ff_dim),
            nn.ReLU(),
            MyLinear(ff_dim, d_model)
        )
        self.dropout2  = nn.Dropout(0.0)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class ECGTransformerAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_projection = MyLinear(config["in_channels"], config["d_model"])
        self.dropout_emb      = nn.Dropout(0.2)

        pos_encoding = get_sinusoidal_encoding(config["seq_len"], config["d_model"])
        self.register_buffer("pos_embedding", pos_encoding)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerUnbundled(
                config["d_model"],
                config["nhead"],
                config["ff_dim"],
                dropout=0.0
            ) for _ in range(config["num_layers"])
        ])

        self.decoder = nn.Sequential(
            MyLinear(config["d_model"], config["ff_dim"]),
            nn.ReLU(),
            MyLinear(config["ff_dim"], config["in_channels"])
        )

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        x = self.input_projection(x)
        x = self.dropout_emb(x)
        x = x + self.pos_embedding[:, : x.size(1), :]  # type: ignore[index]
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x)