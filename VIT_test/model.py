import torch
from torch import nn


class Patch2Embedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(Patch2Embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn((1, num_patches+1, embed_dim)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls = self.cls.expand(x.shape[0], -1, -1)
        # test = x
        # print(self.conv(test).shape, self.conv(test).flatten(2).shape)
        x = self.conv(x).flatten(2).transpose(-2, -1)
        x = torch.cat([x, cls], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dk = embed_dim // num_heads
        self.Norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3, num_heads, dk] -> [3, B, num_heads, N, dk]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k , v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) / self.dk**-0.5
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.dropout(self.proj(x))
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(Block, self).__init__()
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, embed_dim*4, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x

class vit(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, num_encoder, num_class):
        super(vit, self).__init__()
        self.patch = Patch2Embedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, dropout) for _ in range(num_encoder)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.patch(x)
        x = self.blocks(x)
        x = self.proj(self.norm(x[:,0,:]))
        return x


















