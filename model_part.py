from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.net = PreNorm(in_dim, FeedForward(in_dim, in_dim // 2, out_dim, dropout=dropout))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.net = PreNorm(in_dim, FeedForward(in_dim, in_dim * 2, out_dim, dropout=dropout))

    def forward(self, x):
        return self.net(x)
