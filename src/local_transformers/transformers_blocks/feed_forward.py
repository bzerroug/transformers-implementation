from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim, factor=2):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, factor * embed_dim),
            nn.ReLU(),
            nn.Linear(factor * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.feed_forward(x)
