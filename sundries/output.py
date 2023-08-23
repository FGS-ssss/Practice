import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, dim, vocab):
        super(Generator, self).__init__()
        self.project = nn.Linear(dim, vocab)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)