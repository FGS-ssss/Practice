import torch
import math
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab, dim):
        super(Embeddings,self).__init__()
        self.lat = nn.Embedding(vocab, dim)
        self.dim = dim
    
    def forward(self,x):
        return self.lat(x) * math.sqrt(self.dim)
    """
    传入sentences句长length的文本 X.shape = [sentences, length]
    传出 Em.shape = [sentences, length, dims]
    """
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        # dims: dimensions of embeddings
        # dropout: ratio of set zero
        # max_len: maximal length of sentences
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,dim)
        position = torch.arange(0,max_len).unsqueeze(1)
        """position.shape = max_len, 1"""
        div_term = torch.exp(torch.arange(0,dim,2) * -math.log(10000.0) / dim)
        """div_term.shape = 1, dims/2"""
        pe[:, 0::2] = torch.sin(position * div_term)
        print(pe[:,1::2].shape)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        """pe.shape = 1, max_len, dims"""
        self.register_buffer('pe',pe)
    """
    ?why using torch.sin and torch.cos here. Sin and cos might eliminate positional encoding effect.
    """

    def forward(self,x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)