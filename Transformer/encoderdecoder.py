import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from pyitcast.transformer_utils import Batch
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from torch.autograd import Variable



V = 11
batch_size = 10
num_batch = 20

def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(1-subsequent_mask)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)


    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn,value), p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model


def data_generator(V, batch_size, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))
        data[:,0] = 1

        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)

def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings,self).__init__()
        self.lat = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self,x):
        return self.lat(x) * math.sqrt(self.d_model)
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


class MultiHeadedAttention(nn.Module):
    """多头注意力"""
    def __init__(self, head, embedding_dim, dropout = 0.1):
        super(MultiHeadedAttention,self).__init__()

        assert embedding_dim % head == 0, "The number of heads doesn't match the embeading_dim"

        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        self.linears = clones(nn.Linear(embedding_dim,embedding_dim), 4)

        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2) 
             for model,x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.embedding_dim)

        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    """前馈连接层"""
    def __init__(self, dim1, dim2, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim1)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

class LayerNorm(nn.Module):
    """规范化层"""
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(dim))
        self.b2 = nn.Parameter(torch.zeros(dim))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std =  x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection,self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))        
    

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
        x = self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, size, masked_self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.masked_self_attn = masked_self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
        self.size = size

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.masked_self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))
        return self.sublayer[2](x, self.feed_forward)
        

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)

        return self.norm(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embede, generator):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embede
        self.generator = generator

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)
    
    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), target, source_mask, target_mask)
    

class Generator(nn.Module):
    def __init__(self, dim, vocab):
        super(Generator, self).__init__()
        self.project = nn.Linear(dim, vocab)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)