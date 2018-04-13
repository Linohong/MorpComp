# Encoder Network
import torch
import torch.nn as nn
from torch.autograd import Variable
import Arguments as Args
import torch.nn.functional as F
import numpy as np

class EncoderRNN(nn.Module) :
    def __init__ (self, vocab_size, embed_size, hidden_size) :
        super(EncoderRNN, self).__init__()
        self.max_sent = Args.args.max_sent
        self.batch_size = Args.args.batch_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size) # batch_first : if True, then input, output -> (batch, seq, feature)

    def forward(self, input, hidden) :
        embedded = self.embedding(input).view(self.max_sent, -1, self.embed_size) # embedded - (batchsize x input length x embedding_dim )
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, size) :
        result = Variable(torch.zeros(1, size, self.hidden_size)) # initialize with zeros as hidden_size.
        if not Args.args.no_gpu :
            return result.cuda()
        else :
            return result

class DecoderRNN(nn.Module) :
    def __init__ (self, vocab_size, embed_size, hidden_size) :
        super(DecoderRNN, self).__init__()
        self.max_sent = Args.args.max_sent
        self.embed_size = embed_size
        self.batch_size = Args.args.batch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden) :
        embedded = self.embedding(input).view(1, -1, self.embed_size)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden) # gru(input, h_0) : input=>(seq_len, batch, input_size)
        output = self.softmax(self.fc(output[0]))
        return output, hidden
