# Encoder Network
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import Arguments as Args

class EncoderRNN(nn.Module) :
    def __init__ (self, vocab_size, hidden_size) :
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden) :
        embedded = self.embedding(input).view(len(input), 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self) :
        result = Variable(torch.zeros(1, 1, self.hidden_size)) # initialize with zeros as hidden_size.
        if not Args.args.no_gpu :
            return result.cuda()
        else :
            return result

class DecoderRNN(nn.Module) :
    def __init__ (self, vocab_size, hidden_size, dropout_p=Args.args.dropout) :
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = Args.args.max_sent

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, enc_outputs) :
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), enc_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        #output, hidden = self.gru(output, hidden) # gru(input, h_0) : input=>(seq_len, batch, input_size)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if not Args.args.no_gpu:
            return result.cuda()
        else:
            return result
