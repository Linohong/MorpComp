# Encoder Network
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import Arguments as Args
import math

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

class Attention(nn.Module) :
    def __init__ (self, hidden_size, max_sent) :
        super(Attention, self).__init__()
        self.max_sent = max_sent
        self.fc = nn.Linear(hidden_size * 2, hidden_size) # linear transformation of concatenated vectors of decoder's previous hidden state + current input
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs) :
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (1, 1, hidden_size)
        :param encoder_outputs: 
            outputs of the encoder, in shape (max_sent, hidden_size) 
        :return: 
            attention weight in shape (max_sent)
        '''
        max_sent = self.max_sent
        H = hidden.repeat(max_sent,1, 1).squeeze(1) # (1500, 300)
        attn_energies = self.score(H, encoder_outputs) # 1500 * 1
        return F.softmax(attn_energies, dim=0).transpose(0,1) # return 1 * 1500 of attention weights

    def score(self, hidden, enc_outputs) :
        energies = F.tanh(self.fc(torch.cat([hidden, enc_outputs], 1))) # 1500 * 300 => 1500 * 600 => 1500 * 300
        energies = energies.transpose(1,0) # 300 * 1500
        v = self.v.view(1,1,-1)
        energies = torch.bmm(v, energies.unsqueeze(0)).squeeze(0) # [1 * 300] * [300* 1500]
        return energies.transpose(1,0) # 1500 * 1 of scores

class DecoderRNN(nn.Module) :
    def __init__ (self, vocab_size, hidden_size, dropout_p=Args.args.dropout) :
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = Args.args.max_sent

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = Attention(self.hidden_size, self.max_length) # attention class defined
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, enc_outputs) :
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = self.attn(hidden, enc_outputs).view(1,1,-1) # 1 * 1500
        context = attn_weights.bmm(enc_outputs.unsqueeze(0)) # [1 * 1500] * [1500*300] = 1 * 300

        output = self.fc(torch.cat((embedded, context), 2).squeeze(0)).unsqueeze(0)
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
