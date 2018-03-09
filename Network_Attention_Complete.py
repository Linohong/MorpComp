# Encoder Network
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import Arguments as Args
import math
import numpy as np

class EncoderRNN(nn.Module) :
    '''
        this version of EncoderRNN 
        has bi-directional GRU of which the hidden states are concatenated for 
        further usage by attention. 
    '''
    def __init__ (self, vocab_size, hidden_size) :
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru_bw = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, direction) :
        if ( direction == 'forward' ) :
            embedded = self.embedding(input).view(len(input), 1, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
        else :
            input = Variable(torch.from_numpy( np.flip(input.data.cpu().numpy(), axis=0).copy() ).long())
            if Args.args.no_gpu == False :
                input = input.cuda()
            embedded = self.embedding(input).view(len(input), 1, -1)
            output = embedded
            output, hidden = self.gru_bw(output, hidden)

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
        self.v = nn.Parameter(torch.rand(hidden_size * 4))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, embedded) :
        '''
        :param hidden:
            previous hidden state of the decoder of the second floor, in shape (1, 1, hidden_size)
        :param encoder_outputs: 
            outputs of the encoder, in shape (max_sent, hidden_size * 2) 
        :return: 
            attention weight in shape (max_sent)
        '''
        max_sent = self.max_sent
        hidden = torch.cat((hidden, embedded), 2) # 1 * 1 * 600
        H = hidden.repeat(max_sent, 1, 1).squeeze(1) # (1500, 600)
        attn_energies = self.score(H, encoder_outputs) # 1500 * 1
        return F.softmax(attn_energies, dim=0).transpose(0,1) # return 1 * 1500 of attention weights

    def score(self, hidden, enc_outputs) :
        energies = F.tanh(torch.cat([hidden, enc_outputs], 1)) # 1500 * 1200
        energies = energies.transpose(1,0) # 300 * 1500
        v = self.v.view(1,1,-1)
        energies = torch.bmm(v, energies.unsqueeze(0)).squeeze(0) # [1 * 1200] * [1200* 1500]
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
        self.gru_2nd = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.attn = Attention(self.hidden_size, self.max_length) # attention class defined
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, enc_outputs, floor, hidden_2nd) :
        if (floor == '1st') :
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)
        else :
            embedded = self.dropout(input)

        attn_weights = self.attn(hidden_2nd, enc_outputs, embedded).view(1,1,-1) # 1 * 1500
        context = attn_weights.bmm(enc_outputs.unsqueeze(0)) # [1 * 1500] * [1500*600] = 1 * 600
        output = self.fc(context)

        if ( floor == '1st' ) :
            output, hidden = self.gru(output, hidden)
        else :
            output, hidden = self.gru_2nd(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if not Args.args.no_gpu:
            return result.cuda()
        else:
            return result
