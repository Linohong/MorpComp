import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import Arguments as Args
import dataProcess.Make_ExamplePair as D
import etc.peripheralTools as PT


def Train(input_sent, target_sent, EncNet, DecNet, enc_optim, dec_optim, criterion, max_length=Args.args.max_sent) :
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    loss = 0
    input_length = input_sent.size()[0]
    target_length = target_sent.size()[0]

    # Encoder Part #
    enc_hidden = EncNet.initHidden() # initialized hidden Variable.
    enc_outputs = Variable(torch.zeros(max_length+1, EncNet.hidden_size * 2)) # zeros of input_length+1 * EncNet (+1 for EOS tagging)
    enc_outputs = enc_outputs if Args.args.no_gpu else enc_outputs.cuda()

    enc_fw_output, enc_fw_hidden = EncNet(input_sent, enc_hidden, 'forward')
    enc_bw_output, enc_bw_hidden = EncNet(input_sent, enc_hidden, 'backward')
    for ei in range(input_length) :
        enc_outputs[ei] = torch.cat((enc_fw_output[ei], enc_bw_output[ei]), 1)[0] # 1500 x 600

    # Decoder Part #
    dec_hidden_2nd = dec_hidden = enc_hidden # initialize decoder's hidden state as enc_hidden state
    dec_input_2nd = dec_input = Variable(torch.LongTensor([[D.SOS_token]])) # start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()
    dec_input_2nd = dec_input_2nd if Args.args.no_gpu else dec_input_2nd.cuda()

    for di in range(target_length) :
        dec_output, dec_hidden, dec_attention = DecNet(dec_input, dec_hidden, enc_outputs, '1st', dec_hidden_2nd)
        dec_output, dec_hidden_2nd, dec_attention = DecNet(dec_hidden, dec_hidden_2nd, enc_outputs, '2nd', dec_hidden_2nd)
        topv, topi = dec_output.data.topk(1) # topk returns a tuple of (value, index)
        ni = topi[0][0] # next input

        dec_input = Variable(torch.LongTensor([[ni]]))
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        loss += criterion(dec_output, target_sent[di])
        if ni == D.EOS_token :
            break

    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.data[0] / target_length

def TrainIters(train_index, training_pairs, EncNet, DecNet, trainSize, print_every=1000, epoch_size=10, batch_size=50, lr=0.02) :
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    inter_loss = 0

    encoder_optimizer = optim.SGD(EncNet.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(DecNet.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    iter_time = 1
    for iter in train_index :
        training_pair = training_pairs[iter]
        input_sent_variable = training_pair[0] # Variable of indexes of input sentence
        target_sent_variable = training_pair[1] # Variable of indexes of target sentence

        loss = Train(input_sent_variable, target_sent_variable, EncNet, DecNet, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        inter_loss += loss

        iter_time = iter_time + 1

        if( iter % print_every-1 == 0 ) :
            print("[%d] iteration : loss = %.4f" % (iter, inter_loss/print_every))
            inter_loss = 0

    print_loss_avg = print_loss_total/len(train_index)
    print('%s (%d itered) %.4f' % (PT.timeSince(start, float(iter_time)/trainSize), iter_time, print_loss_avg ))


