import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import Arguments as Args
import dataProcess.Make_ExamplePair as D
import etc.peripheralTools as PT


def Train(batch_sents, batch_labels, EncNet, DecNet, enc_optim, dec_optim, criterion, max_sent=Args.args.max_sent) :
    enc_optim.zero_grad()
    dec_optim.zero_grad()
    batch_size = Args.args.batch_size
    cur_batch_size = len(batch_sents)

    # Encoder Part #
    enc_hidden = EncNet.initHidden(len(batch_sents)) # initialized hidden Variable.
    enc_output, enc_hidden = EncNet(batch_sents, enc_hidden)

    # Decoder Part #
    loss = 0
    batch_labels = batch_labels.transpose(0,1) # transpose in order to get the loss
    dec_hidden = enc_hidden # initialize decoder's hidden state as enc_hidden state => 1 * 32 * 256 (1 * B * H)
    dec_input = Variable(torch.LongTensor([D.SOS_token] * cur_batch_size)).view(cur_batch_size, -1)# start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

    for di in range(max_sent) :
        dec_output, dec_hidden = DecNet(dec_input, dec_hidden)
        topv, topi = dec_output.data.topk(1) # topk returns a tuple of (value, index)

        #ni = topi[0][0] # next input

        dec_input = Variable(torch.cat(topi))
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        loss += criterion(dec_output, batch_labels[di])
        #if ni == D.EOS_token :
        #    break

    loss.backward()
    enc_optim.step()
    dec_optim.step()

    return loss.data[0] / max_sent

def TrainIters(trainloader, EncNet, DecNet, trainSize, print_every=8000, epoch_size=10, batch_size=32, lr=0.0001) :
    start = time.time()
    print_loss_total = 0
    inter_loss = 0

    encoder_optimizer = optim.SGD(EncNet.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(DecNet.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    iter_time = 0
    for iter, (batch_sents, batch_labels) in enumerate(trainloader) :
        if ( Args.args.no_gpu == False ) :
            batch_sents = Variable(batch_sents.cuda())
            batch_labels = Variable(batch_labels.cuda())
        else :
            batch_sents = Variable(batch_sents)
            batch_labels = Variable(batch_labels)

        loss = Train(batch_sents, batch_labels, EncNet, DecNet, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        inter_loss += loss

        if ( (iter * batch_size) % (print_every-1) == 0 ) :
            print("[%d] batch, [%d] iterations : loss = %.4f" % (iter, iter*batch_size, inter_loss/print_every))
            inter_loss = 0
        iter_time += batch_size

    print_loss_avg = print_loss_total/len(trainloader)
    print('%s (%d itered) %.4f' % (PT.timeSince(start, float(iter_time)/trainSize), iter_time, print_loss_avg ))

    return print_loss_avg

def EarlyStopping(prev_loss, loss, early) :
    if loss > prev_loss:
        early -= 1
    else:
        early = Args.args.early

    if early == 0 :
        return True
    else :
        return False
