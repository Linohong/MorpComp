import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import Arguments as Args
import dataProcess.Make_ExamplePair as D
import etc.peripheralTools as PT

def Train(batch_sents, batch_labels, EncNet, DecNet, enc_optim, dec_optim, criterion, out_lang, max_sent=Args.args.max_sent) :
    enc_optim.zero_grad()
    dec_optim.zero_grad()
    cur_batch_size = len(batch_sents)
    batch_size = Args.args.batch_size

    # Encoder Part #
    enc_hidden = EncNet.initHidden(len(batch_sents)) # initialized hidden Variable.
    batch_sents = batch_sents.transpose(0,1)
    enc_output, enc_hidden = EncNet(batch_sents, enc_hidden) # enc_output :   //// enc_hidden : 1 x cur_batch_size x hidden_size

    # Decoder Part #
    loss = 0
    batch_labels = batch_labels.transpose(0,1) # transpose in order to get the loss
    dec_hidden = enc_hidden # initialize decoder's hidden state as enc_hidden state => 1 * 32 * 256 (1 * B * H)
    dec_input = Variable(torch.LongTensor([D.SOS_token] * cur_batch_size)).view(1, cur_batch_size) # start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

    test_predict = [[], [], []]
    test_label = [[], [], []]
    for di in range(max_sent) :
        dec_output, dec_hidden = DecNet(dec_input, dec_hidden)
        topv, topi = dec_output.data.topk(1) # topk returns a tuple of (value, ind`ex)

        dec_input = Variable(torch.cat(topi)).view(1, cur_batch_size)
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        loss += criterion(dec_output, batch_labels[di])

        if cur_batch_size != batch_size :
            for i in range(3) :
                test_predict[i].append(out_lang.index2syll[int(topi[i])])
                test_label[i].append(out_lang.index2syll[int(batch_labels[di][i])])

    if cur_batch_size != batch_size :
        for i in range(3) :
            print('PREDICTION :')
            print(test_predict[i])
            print('LABEL :')
            print(test_label[i])

    loss.backward()
    enc_optim.step()
    dec_optim.step()

    return loss.data[0] / max_sent

def TrainIters(trainloader, EncNet, DecNet, trainSize, out_lang, print_every=2048, epoch_size=10, batch_size=32, lr=0.0001) :
    start = time.time()
    print_loss_total = 0
    inter_loss = 0

    if (Args.args.optim == 'SGD') :
        encoder_optimizer = optim.SGD(EncNet.parameters(), lr=lr)
        decoder_optimizer = optim.SGD(DecNet.parameters(), lr=lr)
    elif (Args.args.optim == 'Adam') :
        encoder_optimizer = optim.Adam(EncNet.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(DecNet.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    num_items = 0
    for iter, (batch_sents, batch_labels) in enumerate(trainloader) :
        if ( Args.args.no_gpu == False ) :
            batch_sents = Variable(batch_sents.cuda())
            batch_labels = Variable(batch_labels.cuda())
        else :
            batch_sents = Variable(batch_sents)
            batch_labels = Variable(batch_labels)

        cur_batch_size = len(batch_sents)

        loss = Train(batch_sents, batch_labels, EncNet, DecNet, encoder_optimizer, decoder_optimizer, criterion, out_lang)
        num_items += cur_batch_size
        print_loss_total += loss * cur_batch_size
        inter_loss += loss * cur_batch_size

        if ( (num_items * batch_size) % (print_every-1) == 0 ) :
            print("[%d] batches, [%d] items passed : latest [%d] average loss = %.4f" % (iter, num_items, print_every, inter_loss/print_every))
            inter_loss = 0

    print_loss_avg = print_loss_total/num_items
    print('%s (%d itered) %.4f' % (PT.timeSince(start, float(num_items)/trainSize), num_items, print_loss_avg ))

    return print_loss_avg

def ZeroWeight(Net, embed_size) :
    import numpy as np
    zero_row = np.array([0] * embed_size)
    Net.embedding.weight[D.ZERO_token].data.copy_(torch.from_numpy(zero_row))

def EarlyStopping(prev_loss, loss, early) :
    if loss > prev_loss:
        early -= 1
    else:
        early = Args.args.early

    if early == 0 :
        return True
    else :
        return False
