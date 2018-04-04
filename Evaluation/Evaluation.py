import os
import torch
from torch.autograd import Variable
import time
import dataProcess.Make_ExamplePair as D
import Arguments as Args
import pickle

def Compare_Dec_Label(Dec, Lab) :
    total = 0
    got_right = 0

    Lab_indexes = [i for i in range(len(Lab)) if 'SPACE' in Lab[i]]
    Dec_indexes = [i for i in range(len(Dec)) if 'SPACE' in Dec[i]]
    Lab_indexes = [0] + Lab_indexes
    Dec_indexes = [0] + Dec_indexes

    for i in range(len(Lab_indexes)-1) :
        total += 1
        try :
            if Lab[Lab_indexes[i]:Lab_indexes[i+1]] == Dec[Dec_indexes[i]:Dec_indexes[i+1]] :
                got_right += 1
        except IndexError :
            if Lab == Dec :
                got_right += 1

    return total, got_right


def Eval(input_sent, target_sent, EncNet, DecNet, input_lang, output_lang) :
    got_right = 0
    input_length = input_sent.size()[0]
    target_length = target_sent.size()[0]

    # Encoder Part #
    enc_hidden = EncNet.initHidden() # initialized hidden Variable.
    enc_outputs = Variable(torch.zeros(input_length, EncNet.hidden_size)) # zeros of input_length * EncNet
    enc_outputs = enc_outputs if Args.args.no_gpu else enc_outputs.cuda()

    for ei in range(input_length) :
         enc_output, enc_hidden = EncNet(input_sent[ei], enc_hidden)
         enc_outputs[ei] = enc_output[0][0]

    Enc_String = []
    for index in input_sent :
        Enc_String.append(input_lang.index2syll[int(index.data)])

    # Decoder Part #
    dec_hidden = enc_hidden # initialize decoder's hidden state as enc_hidden state
    dec_input = Variable(torch.LongTensor([[D.SOS_token]])) # start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

    Dec_String = []
    for di in range(target_length) :
        dec_output, dec_hidden = DecNet(dec_input, dec_hidden)
        topv, topi = dec_output.data.topk(1) # topk returns a tuple of (value, index)
        ni = topi[0][0] # next input
        Dec_String.append(output_lang.index2syll[ni])

        dec_input = Variable(torch.LongTensor([[ni]]))
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        if ni == D.EOS_token :
            break

    Label_String = []
    for index in target_sent:
        Label_String.append(output_lang.index2syll[int(index.data)])


    total, got_right = Compare_Dec_Label(Dec_String, Label_String)
    return total, got_right

def EvalIters(training_pairs, EncNet, DecNet, input_lang, output_lang) :
    start = time.time()
    total = got_right = 0
    t = g = 0

    for training_pair in training_pairs :
        input_sent_variable = training_pair[0] # Variable of indexes of input sentence
        target_sent_variable = training_pair[1] # Variable of indexes of target sentence
        t, g = Eval(input_sent_variable, target_sent_variable, EncNet, DecNet, input_lang, output_lang)
        total += t
        got_right += g

    return total, got_right


#********************************#
#******* Load DATA Part *********#
#********************************#
print("\nLoading Test Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair


path = '../data/train'
if (Args.args.task == 'train'):
    path = '../data/train'
elif (Args.args.task == 'closed_test'):
    path = '../data/test'
elif (Args.args.task == 'test'):
    path = '../data/test_real'
else:
    path = '../data/experiment'

filename = []
for file in os.listdir(path) :
    filename.append(path + '/' + file)

# Load Vocabs
with open('input_lang30sent+moreData.p', 'rb') as fp :
    input_lang = pickle.load(fp)
with open('output_lang30sent+moreData.p', 'rb') as fp :
    output_lang = pickle.load(fp)

corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
print("Done Loading!!!")

input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
training_pairs = [D_pair.variableFromPair(pairs[i]) for i in range(len(input_sent))] # now returned as Variable of indexes

#*********************************#
#******* Evaluation Part *********#
#*********************************#
if (Args.args.model == 'vanilla') :
    EncNet = torch.load('./Enc')
    DecNet = torch.load('./Dec')
else :
    EncNet = torch.load('./saveEntireEnc_Attn1')
    DecNet = torch.load('./saveEntireDec_Attn1')

total, got_right = EvalIters(training_pairs, EncNet, DecNet, input_lang, output_lang)
print("got right : %d  / total : %d" % (got_right, total))
print("Accuracy : %.2f%%" % (100*got_right/total))
