import os
import random
import torch
from torch.autograd import Variable
import time
import dataProcess.Make_ExamplePair as D
import Arguments as Args
import pickle
import etc.peripheralTools as PT

def Compare_Dec_Label(Dec, Lab) :
    total = 0
    got_right = 0

    Lab_indexes = [i for i in range(len(Lab)) if 'SPACE' in Lab[i] or 'EOS' in Lab[i]]
    Dec_indexes = [i for i in range(len(Dec)) if 'SPACE' in Dec[i] or 'EOS' in Dec[i]]
    Lab_indexes = [0] + Lab_indexes
    Dec_indexes = [0] + Dec_indexes

    if Args.args.exam_unit == 'sent' :
        for i in range(len(Lab_indexes)-1) :
            total += 1
            try :
                if Lab[Lab_indexes[i]:Lab_indexes[i+1]] == Dec[Dec_indexes[i]:Dec_indexes[i+1]] :
                    got_right += 1
            except IndexError :
                if Lab == Dec :
                    got_right += 1

    if Args.args.exam_unit == 'word' :
        total += 1
        # Dec : ['을', '해', 'EOS']
        # Lab : ['을', '해', 'EOS', 'ZERO', 'ZERO', ... , 'ZERO']
        if Lab[:Lab.index('EOS')+1] == Dec :
            got_right += 1


    return total, got_right


def Eval(input_sent, target_sent, EncNet, DecNet, input_lang, output_lang) :
    got_right = 0

    # Encoder Part #
    enc_hidden = EncNet.initHidden(1) # initialized hidden Variable.
    input_sent = input_sent
    enc_output, enc_hidden = EncNet(input_sent, enc_hidden)

    Enc_String = []
    for index in input_sent :
        Enc_String.append(input_lang.index2syll[int(index.data)])

    # Decoder Part #
    dec_hidden = enc_hidden  # initialize decoder's hidden state as enc_hidden state => 1 * 32 * 256 (1 * B * H)
    dec_input = Variable(torch.LongTensor([D.SOS_token] * 1)).view(1,-1)  # start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

    Dec_String = []
    for di in range(Args.args.max_sent):
        dec_output, dec_hidden = DecNet(dec_input, dec_hidden)
        topv, topi = dec_output.data.topk(1)  # topk returns a tuple of (value, index)

        ni = topi[0][0] # next input
        Dec_String.append(output_lang.index2syll[ni])
        dec_input = Variable(torch.cat(topi)).view(1,-1)
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        if ni == D.EOS_token :
            break


    Label_String = []
    for index in target_sent:
        Label_String.append(output_lang.index2syll[int(index.data)])

    # print('Enc_String')
    # print(Enc_String)
    # print('Dec_String')
    # print(Dec_String)


    total, got_right = Compare_Dec_Label(Dec_String, Label_String)
    return total, got_right

def EvalIters(input_sent, target_sent, EncNet, DecNet, input_lang, output_lang) :
    start = time.time()
    total = got_right = 0
    t = g = 0

    for i in range(len(input_sent)) :
        in_sent = Variable(input_sent[i].cuda())
        tar_sent = Variable(target_sent[i].cuda())
        t, g = Eval(in_sent, tar_sent, EncNet, DecNet, input_lang, output_lang)
        total += t
        got_right += g
        #time.sleep(5)

    return total, got_right


#********************************#
#******* Load DATA Part *********#
#********************************#
start_time = time.time()
print("\nLoading Test Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair


# Reading Data from proper place according to the task
with open('../../data/test/test_input_' + Args.args.exam_unit + '.txt', 'rb') as fp:  # read output language
    input_sent = pickle.load(fp)
with open('../../data/test/test_output_' + Args.args.exam_unit + '.txt', 'rb') as fp:  # read output language
    output_sent = pickle.load(fp)
with open('../ModelWeights/vocab/' + Args.args.model_name + '_in.p', 'rb') as fp:  # read input language
    input_lang = pickle.load(fp)
with open('../ModelWeights/vocab/' + Args.args.model_name + '_out.p', 'rb') as fp:  # read output language
    output_lang = pickle.load(fp)
print("Done Loading!!!")

# Pack Data
batch_size = Args.args.batch_size
input_sent = torch.LongTensor(input_sent)
output_sent = torch.LongTensor(output_sent)



#*********************************#
#******* Evaluation Part *********#
#*********************************#
if (Args.args.model == 'vanilla') :
    EncNet = torch.load('../ModelWeights/vanilla/Enc_' + Args.args.model_name)
    DecNet = torch.load('../ModelWeights/vanilla/Dec_' + Args.args.model_name)
else :
    EncNet = torch.load('./saveEntireEnc_Attn1')
    DecNet = torch.load('./saveEntireDec_Attn1')

total, got_right = EvalIters(input_sent, output_sent, EncNet, DecNet, input_lang, output_lang)


print("END OF EVALUATION OF : %s" % Args.args.model_name)
print("task : %s" % Args.args.task)
print("got right : %d  / total : %d" % (got_right, total))
print("Accuracy : %.2f%%" % (100*got_right/total))
PT.printTime(start_time)

