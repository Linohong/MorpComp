import os
import torch
from torch.autograd import Variable
import time
import dataProcess.Make_ExamplePair as D
import Arguments as Args

def Eval(input_sent, target_sent, EncNet, DecNet, input_lang, output_lang) :
    got_right = 0
    input_length = input_sent.size()[0]
    target_length = target_sent.size()[0]
    max_sent = Args.args.max_sent

    # Encoder Part #
    enc_hidden = EncNet.initHidden()  # initialized hidden Variable.
    enc_outputs = Variable(
        torch.zeros(max_sent + 1, EncNet.hidden_size * 2))  # zeros of input_length+1 * EncNet (+1 for EOS tagging)
    enc_outputs = enc_outputs if Args.args.no_gpu else enc_outputs.cuda()

    enc_fw_output, enc_fw_hidden = EncNet(input_sent, enc_hidden, 'forward')
    enc_bw_output, enc_bw_hidden = EncNet(input_sent, enc_hidden, 'backward')
    for ei in range(input_length):
        enc_outputs[ei] = torch.cat((enc_fw_output[ei], enc_bw_output[ei]), 1)[0]  # 1500 x 600

    Enc_String = []
    for index in input_sent :
        Enc_String.append(input_lang.index2syll[int(index.data)])
    print("Enc_String Below")
    print(Enc_String)

    # Decoder Part #
    dec_hidden_2nd = dec_hidden = enc_hidden  # initialize decoder's hidden state as enc_hidden state
    dec_input_2nd = dec_input = Variable(torch.LongTensor([[D.SOS_token]]))  # start of the input with SOS_token
    dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()
    dec_input_2nd = dec_input_2nd if Args.args.no_gpu else dec_input_2nd.cuda()

    Dec_String = []
    for di in range(target_length):
        dec_output, dec_hidden, dec_attention = DecNet(dec_input, dec_hidden, enc_outputs, '1st', dec_hidden_2nd)
        dec_output, dec_hidden_2nd, dec_attention = DecNet(dec_hidden, dec_hidden_2nd, enc_outputs, '2nd',
                                                           dec_hidden_2nd)
        topv, topi = dec_output.data.topk(1)  # topk returns a tuple of (value, index)
        ni = topi[0][0]  # next input
        Dec_String.append(output_lang.index2syll[ni])

        dec_input = Variable(torch.LongTensor([[ni]]))
        dec_input = dec_input if Args.args.no_gpu else dec_input.cuda()

        if ni == D.EOS_token:
            break

    print("Dec_String Below")
    print(Dec_String)
    return got_right

def EvalIters(training_pairs, EncNet, DecNet, input_lang, output_lang) :
    start = time.time()

    for training_pair in training_pairs :
        input_sent_variable = training_pair[0] # Variable of indexes of input sentence
        target_sent_variable = training_pair[1] # Variable of indexes of target sentence
        time.sleep(4)
        Eval(input_sent_variable, target_sent_variable, EncNet, DecNet, input_lang, output_lang)


#********************************#
#******* Load DATA Part *********#
#********************************#
print("\nLoading Test Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang
path = '../data/test'
filename = []
for file in os.listdir(path) :
    filename.append(file)

input_lang = Lang.Lang('morp_decomposed')
output_lang = Lang.Lang('morp_composed')
corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
print("Done Loading!!!")

input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
training_pairs = [D_pair.variableFromPair(pairs[i]) for i in range(len(input_sent))] # now returned as Variable of indexes

#*********************************#
#******* Evaluation Part *********#
#*********************************#
EncNet = torch.load('./saveEntireEnc_Attn_Complete')
DecNet = torch.load('./saveEntireDec_Attn_Complete')
EvalIters(training_pairs, EncNet, DecNet, input_lang, output_lang)
