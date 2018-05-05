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
for eval_iter in range(Args.args.eval_iter) :
    start_time = time.time()
    print("\nLoading Test Data...")
    import dataProcess.ReadFromFile as D_read
    import dataProcess.Make_ExamplePair as D_pair


    # Reading Data from proper place according to the task
    # select random data if True
    path = '../data/train'
    trained = open('../data/test/' + Args.args.model_name + '_trained.txt').read().strip()
    allfiles = []
    for file in os.listdir(path) :
        allfiles.append(path + '/' + file)
    allfiles = random.sample(allfiles, len(allfiles)) # shuffle
    filenames = [] # filenames of files which will actually be read
    count = 0
    for filename in allfiles :
        if count == Args.args.files_to_read :
            break

        if Args.args.task == 'test' and filename not in trained :
            filenames.append(filename)
            count += 1
        elif Args.args.task == 'closed_test' and filename in trained :
            filenames.append(filename)
            count += 1


    if (True) : # if True, use designated data
        filenames = ['../data/train/BTHO0378.txt', '../data/train/BTGO0351.txt', '../data/train/BTEO0077.txt',
                     '../data/train/BTAA0013.txt', '../data/train/BTEO0338.txt', '../data/train/BTHO0415.txt',
                     '../data/train/BTAA0155.txt', '../data/train/BTHO0432.txt', '../data/train/BTAE0201.txt']
        # above filenames are for fixed test data for SGD10s with length variants

    # Load Vocabs
    with open('ModelWeights/vocab/' + Args.args.model_name + '_in.p', 'rb') as fp :
        input_lang = pickle.load(fp)
    with open('ModelWeights/vocab/' + Args.args.model_name + '_out.p', 'rb') as fp :
        output_lang = pickle.load(fp)

    if ( Args.args.exam_unit == 'sent' ) :
        corpus = D_read.getData(filenames, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
    elif ( Args.args.exam_unit == 'word') :
        corpus, max_word_length = D_read.getDataWordUnit(filenames, input_lang, output_lang)
    #Args.args.max_sent = max_word_length
    print("Done Loading!!!")





    if (Args.args.exam_unit == 'sent') :
        input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
    elif ( Args.args.exam_unit == 'word') :
        input_sent, output_sent, pairs = D_pair.MakePairWordUnit(corpus, input_lang, output_lang)
    input_sent = input_sent[:Args.args.eval_size]
    output_sent = output_sent[:Args.args.eval_size]
    batch_size = Args.args.batch_size
    input_sent = torch.LongTensor(input_sent)
    output_sent = torch.LongTensor(output_sent)

    #*********************************#
    #******* Evaluation Part *********#
    #*********************************#
    if (Args.args.model == 'vanilla') :
        EncNet = torch.load('ModelWeights/vanilla/Enc_' + Args.args.model_name)
        DecNet = torch.load('ModelWeights/vanilla/Dec_' + Args.args.model_name)
    else :
        EncNet = torch.load('./saveEntireEnc_Attn1')
        DecNet = torch.load('./saveEntireDec_Attn1')

    total, got_right = EvalIters(input_sent, output_sent, EncNet, DecNet, input_lang, output_lang)


    print("END OF EVALUATION OF : %s" % Args.args.model_name)
    print("task : %s" % Args.args.task)
    print("got right : %d  / total : %d" % (got_right, total))
    print("Accuracy : %.2f%%" % (100*got_right/total))
    PT.printTime(start_time)
    print('test sets are listed below')
    for filename in filenames :
        print(filename)
