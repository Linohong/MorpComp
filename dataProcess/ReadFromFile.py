# -*- coding: utf-8 -*-
import os
import Arguments as Args
import hanja

def getFilenames() :
    path = '../data/train'
    if (Args.args.task == 'train'):
        path = '../data/train'
    elif (Args.args.task == 'closed_test'):
        path = '../data/closed_test'
    elif (Args.args.task == 'test'):
        path = '../data/test'
    else:
        path = '../data/experiment'

    filenames = []
    for file in os.listdir(path):
        filenames.append(path + '/' + file)

    return sorted(filenames)

def getDataWordUnit(filename, input_lang, output_lang) :
    max_word_length = 0
    num_words = 0
    corpus = []
    finished = open('../data/train/done.txt').read().strip()

    for file in filename:
        cur_file_sent = 0
        if (file in finished):
            continue

        lines = open(file).read().strip().split('\n')

        cur_sent = [] # BUT WORD UNIT THIS TIME
        skip_flag = 0

        for i, line in enumerate(lines):
            if ('<p>' in line or '<head>' in line or '<l>' in line) :
                continue
            elif ('</p>' in line or '</head>' in line or '</l>' in line):
                continue
            elif ('<title>' in line):
                continue

            # split if right type of word
            cur_word_length = 0
            cur_word = line.split('\t')[1:]
            if '+' in cur_word[0] or len(cur_word) != 2:  # Exception handling for wrong input
                continue

            if skip_flag == 0:  # Exception handling for hanja
                if (hasJapChn(cur_word)):
                    continue

            morp_word = cur_word[1].split('+')
            morp_word = [morp.strip() for morp in morp_word]  # morp_word = ['이루어지/VV', '어/EC']
            try:
                morp_word = [(morp.split('/')[0], morp.split('/')[1]) for morp in morp_word]
            except IndexError:
                continue
            cur_word = [cur_word[0], morp_word]  # cur_word = [이루어져, [(이루어지, VV), (어, EC)]]
            if (Args.args.task != 'test' and Args.args.task != 'closed_test'):
                input_lang.addWord(cur_word[1], Args.args.enc_unit)
                output_lang.addWord(cur_word[0], Args.args.dec_unit)

            corpus.append(cur_word)
            num_words += 1
            # calculate length
            for morp in morp_word :
                cur_word_length += len(morp[0]) + 1
            if ( max_word_length <= cur_word_length) :
                max_word_length = cur_word_length
        print(file)
        # print("Current File[%s] has %d sentences read" % (file, cur_file_sent))

    print("The Number of Words Exists in all of the files : %d" % num_words)
    return corpus, max_word_length


def getData(filename, input_lang, output_lang) :
    num_sent = 0
    corpus = []
    finished = open('../data/train/done.txt').read().strip()

    for file in filename :
        cur_file_sent = 0
        if (file in finished) :
            continue

        lines = open(file).read().strip().split('\n')

        cur_sent = []
        skip_flag = 0

        for i, line in enumerate(lines) :
            if skip_flag :
                if ('</p>' in line or '</head>' in line or '</l>' in line) :
                    skip_flag = 0
                continue
            if ('<p>' in line or '<head>' in line or '<l>' in line) :
                num_sent += 1
                cur_sent = []
                continue
            elif ('</p>' in line or '</head>' in line or '</l>' in line) :
                corpus.append(cur_sent)
                cur_file_sent += 1
                continue
            elif ('<title>' in line ) :
                continue

            # Exception handling
            cur_word = line.split('\t')[1:]
            if '+' in cur_word[0] or len(cur_word) != 2 : # Exception handling for wrong input
                skip_flag = 1
                continue

            if skip_flag == 0 : # Exception handling for hanja
                if ( hasJapChn(cur_word) ) :
                    skip_flag = 1
                    continue

            morp_word = cur_word[1].split('+')
            morp_word = [morp.strip() for morp in morp_word] # morp_word = ['이루어지/VV', '어/EC']
            try :
                morp_word = [(morp.split('/')[0], morp.split('/')[1]) for morp in morp_word]
            except IndexError :
                skip_flag = 1
                continue
            cur_word = [cur_word[0], morp_word] # cur_word = [이루어져, [(이루어지, VV), (어, EC)]]
            if ( Args.args.task != 'test' and Args.args.task != 'closed_test' ) :
                input_lang.addWord(cur_word[1], Args.args.enc_unit)
                output_lang.addWord(cur_word[0], Args.args.dec_unit)

            cur_sent.append(cur_word)
        print(file)
        #print("Current File[%s] has %d sentences read" % (file, cur_file_sent))

    print("The Number of Sentence Exists in all of the files : %d" % num_sent)
    return corpus

def hasJapChn(cur_word) :
    # cur_word = [이루어져, [(이루어지, VV), (어, EC)]]
    for char in cur_word[0] :
        if (hanja.is_hanja(char)) :
            return True

    for input_list in cur_word[1] :
        for char in input_list[0] :
            if (hanja.is_hanja(char)) :
                return True

    return False







