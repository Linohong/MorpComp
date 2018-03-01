def getData(filename, input_lang, output_lang) :
    lines = open('../data/%s.txt' % (filename)).read().strip().split('\n')
    corpus = []
    cur_sent = []
    for line in lines :
        if ('<p>' in line) :
            cur_sent = []
            continue
        elif ('</p>' in line ) :
            corpus.append(cur_sent)
            continue
        elif ('<title>' in line ) :
            continue

        # cur_word = [이루어져, [(이루어지, VV), (어, EC)]]
        cur_word = line.split('\t')[1:]
        morp_word = cur_word[1].split('+')
        morp_word = [morp.strip() for morp in morp_word]
        morp_word = [(morp.split('/')[0], morp.split('/')[1]) for morp in morp_word]
        cur_word = [cur_word[0], morp_word]
        input_lang.addWord(cur_word[1])
        output_lang.addWord(cur_word[0])

        cur_sent.append(cur_word)

    return corpus

corpus = getData('BTHO0138')



