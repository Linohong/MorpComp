def getData(filename, input_lang, output_lang) :
    num_sent = 0
    corpus = []
    finished = open('../data/done.txt').read().strip()
    print(finished)

    for file in filename :
        if (file in finished) :
            continue

        lines = open('../data/%s' % (file)).read().strip().split('\n')
        cur_sent = []
        skip_flag = 0

        for line in lines :
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
                continue
            elif ('<title>' in line ) :
                continue

            # cur_word = [이루어져, [(이루어지, VV), (어, EC)]]
            print(line)
            cur_word = line.split('\t')[1:]
            if '+' in cur_word[0] :
                skip_flag = 1
                continue
            morp_word = cur_word[1].split('+')
            morp_word = [morp.strip() for morp in morp_word]
            morp_word = [(morp.split('/')[0], morp.split('/')[1]) for morp in morp_word]
            cur_word = [cur_word[0], morp_word]
            input_lang.addWord(cur_word[1])
            output_lang.addWord(cur_word[0])

            cur_sent.append(cur_word)


    print("The Number of Sentence : %d" % num_sent)
    return corpus



