import Arguments as Args

def getData(filename, input_lang, output_lang) :
    num_sent = 0
    corpus = []
    finished = open('../data/train/done.txt').read().strip()

    for file in filename :
        cur_file_sent = 0
        if (file in finished) :
            continue

        if (Args.args.task == 'train') :
            lines = open('../data/train/%s' % (file)).read().strip().split('\n')
        elif (Args.args.task == 'closed_test') :
            lines = open('../data/test/%s' % (file)).read().strip().split('\n')
        else :
            lines = open('../data/test_real/%s' % (file)).read().strip().split('\n')
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

            # cur_word = [이루어져, [(이루어지, VV), (어, EC)]]
            cur_word = line.split('\t')[1:]
            if '+' in cur_word[0] or len(cur_word) != 2 :
                skip_flag = 1
                continue
            morp_word = cur_word[1].split('+')
            morp_word = [morp.strip() for morp in morp_word]
            try :
                morp_word = [(morp.split('/')[0], morp.split('/')[1]) for morp in morp_word]
            except IndexError :
                skip_flag = 1
                continue
            cur_word = [cur_word[0], morp_word]
            input_lang.addWord(cur_word[1])
            output_lang.addWord(cur_word[0])

            cur_sent.append(cur_word)

        print("Current File[%s] has %d sentences read" % (file, cur_file_sent))


    print("The Number of Sentence Exists in all of the files : %d" % num_sent)
    return corpus



