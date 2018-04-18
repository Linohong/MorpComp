import torch
from torch.autograd import Variable
import Arguments as Args

SOS_token = 1
EOS_token = 2
SPACE_token = 3
ZERO_token = 0

def ZeroPadding(cur_sent, side) :
    max_sent = Args.args.max_sent # ex) 30 , have to fill all
    count = max_sent - len(cur_sent)
    pad = []
    for i in range(count) :
        pad.append(ZERO_token)

    if ( side == 'front' ) :
        cur_sent = pad + cur_sent
    else :
        cur_sent = cur_sent + pad
    return cur_sent


# make korean sentence into list of indexes of syllables
def tryAddVocab(cur_sent, lang, input, unit) :
    if ( unit == 'syll' ) :
        for syll in input :
            try :
                cur_sent.append(lang.syll2index[syll])
            except KeyError :
                return True # Pass this sentence because of the OOV

    elif ( unit == 'morp' ) :
        try :
            cur_sent.append(lang.syll2index[input])
        except KeyError :
            return True # Pass this sentence because of the OOV

def updateContext(context, cur_word_context, context_length) :
    context += cur_word_context + [SPACE_token]
    context = context[-context_length:]
    return context

def MakePairContextWordUnit(corpus, input_lang, output_lang) :
    input_sent = []
    output_sent = []
    pairs = []

    for sent in corpus :
        pass_this_sent = False
        cur_input_sent = []
        cur_output_sent = []
        priorContext = [ZERO_token, ZERO_token, ZERO_token, ZERO_token, ZERO_token, ZERO_token]
        # Example of word = ['한국의', [('한국', 'NNP'), ('의', 'JKG')]]
        for ind, word in enumerate(sent) :
            if pass_this_sent == True :
                break
            input_word = word[1]
            output_word = word[0]
            cur_word_context = []
            # ADD INPUT INDEXES
            for morp_tuple in input_word : # ('한국', 'NNP'), ('의', 'JKG')
                pass_this_sent = tryAddVocab(cur_input_sent, input_lang, morp_tuple[0], Args.args.enc_unit)
                pass_this_sent = tryAddVocab(cur_input_sent, input_lang, morp_tuple[1], 'morp') # put POS as a whole
                if pass_this_sent is not True :
                    cur_word_context.append(input_lang.syll2index[morp_tuple[1]])

            # ADD prior context
            cur_input_word = priorContext + cur_input_word
            priorContext = updateContext(priorContext, cur_word_context, 6)

            # ADD OUTPUT INDEXES and space if not the end
            pass_this_sent = tryAddVocab(cur_output_sent, output_lang, output_word, Args.args.dec_unit)
            if ( ind != len(sent)-1 ) :
                cur_output_sent.append(output_lang.syll2index['SPACE'])

        # If current sentence is longer than the max_sent length, skip it !
        if ( len(cur_input_sent) < Args.args.max_sent and len(cur_output_sent) < Args.args.max_sent ) :
            cur_input_sent.append(EOS_token)
            cur_output_sent.append(EOS_token)
            cur_input_sent = ZeroPadding(cur_input_sent, 'front')
            cur_output_sent = ZeroPadding(cur_output_sent, 'back')

            input_sent.append(cur_input_sent)
            output_sent.append(cur_output_sent)
            pairs.append([cur_input_sent, cur_output_sent])

    print("Actual Number of sentences read : %d" % len(input_sent))
    return input_sent, output_sent, pairs

def MakePair(corpus, input_lang, output_lang) :
    input_sent = []
    output_sent = []
    pairs = []

    for sent in corpus :
        pass_this_sent = False
        cur_input_sent = []
        cur_output_sent = []
        # Example of word = ['한국의', [('한국', 'NNP'), ('의', 'JKG')]]
        for ind, word in enumerate(sent) :
            if pass_this_sent == True :
                break
            input_word = word[1]
            output_word = word[0]
            # ADD INPUT INDEXES
            for morp_tuple in input_word : # ('한국', 'NNP'), ('의', 'JKG')
                pass_this_sent = tryAddVocab(cur_input_sent, input_lang, morp_tuple[0], Args.args.enc_unit)
                pass_this_sent = tryAddVocab(cur_input_sent, input_lang, morp_tuple[1], 'morp') # put POS as a whole

            # ADD OUTPUT INDEXES
            pass_this_sent = tryAddVocab(cur_output_sent, output_lang, output_word, Args.args.dec_unit)
            if ( ind != len(sent)-1 ) :
                cur_output_sent.append(output_lang.syll2index['SPACE'])

        # If current sentence is longer than the max_sent length, skip it !
        if ( len(cur_input_sent) < Args.args.max_sent and len(cur_output_sent) < Args.args.max_sent ) :
            cur_input_sent.append(EOS_token)
            cur_output_sent.append(EOS_token)
            cur_input_sent = ZeroPadding(cur_input_sent, 'front')
            cur_output_sent = ZeroPadding(cur_output_sent, 'back')

            input_sent.append(cur_input_sent)
            output_sent.append(cur_output_sent)
            pairs.append([cur_input_sent, cur_output_sent])

    print("Actual Number of sentences read : %d" % len(input_sent))
    return input_sent, output_sent, pairs

def MakePairWordUnit(corpus, input_lang, output_lang) :
    input_sent = []
    output_sent = []
    pairs = []
    outtaVocab = 0

    for ind, word in enumerate(corpus) :
        pass_this_word = False
        cur_input_word = []
        cur_output_word = []
        # Example of word = ['한국의', [('한국', 'NNP'), ('의', 'JKG')]]

        input_word = word[1]
        output_word = word[0]

        # ADD INPUT INDEXES
        for morp_tuple in input_word : # ('한국', 'NNP'), ('의', 'JKG')
            pass_this_word = tryAddVocab(cur_input_word, input_lang, morp_tuple[0], Args.args.enc_unit)
            pass_this_word = tryAddVocab(cur_input_word, input_lang, morp_tuple[1], 'morp') # put POS as a whole

        # ADD OUTPUT INDEXES
        pass_this_word = tryAddVocab(cur_output_word, output_lang, output_word, Args.args.dec_unit)
        if pass_this_word == True :
            continue

        # If current sentence is longer than the max_sent length, skip it !
        if ( len(cur_input_word) < Args.args.max_sent and len(cur_output_word) < Args.args.max_sent ) :
            cur_input_word.append(EOS_token)
            cur_output_word.append(EOS_token)
            cur_input_word = ZeroPadding(cur_input_word, 'front')
            cur_output_word = ZeroPadding(cur_output_word, 'back')

            input_sent.append(cur_input_word)
            output_sent.append(cur_output_word)
            pairs.append([cur_input_word, cur_output_word])

    print("Actual Number of words read : %d" % len(input_sent))
    return input_sent, output_sent, pairs

def variableFromSentence(sentence) :
    sentence.append(EOS_token)
    result = Variable(torch.LongTensor(sentence).view(-1, 1))
    if ( Args.args.no_gpu ) :
        return result
    else :
        return result.cuda()

def variableFromPair(pair) :
    input_variable = variableFromSentence(pair[0])
    target_variable = variableFromSentence(pair[1])
    return (input_variable, target_variable)

def TensorFromSentence(sentence) :
    #sentence.append(EOS_token)
    result = torch.LongTensor(sentence).view(-1, 1)
    return result
    if ( Args.args.no_gpu ) :
        return result
    else :
        return result.cuda()

def TensorFromPair(pair) :
    input_variable = TensorFromSentence(pair[0])
    target_variable = TensorFromSentence(pair[1])
    return (input_variable, target_variable)
