import torch
from torch.autograd import Variable
import Arguments as Args

SOS_token = 0
EOS_token = 1
SPACE_token = 2

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

def MakePair(corpus, input_lang, output_lang) :
    input_sent = []
    output_sent = []
    pairs = []

    for ind, sent in enumerate(corpus) :
        pass_this_sent = False
        cur_input_sent = []
        cur_output_sent = []
        # Example of word = ['한국의', [('한국', 'NNP'), ('의', 'JKG')]]
        for word in sent :
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
        if ( len(cur_input_sent) <= Args.args.max_sent ) :
            input_sent.append(cur_input_sent)
            output_sent.append(cur_output_sent)
            pairs.append([cur_input_sent, cur_output_sent])

    print("Actual Number of sentences read : %d" % len(input_sent))
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
