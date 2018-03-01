import torch
from torch.autograd import Variable
import Arguments as Args

SOS_token = 0
EOS_token = 1
SP_token = 2

# make korean sentence into list of indexes of syllables
def MakePair(corpus, input_lang, output_lang) :
    input_sent = []
    output_sent = []
    pairs = []

    for ind, sent in enumerate(corpus) :
        cur_input_sent = []
        cur_output_sent = []
        # Example of word = ['한국의', [('한국', 'NNP'), ('의', 'JKG')]]
        for word in sent :
            input_word = word[1]
            output_word = word[0]
            for morp_tuple in input_word : # ('한국', 'NNP'), ('의', 'JKG')
                for syll in morp_tuple[0] : # 한, 국
                    cur_input_sent.append(input_lang.syll2index(syll))
                cur_input_sent.append(input_lang.syll2index(morp_tuple[1])) #'NNP'

            for syll in output_word :
                cur_output_sent.append(output_lang.syll2index(syll))
            if ( ind != len(sent) ) :
                cur_output_sent.append(output_lang.syll2index('SP'))

        input_sent.append(cur_input_sent)
        output_sent.append(cur_output_sent)
        pairs.append([cur_input_sent, cur_output_sent])

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
