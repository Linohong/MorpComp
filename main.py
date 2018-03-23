import Arguments as Args
import torch
import os
from sklearn.model_selection import KFold

print("train with vanilla 50 epochs, many 30 sents examples, saved as vanilla4, 0.0005 lr")

if (Args.args.model == 'vanilla') :
    import Network
    import Train_KFold as T
elif (Args.args.model == 'attn') :
    import Network_Attention_Complete as Network
    import Train_KFold_Attn_Complete as T

torch.manual_seed(1)
#********************************#
#******* Load DATA Part *********#
#********************************#
print("\nLoading Train Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang
path = '../data/train'
filename = []
for file in os.listdir(path) :
    filename.append(file)

input_lang = Lang.Lang('morp_decomposed')
output_lang = Lang.Lang('morp_composed')
corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
print("Done Loading!!!")


#****************************#
#******* Train Part *********#
#****************************#
trainSize = Args.args.train_size
input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
if (trainSize > len(input_sent)) :
    trainSize = len(input_sent)
training_pairs = [D_pair.variableFromPair(pairs[i]) for i in range(trainSize)] # now returned as Variable of indexes
kf = KFold(n_splits=Args.args.kfold)
kf.get_n_splits(training_pairs)
k=0
train_index = [i for i in range(trainSize)]

EncNet = Network.EncoderRNN(input_lang.n_sylls, Args.args.hidden_size)
DecNet = Network.DecoderRNN(output_lang.n_sylls, Args.args.hidden_size)
if Args.args.no_gpu == False:
    EncNet.cuda()
    DecNet.cuda()

print("\nTraining...")
for epoch in range(Args.args.epoch) :
    print("[%d]-Fold, epoch : [%d]" % (k, epoch))
    T.TrainIters(train_index, training_pairs, EncNet, DecNet, trainSize=trainSize, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)
print("\nDone Training !")


#************************************#
#******* Saving the Network *********#
#************************************#
print('Saving the Model...')
torch.save(EncNet, './saveEntireEnc' + Args.args.model_name)
torch.save(DecNet, './saveEntireDec' + Args.args.model_name)

# import Evaluation as E
# print("\nLoading Test Data...")
# path = '../data/test'
# filename = []
# for file in os.listdir(path) :
#     filename.append(file)
#
# input_lang = Lang.Lang('morp_decomposed')
# output_lang = Lang.Lang('morp_composed')
# corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
# print("Done Loading!!!")
#
# input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
# training_pairs = [D_pair.variableFromPair(pairs[i]) for i in range(trainSize)] # now returned as Variable of indexes
# E.EvalIters(training_pairs, EncNet, DecNet, input_lang, output_lang)

