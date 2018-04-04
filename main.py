import pickle
import torch
from sklearn.model_selection import KFold

import Arguments as Args

print("train with vanilla 50 epochs, many 25 sents examples, saved as morpTest, 0.003 lr, 3 early stopping, SGD Optimizer")
print("testing Enc : morp unit, Dec : Syll unit")

if (Args.args.model == 'vanilla') :
    from Model import Network as Network
    from Train import Train_KFold as T
elif (Args.args.model == 'attn') :
    from Model import Network_Attention_Complete as Network
    from Train import Train_KFold_Attn_Complete as T

torch.manual_seed(1)
#********************************#
#******* Load DATA Part *********#
#********************************#
print("\nLoading Train Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang


filename = D_read.getFilenames()
input_lang = Lang.Lang('morp_decomposed')
output_lang = Lang.Lang('morp_composed')
corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable

# with open('corpus30sent.p', 'wb') as fp :
#      pickle.dump(corpus, fp, protocol=pickle.HIGHEST_PROTOCOL)
print("Input vocab : %d" % input_lang.n_sylls)
print("Output vocab : %d" % output_lang.n_sylls)
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


with open(Args.args.model_name + 'input_lang.p', 'wb') as fp :
     pickle.dump(input_lang, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(Args.args.model_name + 'output_lang.p', 'wb') as fp :
     pickle.dump(output_lang, fp, protocol=pickle.HIGHEST_PROTOCOL)

EncNet = Network.EncoderRNN(input_lang.n_sylls, Args.args.hidden_size)
DecNet = Network.DecoderRNN(output_lang.n_sylls, Args.args.hidden_size)
if Args.args.no_gpu == False:
    EncNet.cuda()
    DecNet.cuda()

print("\nTraining...")
prev_loss = 1000 # random big number
early = Args.args.early
for epoch in range(Args.args.epoch) :
    print("[%d]-Fold, epoch : [%d]" % (k, epoch))
    loss = T.TrainIters(train_index, training_pairs, EncNet, DecNet, trainSize=trainSize, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)

    # Early Stopping Part
    if ( Args.args.early != None ) :
        if T.EarlyStopping(prev_loss, loss, early) == True :
            print('Early stopped at [%d] Epoch - [%.2f] loss' % epoch, loss)
            break


print("\nDone Training !")
#************************************#
#******* Saving the Network *********#
#************************************#
print('Saving the Model...')
torch.save(EncNet, './saveEntireEnc' + Args.args.model_name)
torch.save(DecNet, './saveEntireDec' + Args.args.model_name)

