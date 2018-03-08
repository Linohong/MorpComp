import Arguments as Args
import torch
import os
import Network_Attention as Network
import Train_KFold_Attn as T
from sklearn.model_selection import KFold

torch.manual_seed(1)

#********************************#
#******* Load DATA Part *********#
#********************************#
print("\nLoading Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang
path = '../data'
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
# import Evaluate as E

trainSize = Args.args.train_size
input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
if (trainSize > len(input_sent)) :
    trainSize = len(input_sent)
training_pairs = [D_pair.variableFromPair(pairs[i]) for i in range(trainSize)] # now returned as Variable of indexes
kf = KFold(n_splits=Args.args.kfold)
kf.get_n_splits(training_pairs)

k=0
train_index = [i for i in range(trainSize)]
# for train_index, test_index in kf.split(training_pairs) :
#k = k + 1
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

    #print("Evaluation at [%d]-Fold" % k)
    #E.Evaluate(EncNet, DecNet, test_index, output_lang, training_pairs)


#************************************#
#******* Saving the Network *********#
#************************************#
print('Saving the Model...')
torch.save(EncNet.state_dict(), './savedEnc_Att')
torch.save(EncNet, './saveEntireEnc_Att')
torch.save(DecNet.state_dict(), './savedDec_Att')
torch.save(DecNet, './saveEntireDec_Att')


# Print Result with arguments both into prompt and file
# filename = "test"
# outfile = open("%s.txt" % filename,"w")
# print("\nParameters :")
# for attr, value in sorted(Args.args.__dict__.items()) :
#     print("\t{}={}".format(attr.upper(), value))
#     outfile.write("\t{}={}\n".format(attr.upper(), value))
