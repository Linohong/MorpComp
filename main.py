import Arguments as Args
import torch
import random
import Network

torch.manual_seed(1)

# LOAD DATA PART
print("\nLoading Data...")
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang
filename = ['BTHO0140']
input_lang = Lang('morp_decomposed')
output_lang = Lang('morp_composed')
corpus = D_read.getData(filename[0], input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
print("Done Loading!!!")

# Train
import Train_KFold as T
# import Evaluate as E
from sklearn.model_selection import KFold
trainSize = Args.args.train_size
pairs = D_pair.MakePair(corpus, input_lang, output_lang)
training_pairs = [D_pair.variableFromPair(pairs) for i in range(trainSize)] # now returned as Variable of indexes
kf = KFold(n_splits=Args.args.kfold)
kf.get_n_splits(training_pairs)

k=0
for train_index, test_index in kf.split(training_pairs) :
    k = k + 1
    EncNet = Network.EncoderRNN(input_lang.n_words, Args.args.hidden_size)
    DecNet = Network.DecoderRNN(output_lang.n_words, Args.args.hidden_size)
    if Args.args.no_gpu == False:
        EncNet.cuda()
        DecNet.cuda()

    print("\nTraining...")
    for epoch in range(Args.args.epoch) :
        print("[%d]-Fold, epoch : [%d]" % (k, epoch))
        T.TrainIters(train_index, training_pairs, EncNet, DecNet, trainSize=trainSize, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)
    print("\nDone Training !")

    print("Evaluation at [%d]-Fold" % k)
    E.Evaluate(EncNet, DecNet, test_index, output_lang, training_pairs)


# Print Result with arguments both into prompt and file
filename = "test"
outfile = open("%s.txt" % filename,"w")
print("\nParameters :")
for attr, value in sorted(Args.args.__dict__.items()) :
    print("\t{}={}".format(attr.upper(), value))
    outfile.write("\t{}={}\n".format(attr.upper(), value))
