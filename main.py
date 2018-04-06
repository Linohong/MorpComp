import pickle
import torch
import numpy as np
import torch.utils.data as data
#from sklearn.model_selection import KFold
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang
import Arguments as Args

torch.manual_seed(1)
print("train with vanilla 50 epochs, many 25 sents examples, saved as morpTest, 0.003 lr, 3 early stopping, SGD Optimizer")
print("testing Enc : morp unit, Dec : Syll unit")

if (Args.args.model == 'vanilla') :
    from Model import Network as Network
    from Train import Train_miniBatch as T
elif (Args.args.model == 'attn') :
    from Model import Network_Attention_Complete as Network
    from Train import Train_KFold_Attn_Complete as T


#********************************#
#******* Load DATA Part *********#
#********************************#
print("\nLoading Train Data...")
filename = D_read.getFilenames()
input_lang = Lang.Lang('morp_decomposed')
output_lang = Lang.Lang('morp_composed')
corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
print("Input vocab : %d" % input_lang.n_sylls)
print("Output vocab : %d" % output_lang.n_sylls)
print("Done Loading!!!")


#************************************#
#******* Prepare Train DATA *********#
#************************************#
trainSize = Args.args.train_size
input_sent, output_sent, pairs = D_pair.MakePair(corpus, input_lang, output_lang)
trainSize = len(input_sent) if trainSize > len(input_sent) else trainSize

# Mini_Batch
batch_size = Args.args.batch_size
x_train = torch.LongTensor(input_sent)
y_train = torch.LongTensor(output_sent)
train_data = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=32)


with open(Args.args.model_name + 'input_lang.p', 'wb') as fp : # Write Vocabulary
     pickle.dump(input_lang, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(Args.args.model_name + 'output_lang.p', 'wb') as fp :
     pickle.dump(output_lang, fp, protocol=pickle.HIGHEST_PROTOCOL)


#************************************#
#******* LOAD Network ***************#
#************************************#
EncNet = Network.EncoderRNN(input_lang.n_sylls, Args.args.embed_size, Args.args.hidden_size)
DecNet = Network.DecoderRNN(output_lang.n_sylls, Args.args.embed_size, Args.args.hidden_size)
if Args.args.no_gpu == False:
    EncNet.cuda()
    DecNet.cuda()


#************************************#
#******* Train Starts ***************#
#************************************#
print("\nTraining...")
prev_loss = 1000 # random big number
early = Args.args.early
for epoch in range(Args.args.epoch) :
    print("epoch : [%d]" % (epoch))
    loss = T.TrainIters(trainloader, EncNet, DecNet, trainSize=trainSize, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)

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
torch.save(EncNet, './Enc' + Args.args.model_name)
torch.save(DecNet, './Dec' + Args.args.model_name)

