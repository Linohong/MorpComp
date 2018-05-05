import pickle
import torch
import torch.utils.data as data
import dataProcess.ReadFromFile as D_read
import dataProcess.Make_ExamplePair as D_pair
import dataProcess.Lang as Lang
import Arguments as Args

torch.manual_seed(1)

if (Args.args.model == 'vanilla') :
    from Model import Network as Network
    from Train import Train_miniBatch as T
elif (Args.args.model == 'attn') :
    from Model import Network_Attention_Complete as Network
    from Train import Train_KFold_Attn_Complete as T




#********************************#
#******* Load DATA Part *********#
#********************************#
if (True) : # if input_sent.txt, output_sent.txt exists
               # if false , you don't have to read all the 244 files in the train list
    print("\nLoading Train Data...")
    filename = D_read.getFilenames()
    input_lang = Lang.Lang('morp_decomposed')
    output_lang = Lang.Lang('morp_composed')
    if ( Args.args.exam_unit == 'sent' or Args.args.exam_unit =='context_word' ) :
        corpus = D_read.getData(filename, input_lang, output_lang) # to this point, we only read data but make a sentence of indexes nor wrap them with Variable
    elif ( Args.args.exam_unit == 'word') :
        corpus, max_word_length = D_read.getDataWordUnit(filename, input_lang, output_lang)

    print("Input vocab : %d" % input_lang.n_sylls)
    print("Output vocab : %d" % output_lang.n_sylls)
    print("Done Loading!!!")

    #print('Saving the Vocab...')
    #with open('ModelWeights/vocab/' + Args.args.model_name + '_in.p', 'wb') as fp:  # Write Vocabulary
    #    pickle.dump(input_lang, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('ModelWeights/vocab/' + Args.args.model_name + '_out.p', 'wb') as fp:
    #    pickle.dump(output_lang, fp, protocol=pickle.HIGHEST_PROTOCOL)



#************************************#
#******* Prepare Train DATA *********#
#************************************#
if (True) : # if False, you don't have to process given corpus
    #with open("../data/train/read_corpus.txt", "rb") as fp: # read existing corpus
    #    read_corpus = pickle.load(fp)
    # test_read_corpus = read_corpus[len(read_corpus)-9000:] # for word unit, uncomment this
    # read_corpus = read_corpus[:len(read_corpus)-9000] # for word unit, uncomment this
    # read_corpus = D_read.corpusSent2Word(read_corpus) # for word unit, uncomment this
    # test_read_corpus = D_read.corpusSent2Word(test_read_corpus) # for word unit, uncomment this

    print('Reading the Vocab...')
    with open('ModelWeights/vocab/' + Args.args.model_name + '_in.p', 'rb') as fp : # read input language
        input_lang = pickle.load(fp)
    with open('ModelWeights/vocab/' + Args.args.model_name + '_out.p', 'rb') as fp : # read output language
        output_lang = pickle.load(fp)

    if (Args.args.exam_unit == 'sent') :
        input_sent, output_sent, read_corpus = D_pair.MakePair(corpus, input_lang, output_lang)
        #input_sent, output_sent, read_corpus = D_pair.MakePair(read_corpus, input_lang, output_lang)
    elif ( Args.args.exam_unit == 'word') :
        input_sent, output_sent, pairs = D_pair.MakePairWordUnit(corpus, input_lang, output_lang)
        # input_sent, output_sent, _ = D_pair.MakePairWordUnit(read_corpus, input_lang, output_lang)
        # test_input_sent, test_output_sent, _ = D_pair.MakePairWordUnit(test_read_corpus, input_lang, output_lang)

    D_pair.makeStandardDataSet(input_sent, output_sent, 'input_sent', 'output_sent', Args.args.exam_unit)
    #D_pair.makeStandardDataSet(input_sent, output_sent, 'input_word', 'output_word', Args.args.exam_unit, test_input_sent, test_output_sent) # uncomment this for word unit



#*********************************************************#
#******* Just Reading from already prepared data *********#
#*********************************************************#

if (True) : # if True, then read from the existing data in the train folder
            # saved as, input_word.txt, output_word.txt
    with open('../data/train/input_' + Args.args.exam_unit + '.txt', 'rb') as fp:  # read output language
        input_sent = pickle.load(fp)
    with open('../data/train/output_' + Args.args.exam_unit + '.txt', 'rb') as fp:  # read output language
        output_sent = pickle.load(fp)

    #with open('ModelWeights/vocab/' + Args.args.model_name + '_in.p', 'rb') as fp:  # read input language
    #    input_lang = pickle.load(fp)
    #with open('ModelWeights/vocab/' + Args.args.model_name + '_out.p', 'rb') as fp:  # read output language
    #    output_lang = pickle.load(fp)

count = 0
for sent in input_sent :
    for entry in sent :
        count+=1
print('the number of words in sents : %d' % count)
exit()

if (Args.args.exam_unit == 'sent') :
    print("The number of sentences to be trained : %d" % len(input_sent))
else :
    print("The number of words to be trained : %d" % len(input_sent))
trainSize = Args.args.train_size
trainSize = len(input_sent) if trainSize > len(input_sent) else trainSize
print('Set trainSize to : %d' % trainSize)
input_sent = input_sent[:trainSize]
output_sent = output_sent[:trainSize]


# Mini_Batch
batch_size = Args.args.batch_size
x_train = torch.LongTensor(input_sent)
y_train = torch.LongTensor(output_sent)
train_data = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=32)






#************************************#
#******* LOAD Network ***************#
#************************************#

EncNet = Network.EncoderRNN(input_lang.n_sylls, Args.args.embed_size, Args.args.hidden_size)
DecNet = Network.DecoderRNN(output_lang.n_sylls, Args.args.embed_size, Args.args.hidden_size)
if Args.args.no_gpu == False:
    EncNet.cuda()
    DecNet.cuda()
T.ZeroWeight(EncNet, Args.args.embed_size)
T.ZeroWeight(DecNet, Args.args.embed_size)




#************************************#
#******* Train Starts ***************#
#************************************#

print("\nTraining...")
prev_loss = 1000 # random big number
early = Args.args.early
for epoch in range(Args.args.epoch) :
    print("epoch : [%d]" % (epoch))
    loss = T.TrainIters(trainloader, EncNet, DecNet, trainSize=trainSize, out_lang=output_lang, epoch_size=Args.args.epoch, batch_size=Args.args.batch_size, lr=Args.args.learning_rate)

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
torch.save(EncNet, 'ModelWeights/vanilla/Enc_' + Args.args.model_name)
torch.save(DecNet, 'ModelWeights/vanilla/Dec_' + Args.args.model_name)
print('Recording the trained files')
fw = open('../data/test/' + Args.args.model_name+ '_trained.txt', 'w')
filename.append('../data/train/shuffle.py')
filename.append('../data/train/done.txt')
filename.append('../data/train/done.txt.backup')

for file in filename :
  fw.write(file+'\n')

