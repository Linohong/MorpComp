# MorpComp

<<<<<<< HEAD
1. DataProcessing.
    i) firstly, read the file @ReadFromFile.py
        => returns corpus
    ii) secondly, properly process the corpus into input unit with respect to the encode network.
        => pairs (encode, decode) as a sentence unit. since one decomposed sentence will be composed and will come out as a sentence.
    iii) wrap them with Variable.

=======
Commit Information
03/01 : initial commit. Not Working yet.
03/08 : Attention model reference site below.
    https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq/blob/master/attentionRNN.py

=======
03/12 experiment : Seq2seq experiment : loss reduction continued till 30 epoch and then it rose again
weights are saved as saveEntireEnc, saveEntireDec
    => loss about 1.2

03/13 experiment : Seq2seq experiment :
  i) remove ReLu function of the input in the Decoder
    output = F.Relu(embedded)
    -> output = embedded
    set epoch as 30.
    => loss about 1.64

  ii) retrieve ReLu function + add dropout before linear function
        output, hidden = self.gru(output, hidden) # gru(input, h_0) : input=>(seq_len, batch, input_size)
        output = self.dropout(output)
        output = self.softmax(self.fc(output[0]))

  iii) remove dropout, 30 epoch.
    => 1.2147 loss at last epoch

  iv)
    - 50 epoch, learning rate=0.001. (vanilla)
        => 0.6544 loss at last epochs. and still was decreasing.
    - learning rate = 0.1
        => divergence occurs
    - 100 epochs, learning rate=0.001, attn model
        => first 0.58 loss at 65 epochs
        => first 0.5635 loss at 83 epochs
        => 0.7231 loss at 100 epochs
    - 50 epochs, learning rate=0.05
        => diverges
    - 100 epochs, learning rate=0.001


04/06 experiment]
    - Crucial mistake on embedding view before forwarding into the GRU network.
    - batch *
