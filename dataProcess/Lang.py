import Arguments as Args

class Lang :
    def __init__ (self, name) :
        self.name = name
        if ( Args.args.exam_unit == 'sent' ) :
            self.syll2index = {'ZERO':0, 'SOS': 1, 'EOS': 2, 'SPACE': 3} # syllable to index
            self.syll2count = {'ZERO':0, 'SOS':0, 'EOS':0, 'SPACE':0} # count the number of occurrences of syllables in the corpus
            self.index2syll = {0:'ZERO', 1: "SOS", 2:"EOS", 3:"SPACE"}
            self.n_sylls = 4
        elif ( Args.args.exam_unit == 'word' ) :
            self.syll2index = {'ZERO': 0, 'SOS': 1, 'EOS': 2}  # syllable to index
            self.syll2count = {'ZERO': 0, 'SOS': 0, 'EOS': 0}  # count the number of occurrences of syllables in the corpus
            self.index2syll = {0: 'ZERO', 1: "SOS", 2: "EOS"}
            self.n_sylls = 3

    def addWord(self, Word, unit) :
        if (type(Word) == list):  # for input word, decomposed as morphemes, consider : [('사상', 'NNG'), ('이', 'JKS')]
            if ( unit == 'syll' ) :
                for morpTuple in Word :
                    for syll in morpTuple[0] :
                        self.addSyll(syll)
                    self.addSyll(morpTuple[1]) # add POS
            elif ( unit == 'morp' ) :
                for morpTuple in Word :
                    self.addSyll(morpTuple[0])
                    self.addSyll(morpTuple[1])

        elif ( type(Word) == str ) : # for output word (composed) as a whole word
            if ( unit == 'syll' ) :
                for syll in Word :
                    self.addSyll(syll)

    def addSyll(self, syll) :
        if syll not in self.syll2index :
            self.syll2index[syll] = self.n_sylls
            self.syll2count[syll] = 1
            self.index2syll[self.n_sylls] = syll
            self.n_sylls += 1
        else :
            self.syll2count[syll] += 1



