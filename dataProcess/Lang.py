class Lang :
    def __init__ (self, name) :
        self.name = name
        self.syll2index = {'SOS': 0, 'EOS': 1, 'SPACE': 2} # syllable to index
        self.syll2count = {} # count the number of occurrences of syllables in the corpus
        self.index2syll = {0: "SOS", 1:"EOS", 2:"SPACE"}
        self.n_sylls = 3

    def addWord(self, Word) :
        if (type(Word) == list):  # for input word, decomposed as morphemes, consider : [('사상', 'NNG'), ('이', 'JKS')]
            for morpTuple in Word :
                for syll in morpTuple[0] :
                    self.addSyll(syll)
                self.addSyll(morpTuple[1]) # add POS

        elif ( type(Word) == str ) : # for output word (composed) as a whole word
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



