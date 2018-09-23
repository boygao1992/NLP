from collections import Counter
def count_ngrams(self, corpus):
    """
    COMPLETE THIS METHOD (PART 2)
    Given a corpus iterator, populate dictionaries of unigram, bigram,
    and trigram counts.
    """

    # self.unigramcounts = {} # might want to use defaultdict or Counter instead
    # self.bigramcounts = {}
    # self.trigramcounts = {}

    self.unigramcounts = Counter()
    self.bigramcounts = Counter()
    self.trigramcounts = Counter()

    ##Your code here

    for sentence in corpus:
        self.unigramcounts += Counter(get_ngrams(sentence, 1))
        # special treatment of ("START",) and ("STOP",) ? (100% present)
        # self.unigramcounts += Counter([t for t in get_ngrams(sentence, 1) if t not in [("START",), ("STOP",)]])

        self.bigramcounts += Counter(get_ngrams(sentence, 2))

        self.trigramcounts += Counter(get_ngrams(sentence, 3))
        # special treatment of ("START", "START", w0) ?

    self.wordtotal = sum(self.unigramcounts.values())
    # self.wordtotal = sum(self.unigramcounts.values()) - self.unigramcounts[("START",)] - self.unigramcounts[("STOP",)]

    return
