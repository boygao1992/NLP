def raw_trigram_probability(self,trigram):
    """
    COMPLETE THIS METHOD (PART 3)
    Returns the raw (unsmoothed) trigram probability
    """
    return self.trigramcounts[trigram] / sum(self.trigramcounts.values())

def raw_bigram_probability(self, bigram):
    """
    COMPLETE THIS METHOD (PART 3)
    Returns the raw (unsmoothed) bigram probability
    """
    return self.bigramcounts[bigram] / sum(self.bigramcounts.values())

def raw_unigram_probability(self, unigram):
    """
    COMPLETE THIS METHOD (PART 3)
    Returns the raw (unsmoothed) unigram probability.
    """

    #hint: recomputing the denominator every time the method is called
    # can be slow! You might want to compute the total number of words once,
    # store in the TrigramModel instance, and then re-use it.
    return self.unigramcounts[unigram] / self.wordtotal
