def perplexity(self, corpus):
    """
    COMPLETE THIS METHOD (PART 6)
    Returns the log probability of an entire sequence.
    """
    return pow(2, - sum(map(lambda s: sentence_logprob(self, s), corpus)) / len(self.unigramcounts))
