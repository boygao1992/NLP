import math

def sentence_logprob(self, sentence):
    """
    COMPLETE THIS METHOD (PART 5)
    Returns the log probability of an entire sequence.
    """
    return sum(map(lambda t: math.log2(smoothed_trigram_probability(self, t)), get_ngrams(sentence,3)))
