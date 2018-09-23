def smoothed_trigram_probability(self, trigram):
    """
    COMPLETE THIS METHOD (PART 4)
    Returns the smoothed trigram probability (using linear interpolation).
    """
    lambda1 = 1/3.0
    lambda2 = 1/3.0
    lambda3 = 1/3.0

    return lambda1 * raw_trigram_probability(self, trigram)
+ lambda2 * raw_bigram_probability(self, trim_tuple_head(trigram,2))
+ lambda3 * raw_unigram_probability(self, trim_tuple_head(trigram,1))
