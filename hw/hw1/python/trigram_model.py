import sys
from collections import defaultdict
import math
import random
import os
import os.path

from itertools import accumulate
from collections import Counter
"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile,'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)

def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    # assert isinstance(sequence, list)
    # assert isinstance(n, int) and n >= 1

    result = []
    length = len(sequence)
    if n > 1:
        for i in range(length):
            if i > 0:
                result.append(result[i-1][-(n-1):] + (sequence[i],))
            else:
                result.append((sequence[0],))
        for i in range(length):
            result[i] = (("START",)*(n - i - 1) + result[i])
        result.append(result[length-1][-(n-1):] + ("STOP",))
    else:
        for word in sequence:
            result.append((word,))
        result.append(("STOP",))

    if n == 1:
        result = [("START",)] + result

    return result


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        # self.unigramcounts = {} # might want to use defaultdict or Counter instead
        # self.bigramcounts = {}
        # self.trigramcounts = {}

        ##Your code here
        unigrams = []
        bigrams = []
        trigrams = []
        for sentence in corpus:
            unigrams += get_ngrams(sentence, 1)

            bigrams += get_ngrams(sentence, 2)

            trigrams += get_ngrams(sentence, 3)

        self.unigramcounts = Counter(unigrams)
        self.bigramcounts = Counter(bigrams)
        self.trigramcounts = Counter(trigrams)

        self.sentence_total = self.unigramcounts[("START",)]

        del self.unigramcounts[("START",)]
        del self.unigramcounts[("STOP",)]

        self.word_total = sum(self.unigramcounts.values())

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if (trigram[0] == "START") and (trigram[1] == "START"):
            return self.trigramcounts[trigram] / self.sentence_total

        if self.bigramcounts[trigram[:-1]] == 0:
            return 0
        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:-1]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        if (bigram[0] == "START"):
            return self.bigramcounts[bigram] / self.sentence_total

        if self.unigramcounts[bigram[:-1]] == 0:
            return 0
        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:-1]]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        if (unigram[0] == "START") or (unigram[0] == "STOP"):
            return 1
        return self.unigramcounts[unigram] / self.word_total

    def generate_sentence(self,t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[-2:]) + lambda3 * self.raw_unigram_probability(trigram[-1:])

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        return sum(map(lambda t: math.log2(self.smoothed_trigram_probability(t))
                       , get_ngrams(sentence,3)))

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        l = 0
        word_total = 0
        for s in corpus:
            l += self.sentence_logprob(s)
            word_total += len(s)
        l /= word_total
        return pow(2, -l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # ..

        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # ..

        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.


    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
