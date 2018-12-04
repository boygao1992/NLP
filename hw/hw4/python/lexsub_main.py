#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
from collections import defaultdict

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

def underscore2space(str):
    return str.replace('_', ' ')

def space2underscore(str):
    return str.replace(' ', '_')

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = set()
    for lemma1 in wn.lemmas(space2underscore(lemma), pos):
        for lemma2 in lemma1.synset().lemmas():
            possible_synonyms.add(lemma2.name())
    possible_synonyms.remove(space2underscore(lemma))
    return list(map(underscore2space, possible_synonyms))


def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    lemma = space2underscore(context.lemma)
    pos = context.pos
    cooccurrence = defaultdict(int)
    for lemma1 in wn.lemmas(lemma, pos):
        count = lemma1.count()
        for lemma2 in lemma1.synset().lemmas():
            cooccurrence[lemma2.name()] += count
    if lemma in cooccurrence:
        del cooccurrence[lemma]
    return underscore2space(max(cooccurrence, key=cooccurrence.get)) # replace for part 2

def wn_simple_lesk_predictor(context):
    return None #replace for part 3

class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context):
        return None # replace for part 4

    def predict_nearest_with_context(self, context):
        return None # replace for part 5

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        prediction = wn_frequency_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
