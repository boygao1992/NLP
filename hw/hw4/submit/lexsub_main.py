# !/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
from collections import defaultdict
import string
from functools import reduce
import math
import scipy.spatial.distance as distance

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


# Total = 298, attempted = 298
# precision = 0.087, recall = 0.087
# Total with mode 206 attempted 206
# precision = 0.112, recall = 0.112
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
    return underscore2space(max(cooccurrence, key=cooccurrence.get))


# Total = 298, attempted = 298
# precision = 0.121, recall = 0.121
# Total with mode 206 attempted 206
# precision = 0.170, recall = 0.170
def wn_simple_lesk_predictor(context):
    lemma = space2underscore(context.lemma)
    pos = context.pos
    left_context = context.left_context
    right_context = context.right_context

    stop_words = set(stopwords.words('english'))
    isNotStopWord = lambda word: not (word in stop_words) and not (word in string.punctuation)
    normalize = lambda arr: set(map(lambda s: s.lower(), filter(isNotStopWord, arr)))

    context_wordbag = normalize(left_context + right_context)

    overlap = []
    frequency = []
    synsets = []

    for lemma1 in wn.lemmas(lemma, pos):
        # lemma1 ~ sense
        predicate = reduce(lambda acc,lemma2: acc or (lemma2.name() != lemma), lemma1.synset().lemmas(), False)
        if not predicate:
            continue
        frequency.append(lemma1.count())
        synsets.append(lemma1.synset())
        examples = lemma1.synset().examples()
        for hypernym_synset in lemma1.synset().hypernyms():
            examples += hypernym_synset.examples()

        # examples = ['the sun was bright and hot', 'a bright sunlit room']
        # examples_wordbag = {'the', 'a', 'hot', 'and', 'room', 'was', 'sunlit', 'sun', 'bright'}
        examples_wordbag = normalize(' '.join(examples).split(' '))
        overlap.append(len(examples_wordbag & context_wordbag))

    synset_idx = np.argmax(overlap)
    if overlap[synset_idx] == 0:
        synset_idx = np.argmax(frequency)

    lemmas = list(filter(lambda lemma2: lemma2.name() != lemma, synsets[synset_idx].lemmas()))
    lemma_idx = np.argmax(list(map(lambda lemma1: lemma1.count(), lemmas)))

    return lemmas[lemma_idx].name()

class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

# Total = 298, attempted = 298
# precision = 0.053, recall = 0.053
# Total with mode 206 attempted 206
# precision = 0.063, recall = 0.063
    def predict_nearest(self,context):
        lemma = context.lemma
        pos = context.pos
        candidates = get_candidates(lemma, pos)

        result = None
        best_dis = math.inf
        for candidate in candidates:
            if candidate in self.model.wv:
                dis = self.model.similarity(candidate, lemma)
                if dis < best_dis:
                    result = candidate
                    best_dis = dis

        return result # replace for part 4

# Total = 298, attempted = 298
# precision = 0.117, recall = 0.117
# Total with mode 206 attempted 206
# precision = 0.184, recall = 0.184
    def predict_nearest_with_context(self, context):
        stop_words = set(stopwords.words('english'))
        isNotStopWord = lambda word: not (word in stop_words) and not (word in string.punctuation)
        deStopWord = lambda arr: list(filter(isNotStopWord, arr))
        normalize = lambda arr: set(map(lambda s: s.lower(), arr))

        left_context = deStopWord(context.left_context)
        right_context = deStopWord(context.right_context)

        left_boundary = 0
        if (len(left_context) > 5):
            left_boundary = len(left_context) - 5

        context_wordbag = normalize(left_context[left_boundary:] + right_context[:5])

        center = np.copy(self.model.wv[lemma])
        for word in context_wordbag:
            if word in self.model.wv:
                center += self.model.wv[word]

        lemma = space2underscore(context.lemma)
        pos = context.pos
        candidates = get_candidates(lemma, pos)

        result = None
        best_dis = math.inf
        for candidate in candidates:
            if candidate in self.model.wv:
                dis = distance.cosine(center, self.model.wv[candidate])
                if dis < best_dis:
                    result = candidate
                    best_dis = dis

        return  result

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict_nearest_with_context(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))


