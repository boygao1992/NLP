from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)



def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps

class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):

    deprels = dep_structure.deprels

    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


NULL = "<NULL>"

def translate(word_vocab, word, pos):
    # special treatment in `get_input_representation`
    if pos == NULL:
        return word_vocab[NULL]

    if word == None:
        return word_vocab["<Root>"]

    if pos == "CD":
        return word_vocab["<CD>"]

    if pos == "NNP":
        return word_vocab["<NNP>"]

    if word in word_vocab:
        return word_vocab[word]

    return word_vocab["<UNK>"]


class FeatureExtractor(object):

    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))

        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

# self.word_vocab = {'<CD>': 0, '<NNP>': 1, '<UNK>': 2, '<ROOT>': 3, '<NULL>': 4, 'completely': 5, ... }
# self.pos_vocab = {'<UNK>': 0, '<ROOT>': 1, '<NULL>': 2, 'RBS': 3, 'NNS': 4, '-RRB-': 5, 'WP$': 6, 'JJ': 7, 'VBG': 8, 'MD': 9, ':': 10, 'VB': 11, 'DT': 12, 'CD': 13, 'RP': 14, 'VBZ': 15, 'EX': 16, '.': 17, '#': 18, '-LRB-': 19, 'PDT': 20, 'NNPS': 21, 'VBP': 22, 'FW': 23, 'VBD': 24, 'CC': 25, 'WP': 26, 'PRP': 27, 'TO': 28, 'WRB': 29, 'IN': 30, 'VBN': 31, 'POS': 32, 'NN': 33, 'JJS': 34, 'LS': 35, 'PRP$': 36, 'JJR': 37, '$': 38, ',': 39, '``': 40, 'NNP': 41, 'UH': 42, 'SYM': 43, 'WDT': 44, 'RBR': 45, 'RB': 46, "''": 47}
# self.output_labels = {('shift', None): 0, ('left_arc', 'tmod'): 1, ('right_arc', 'tmod'): 2, ('left_arc', 'vmod'): 3, ... }
# words = [None, 'BUSH', 'AND', 'GORBACHEV', 'WILL', 'HOLD', 'two', 'days', 'of', 'informal', 'talks', 'next', 'month', '.']
# pos = [None, 'NNP', 'CC', 'NNP', 'MD', 'VB', 'CD', 'NNS', 'IN', 'JJ', 'NNS', 'JJ', 'NN', '.']
# iter:  1
# state.stack:  [0]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# state.deps:  set()
# iter:  2
# state.stack:  [0, 1]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
# state.deps:  set()
# iter:  3
# state.stack:  [0]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 1]
# state.deps:  {(1, 2, 'cc')}
# iter:  4
# state.stack:  [0, 1]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
# state.deps:  {(1, 2, 'cc')}
# iter:  5
# state.stack:  [0]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1]
# state.deps:  {(1, 2, 'cc'), (1, 3, 'conj')}
# iter:  6
# state.stack:  [0, 1]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
# state.deps:  {(1, 2, 'cc'), (1, 3, 'conj')}
# iter:  7
# state.stack:  [0, 1, 4]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5]
# state.deps:  {(1, 2, 'cc'), (1, 3, 'conj')}
# iter:  8
# state.stack:  [0, 1]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5]
# state.deps:  {(1, 2, 'cc'), (1, 3, 'conj'), (5, 4, 'aux')}
# iter:  9
# state.stack:  [0]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6, 5]
# state.deps:  {(1, 2, 'cc'), (5, 1, 'nsubj'), (1, 3, 'conj'), (5, 4, 'aux')}
# iter:  10
# state.stack:  [0, 5]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7, 6]
# state.deps:  {(1, 2, 'cc'), (5, 1, 'nsubj'), (1, 3, 'conj'), (5, 4, 'aux')}
# iter:  11
# state.stack:  [0, 5, 6]
# state.buffer:  [13, 12, 11, 10, 9, 8, 7]
# state.deps:  {(1, 2, 'cc'), (5, 1, 'nsubj'), (1, 3, 'conj'), (5, 4, 'aux')}

    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2

        NUM = 3
        state_stack = state.stack[-NUM:]
        state_buffer = np.flip(state.buffer[-NUM:])

        stack_rep = []
        buffer_rep = []
        for i in range(NUM):
            if i < len(state_stack):
                idx = state_stack[i]
                stack_word = words[ idx ]
                stack_pos = pos[ idx ]
                stack_rep.append(translate(self.word_vocab, stack_word, stack_pos))
            else:
                stack_rep.append(translate(self.word_vocab, "whatever", NULL))

            if i < len(state_buffer):
                idx = state_buffer[i]
                buffer_word = words[ idx ]
                buffer_pos = pos[ idx ]
                buffer_rep.append(translate(self.word_vocab, buffer_word, buffer_pos))
            else:
                buffer_rep.append(translate(self.word_vocab, "whatever", NULL))

        return np.concatenate(stack_rep, buffer_rep)

    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2
        return np.zeros(91)



def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)



if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)


    with open(sys.argv[1],'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)
