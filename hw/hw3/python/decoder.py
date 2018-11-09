from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State, dep_relations, NUM_DEP, NUM_CLASSES

def isLegal(state):
    def _isLegal(transition_index):
        # illegal moves
        # 1. `left_arc` and `right_arc` when stack is empty
        if len(state.stack) == 0:
            if transition_index < NUM_DEP * 2:
                return False
        # 2. `shift` when buffer is empty but stack is not empty
        if (len(state.buffer) == 0) and (len(state.stack) > 0):
            if transition_index == NUM_DEP * 2 + 1:
                return False
        # 3. `left_arc` when root node is on the top of the stack
        if (len(state.stack) > 0) and (state.stack[-1] == 0):
            if transition_index < NUM_DEP:
                return False
        return True
    return _isLegal

def contains(legal_moves):
    def _contains(transition_index):
        return transition_index in legal_moves
    return _contains

def toTransition(transition_index):
    if transition_index < NUM_DEP:
        return ("left_arc", dep_relations[transition_index])
    if transition_index < NUM_DEP*2:
        return ("right_arc", dep_relations[transition_index - NUM_DEP])
    return ("shift", None)

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)

        while state.buffer:
            # TODO: Write the body of this loop for part 4

            # TODO when the buffer is empty
            input_rep = self.extractor.get_input_representation(words, pos, state)
            output_rep = self.model.predict(np.array( [input_rep] ))
            legal_moves = list(filter(isLegal(state), np.arange(NUM_CLASSES)))
            sorted_transition_index = reversed(np.argsort(output_rep)[0])
            legal_transition_index = list(filter(contains(legal_moves), sorted_transition_index))
            transition_index = legal_transition_index[0]
            (operator, label) = toTransition(transition_index)
            if operator == "left_arc":
                state.left_arc(label)
            if operator == "right_arc":
                state.right_arc(label)
            if operator == "shift":
                state.shift()

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
