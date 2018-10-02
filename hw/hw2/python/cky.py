"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 2 - Parsing with Context Free Grammars
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            # TODO bps should be a set
            # the same Non-terminal can be constituted by multiple rules
            # e.g. VP -> V   NP
            #      VP -> AUX VNP
            # or maybe there's a uniqueness constraint in the dataset
            if isinstance(bps, str): # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                # TODO (A, i, k) != (i,k,A)
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


def initBackpointer():
    return defaultdict()

class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar.
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(initBackpointer)
        probs = None
        n = len(tokens)
        for i in range(0, n):
            token = tokens[i]
            for rule in self.grammar.rhs_to_rules[(token,)]:
                lhs_np = rule[0]
                window = (i,i+1)
                table[window][lhs_np] = token

        for l in range(2, n+1):
            for i in range(0, n-l+1):
                for k in range(i+1, i+l):
                    left_window = (i,k)
                    right_window = (k,i+l)
                    for left_np in table[left_window]:
                        for right_np in table[right_window]:
                            for rule in self.grammar.rhs_to_rules[(left_np, right_np)]:
                                lhs_np = rule[0]
                                window = (i, i+l)
                                table[window][lhs_np] = ( (left_np, i, k), (right_np, k, i+l) )

        return table, probs


def get_tree(chart, i,j,nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    return None


if __name__ == "__main__":

    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        #print(parser.is_in_language(toks))
        #table,probs = parser.parse_with_backpointers(toks)
        #assert check_table_format(chart)
        #assert check_probs_format(probs)
