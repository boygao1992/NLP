"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 2 - Parsing with Context Free Grammars
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum

# Predicates
def isNonTerminal(symbol):
    return symbol.isupper() # 'COST'.isupper() => True
def isTerminal(symbol):
    return not symbol.isupper()
# special cases not covered by 'islower()' :
# - ('WHNP', ('0',))
# - ('PUN',  ('.',))

class Pcfg(object):
    """
    Represent a probabilistic context free grammar.
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self,grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()


    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False.
        """
        # TODO, Part 1

        result = True

        # step 1: check that the rules have the right format
        # Context-free Grammar (CFG), G = (N, T, R, S)
        #   N = Non-terminal
        #   T = Terminal
        #   R = production Rules
        #   S = Starting non-terminal
        # Chomsky Normal Form (CNF):
        #   1. A -> B C (N -> N N)
        #     e.g. ('PP', ('AFTER', 'NP'), 0.0253671562083)
        #   2. A -> b (N -> T)
        #     e.g. ('COST', ('cost',), 1.0)
        # Irregularity (but CNF-legit):
        #   1. no POS tags
        #   2. phrase (N) -> lowercase word (T)
        #   3. uppercase word (N) -> lowercase word (T)

        # Rules:
        #   1. lhs is always a non-terminal (in uppercase)
        #   2. rhs
        #     2.1 a Tuple of 2 non-terminals
        #     2.2 a Tuple of 1 terminal

        for item in self.lhs_to_rules.items():
            lhs = item[0]
            rules = item[1]

            # rule 1
            if not isNonTerminal( lhs ):
                result = False
                print( lhs, " : invalid lhs" )

            # rule 2
            for rule in rules:
                rhs = rule[1]

                if ( len( rhs ) != 1 ) and ( len( rhs ) != 2 ):
                    result = False
                    print( rule[:2], " : invalid rhs, must be one or two symbols" )

                if ( len( rhs ) == 2 ) and ( ( not isNonTerminal( rhs[0] ) ) or ( not isNonTerminal( rhs[1] ) ) ):
                    result = False
                    print( rule[:2], " : 2.1 a Tuple of 2 non-terminals")

                if ( len( rhs ) == 1 ) and ( not isTerminal( rhs[0] ) ):
                    result = False
                    print( rule[:2], " : 2.2 a Tuple of 1 terminal")

        # step 2: check that all probabilities of the same lhs symbol sum to 1.0
        for item in self.lhs_to_rules.items():
            # item = ('WITH', [('WITH', ('with',), 1.0)])
            lhs = item[0] # 'WITH'
            rules = item[1] # [('WITH', ('with',), 1.0)]
            prob = fsum( map( lambda t: t[2], rules ) )
            # if (prob != 1.0):
            #     print (key, " : ", prob)
            #     result = false
            # >>>
            # ADJP  :  1.0000000000012
            # ADVP  :  1.0000000000001
            # FRAG  :  0.9999999999996
            # FRAGBAR  :  0.9999999999993
            # NP  :  0.999999999999824
            # NPBAR  :  0.99999999999996
            # PP  :  1.00000000000053
            # S  :  0.9999999999997999
            # SBAR  :  0.9999999999999
            # SBARQ  :  1.0000000000001
            # SQ  :  0.9999999999999001
            # SQBAR  :  1.0000000000006
            # TOP  :  0.99999999999993
            # VPBAR  :  1.00000000000024
            # WHNP  :  1.0000000000003
            # X  :  0.9999999999989999
            # INTJ  :  0.9999999999999
            if ( abs( prob - 1.0 ) > 1e-10 ):
                result = False

        return result


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        if grammar.verify_grammar():
            print("verify_grammar: succeeded")
        else:
            print("verify_grammar: failed")
