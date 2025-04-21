# same as alpha_optimization.py but for two rounds, like preflop-flop and turn-river

import numpy as np
from scipy import optimize
import pickle
import time
import matplotlib.pyplot as plt
import copy
import math
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import sys
from functools import lru_cache

# Constants (existing code)
# RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
RANKS_RIVER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O"}
ACTIONS = ["k", "c", "b", "f"]  # {k: check, c: call, b: bet/raise/all-in, f: fold}

TERMINAL_ACTION_STRS_TURN = {'bc', 'kbc', 'bbc', 'kbbc', 'bbbc', 'kbbbc', 'bbbbc', 'kbbbbc', 'kk'}

TERMINAL_ACTION_STRS_TURN_RIVER = {
    'kbbbck/bbbc', 'bbck/kbbf', 'bbbc/kbbc', 'bbbc/bbbbk', 'kbbbck/bbc', 'bc/bbbbk', 'bbck/bbbbk', 'kbbc/bbc', 'bbck/bbf', 'kbbc/k', 'bc/kbbc', 'bbbc/bbbbf', 
    'bbck/bbbbf', 'kbbbck/bf', 'bbbc/kbbbf', 'kbbbbc/bbbf', 'bbbf', 'kbck/kk', 'kk/bbf', 'kbbc/bbbbc', 'bbck/kbbc', 'bbbc/bbbc', 'kk/kbbbf', 'kbck/kbbf', 'kbck/bbbbc',
    'kbbbbc/bbf', 'kbbbbc/kbc', 'kbbbbc/kk', 'bc/kbc', 'kbck/kbf', 'kbbbbc/k', 'bf', 'bbbbck/bbbbk', 'bbbc/bbbf', 'kbbc/bbbbf', 'bbck/kbbbf', 'bbck/k', 'bbbc/bbc', 
    'kk/kk', 'bc/k', 'kk/k', 'bbbbck/kbbf', 'kbbbck/kbbf', 'kbbc/kbf', 'bbbbck/bbbbf', 'kk/bbbbf', 'bbbc/kk', 'kbck/k', 'kbbbbc/kbbf', 'kbbbbc/bf', 'bc/bc',
    'kbbbck/kbbbf', 'bc/bbc', 'kk/bbbf', 'bc/bbbbc', 'kk/bc', 'kbbc/bbbbk', 'kbck/bbc', 'kbck/bbbf', 'kbbc/kk', 'kbbbck/bbbf', 'kbbf', 'bbck/bbc', 'kk/bf', 'kbbbf', 
    'kbck/kbbbf', 'kk/kbc', 'bc/kbbbc', 'bbbbck/bbc', 'kbbbbf', 'bbbbck/bbf', 'bbbbf', 'kbbbck/kbc', 'bbbbck/bc', 'bbbbck/bbbf', 'bbck/kbf', 'kbck/bbbbk', 
    'kbbbbc/kbbbf', 'kbbbbc/bbbbk', 'kk/bbbc', 'bbck/bbbc', 'kbbc/kbbf', 'bc/kbbf', 'kbbbck/kk', 'kbbbbc/kbf', 'kk/kbbf', 'kbbc/bc', 'kbbbck/bbbbc', 'kbbbbc/bbbc', 
    'kbbc/kbbbf', 'kbbc/bbbc', 'bbbbck/kbf', 'kbck/bf', 'kbbbck/bbbbk', 'kbck/bbbbf', 'bbbbck/kbbbf', 'bbbbck/bbbc', 'bbck/kk', 'kk/bbc', 'bbck/bf', 'kbbbbc/kbbbc', 
    'bc/kbbbf', 'kk/kbbc', 'bbck/bbbbc', 'bbbc/kbf', 'bbbbck/kbbc', 'kbbbck/bc', 'bbbc/kbc', 'bbck/bc', 'kbbbck/k', 'bbbbck/kbc', 'kk/bbbbk', 'bc/bbbf', 'kk/bbbbc',
    'kbck/kbbbc', 'kbck/bbbc', 'kbbbbc/bbc', 'bbf', 'kbck/bc', 'kbbbck/kbbc', 'bbck/kbbbc', 'bbbbck/bbbbc', 'kk/kbf', 'bbbc/bbf', 'kbbc/kbbbc', 'kbbbbc/kbbc', 
    'kbbbck/kbf', 'bc/kk', 'kbbc/bf', 'kbck/kbbc', 'kbbc/bbbf', 'kbbc/bbf', 'kbbbck/kbbbc', 'bc/bf', 'bbbbck/k', 'kbbbbc/bc', 'kbbbck/bbbbf', 'bbbbck/kk', 'bbbc/k', 
    'kbbbbc/bbbbc', 'kbck/bbf', 'bbck/kbc', 'bbbc/kbbf', 'bc/bbbbf', 'bbbbck/kbbbc', 'bbck/bbbf', 'bc/bbf', 'kbf', 'bbbc/bf', 'bc/kbf', 'bbbc/bbbbc', 'bc/bbbc', 
    'kbbbck/bbf', 'kbbbbc/bbbbf', 'bbbbck/bf', 'bbbc/bc', 'kk/kbbbc', 'bbbc/kbbbc', 'kbbc/kbc', 'kbbc/kbbc', 'kbck/kbc'
}


TERMINAL_CHIPCOUNT_TURN_RIVER = {
    'bbbc/kbbc': 12, 'bbbbck/bbf': 18, 'bc/k': 3, 'bbbbf': 8, 'kk/k': 2, 'bbbbck/kbbc': 20, 'bbbc/kbbbf': 12, 'bbck/kk': 5, 'kbbbck/kbc': 10, 'bc/bc': 4,
    'kbck/bbbf': 6, 'kbbbck/bc': 10, 'kk/kbc': 3, 'bbbc/bbc': 12, 'kbck/kbbbc': 10, 'kbbbck/k': 9, 'bbbbck/kbc': 18, 'kbck/bbbc': 10, 'kk/bbbbf': 8, 
    'kbck/bbf': 4, 'bbbbck/kbbf': 18, 'bbbbck/bf': 16, 'bbbbck/kk': 17, 'kbbc/bc': 6, 'kbbc/bbbbf': 12, 'kbbc/bbbc': 12, 'bbbbck/k': 17, 'kbck/kk': 3, 
    'bc/kbbbc': 10, 'kbbbbc/bbf': 20, 'bbbbck/bbbbk': 32, 'kbck/bc': 4, 'kbbbbc/kk': 17, 'kbbbck/kbbf': 10, 'bbbc/bbbbk': 24, 'bbck/bbc': 8, 'bbbc/kbf': 8, 
    'bbbc/kbc': 20, 'bbck/bc': 6, 'kk/kbf': 1, 'bbbf': 4, 'kk/bc': 3, 'bbbbck/bc': 18, 'kbf': 1, 'bbbbck/bbbc': 24, 'kbck/bbbbf': 10, 'bbck/bbbbk': 12,
    'kbbc/kbbbf': 8, 'kk/bbc': 5, 'kbbbck/bbbf': 12, 'kk/bbbc': 9, 'kbck/k': 3, 'bbbbck/bbbbf': 24, 'bbbc/bbbbf': 16, 'kbbf': 2, 'kbck/bbc': 6, 'bbbbck/bbbbc': 32,
    'bc/bbbf': 6, 'bbck/bbbbf': 12, 'bbck/kbf': 4, 'kbck/kbbc': 6, 'bc/bbbbf': 10, 'kbbbbc/kbc': 18, 'kbbbbc/bf': 16, 'bbbbck/bbbf': 20, 'bbck/bf': 4, 
    'kbbbbf': 8, 'kbbc/kk': 5, 'bc/bbf': 4, 'bc/kk': 3, 'bbbbck/kbbbc': 24, 'bc/kbbf': 4, 'kbbbbc/k': 17, 'bbbc/bbbc': 16, 'kbck/kbbf': 4, 'kbbc/kbbf': 6, 
    'bc/bbbbk': 18, 'kbbbbc/bbbbk': 32, 'kbbbck/bbbc': 16, 'bc/kbbc': 6, 'kbbc/bbf': 6, 'bbck/kbbf': 6, 'bc/bf': 2, 'kbbbck/kbf': 8, 'kk/kbbbc': 9, 
    'kbck/bbbbc': 18, 'kk/bbbf': 5, 'bbbc/bf': 8, 'kk/kk': 2, 'kbbc/kbbbc': 12, 'kbbbck/kbbbf': 12, 'kbbc/bbbbc': 20, 'kk/bf': 1, 'kbbbbc/bbbbf': 24, 
    'bbck/kbbbf': 8, 'bbbbck/bbc': 20, 'kbbbbc/kbbbc': 24, 'bc/kbf': 2, 'bbbc/bbbf': 12, 'kbck/kbc': 4, 'bc/bbbbc': 18, 'bbck/bbbbc': 20, 'kbbbck/bbbbf': 16, 
    'bbbc/kbbf': 10, 'kbbbbc/bc': 18, 'bc/bbc': 6, 'kk/kbbbf': 5, 'bbck/kbc': 6, 'bbck/kbbc': 8, 'kbbc/bf': 4, 'bf': 1, 'bbbbck/kbbbf': 20, 'kbbbck/bf': 8,
    'kbbbbc/kbbbf': 20, 'kbbbbc/bbc': 20, 'bc/kbc': 4, 'bbck/kbbbc': 12, 'kbbbck/bbbbk': 24, 'kbbbbc/bbbbc': 32, 'kbbbf': 4, 'kbbc/k': 5, 'bbbc/k': 9, 
    'bbf': 2, 'kk/bbbbc': 17, 'bbbc/kk': 9, 'kbbc/bbbbk': 20, 'kbbbbc/kbbc': 20, 'kbbbck/kbbbc': 16, 'kbbbck/bbf': 10, 'kbbbck/kbbc': 12, 'kk/bbf': 2,
    'kbbc/kbbc': 8, 'bbbbck/kbf': 16, 'bbbc/bbf': 10, 'bbck/bbbc': 12, 'kbbbbc/kbbf': 18, 'kbck/bf': 2, 'kbbc/kbf': 4, 'bbck/bbbf': 8, 'kbbbck/bbc': 12,
    'kbbbck/bbbbc': 24, 'bbbc/kbbbc': 16, 'bbck/k': 5, 'bbck/bbf': 6, 'bc/kbbbf': 6, 'kk/bbbbk': 17, 'kbck/kbf': 2, 'bbbc/bc': 10, 'bbbc/bbbbc': 24, 
    'kbbbbc/bbbc': 24, 'kk/kbbf': 3, 'kbbbbc/bbbf': 20, 'kbck/kbbbf': 6, 'kbbc/bbbf': 8, 'kk/kbbc': 5, 'kbbc/bbc': 8, 'kbbbck/kk': 9, 'kbck/bbbbk': 18, 
    'kbbbbc/kbf': 16, 'kbbc/kbc': 6, 'bc/bbbc': 10
}

INFOSET_ACTION_STRS_TURN_RIVER = {
'', 'k', 'b', 'kb', 'bb', 'bc/', 'bc/b', 'bc/k', 'bc/kb', 'bc/bb', 'bc/bbb', 'bc/kbb', 'bc/kbbb', 'bc/bbbb', 'kk/', 'kk/k', 'kk/b',
'kk/kb', 'kk/bb', 'kk/kbb', 'kk/bbb', 'kk/kbbb', 'kk/bbbb', 'kbb', 'bbb', 'bbbb', 'bbck/', 'kbbb', 'bbbc/', 'kbck/', 'kbbc/', 'kbck/k', 
'bbbc/b', 'kbck/b', 'bbck/k', 'kbbc/b', 'kbbc/k', 'bbbc/k', 'bbck/b', 'bbbc/kb', 'kbck/bb', 'kbbc/bb', 'bbck/bb', 'bbbc/bb', 'kbck/kb', 
'bbck/kb', 'kbbc/kb', 'kbbc/bbb', 'kbck/kbb', 'bbbc/bbb', 'bbbc/kbb', 'bbck/kbb', 'bbck/bbb', 'kbck/bbb', 'kbbc/kbb', 'bbck/kbbb',
'bbbc/bbbb', 'kbck/bbbb', 'kbbc/kbbb', 'bbck/bbbb', 'kbbc/bbbb', 'bbbc/kbbb', 'kbck/kbbb', 'kbbbb', 'kbbbck/', 'kbbbbc/', 'bbbbck/', 
'kbbbbc/k', 'bbbbck/k', 'bbbbck/b', 'kbbbck/b', 'kbbbck/k', 'kbbbbc/b', 'kbbbbc/kb', 'kbbbck/kb', 'bbbbck/bb', 'kbbbbc/bb', 'kbbbck/bb',
'bbbbck/kb', 'bbbbck/kbb', 'kbbbck/kbb','kbbbbc/bbb', 'bbbbck/bbb', 'kbbbbc/kbb', 'kbbbck/bbb', 'bbbbck/bbbb', 'kbbbbc/bbbb', 'kbbbck/bbbb',
'bbbbck/kbbb', 'kbbbbc/kbbb', 'kbbbck/kbbb'}

INFOSET_LEGAL_ACTIONS_TURN_RIVER = {
    '': ['k', 'b'], 'kbck/bbb': ['b', 'c', 'f'], 'kbbbck/k': ['k', 'b'], 'kbbc/': ['k', 'b'], 'kk/': ['k', 'b'], 'kbck/kbb': ['b', 'c', 'f'], 
    'bbck/bbb': ['b', 'c', 'f'], 'kbck/kbbb': ['c', 'f'], 'kbbbck/bbbb': ['c', 'f'], 'kk/bbbb': ['c', 'f'], 'bbbbck/k': ['k', 'b'], 'bbck/b': ['b', 'c', 'f'], 
    'bbbbck/bbb': ['b', 'c', 'f'], 'bbck/': ['k', 'b'], 'bbbc/bb': ['b', 'c', 'f'], 'bbbbck/kbb': ['b', 'c', 'f'], 'kbbbbc/b': ['b', 'c', 'f'], 'k': ['k', 'b'], 
    'bb': ['b', 'c', 'f'], 'bc/kb': ['b', 'c', 'f'], 'bbbbck/b': ['b', 'c', 'f'], 'bbbbck/kb': ['b', 'c', 'f'], 'kbbbck/kbb': ['b', 'c', 'f'],
    'bbck/bb': ['b', 'c', 'f'], 'kbbbck/b': ['b', 'c', 'f'], 'bbbc/bbb': ['b', 'c', 'f'], 'bbbc/': ['k', 'b'], 'kk/k': ['k', 'b'], 'kbbc/k': ['k', 'b'], 
    'kk/kbb': ['b', 'c', 'f'], 'kbbbbc/kb': ['b', 'c', 'f'], 'b': ['b', 'c', 'f'], 'kbbc/bbb': ['b', 'c', 'f'], 'kbb': ['b', 'c', 'f'], 'kbbbck/kb': ['b', 'c', 'f'],
    'bc/k': ['k', 'b'], 'kbck/bb': ['b', 'c', 'f'], 'bbck/kb': ['b', 'c', 'f'], 'bc/b': ['b', 'c', 'f'], 'bbbc/kb': ['b', 'c', 'f'], 'bbbc/bbbb': ['c', 'f'],
    'kbbbbc/k': ['k', 'b'], 'kbbbbc/': ['k', 'b'], 'bbbbck/kbbb': ['c', 'f'], 'kbck/bbbb': ['c', 'f'], 'bc/bb': ['b', 'c', 'f'], 'kbbbbc/kbbb': ['c', 'f'],
    'bbbbck/bb': ['b', 'c', 'f'], 'kbck/b': ['b', 'c', 'f'], 'bbbc/k': ['k', 'b'], 'kbck/k': ['k', 'b'], 'bbck/kbb': ['b', 'c', 'f'], 'kk/kbbb': ['c', 'f'], 
    'kbbbbc/kbb': ['b', 'c', 'f'], 'bbbc/b': ['b', 'c', 'f'], 'kbbbck/bb': ['b', 'c', 'f'], 'kb': ['b', 'c', 'f'], 'kbbbbc/bbbb': ['c', 'f'], 
    'kbbbbc/bb': ['b', 'c', 'f'], 'kbbbb': ['c', 'f'], 'bbbbck/': ['k', 'b'], 'kk/kb': ['b', 'c', 'f'], 'kbbc/kb': ['b', 'c', 'f'], 'kbbbck/': ['k', 'b'], 
    'kbbc/bb': ['b', 'c', 'f'], 'bbck/kbbb': ['c', 'f'], 'bbbbck/bbbb': ['c', 'f'], 'bc/kbbb': ['c', 'f'], 'kbbc/kbbb': ['c', 'f'], 'bbck/k': ['k', 'b'], 
    'bc/kbb': ['b', 'c', 'f'], 'bc/bbbb': ['c', 'f'], 'kbbbbc/bbb': ['b', 'c', 'f'], 'kbck/kb': ['b', 'c', 'f'], 'bc/bbb': ['b', 'c', 'f'], 'bbck/bbbb': ['c', 'f'],
    'bbbb': ['c', 'f'], 'bc/': ['k', 'b'], 'kbbbck/kbbb': ['c', 'f'], 'kk/bbb': ['b', 'c', 'f'], 'kbbb': ['c', 'b', 'f'], 'bbbc/kbbb': ['c', 'f'], 'kbbc/b': ['b', 'c', 'f'],
    'kbbc/bbbb': ['c', 'f'], 'kk/b': ['b', 'c', 'f'], 'kbbc/kbb': ['b', 'c', 'f'], 'kbck/': ['k', 'b'], 'bbb': ['b', 'c', 'f'], 'bbbc/kbb': ['b', 'c', 'f'], 
    'kbbbck/bbb': ['b', 'c', 'f'], 'kk/bb': ['b', 'c', 'f'], 'k': ['k', 'b']
}# ATTN::convert to TERMINAL: infosets which yield only **1** legal action
TURN_BUCKET_PROBS = {'A': 0.0084, 'B': 0.0499, 'C': 0.0999, 'D': 0.1604, 'E': 0.3452,
                    'F': 0.1183, 'G': 0.0801, 'H': 0.0694, 'I': 0.0356, 'J': 0.0328}

RIVER_BUCKET_PROBS = {'A': 0.0530, 'B': 0.0714, 'C': 0.0752, 'D': 0.0910, 'E': 0.2005,
                    'F': 0.1894, 'G': 0.0965, 'H': 0.0773, 'I': 0.0651, 'J':0.0805}

BUCKET67_PROBS = {
'AA': 0.7589948475, 'AB': 0.0916553826, 'AC': 0.0124202577, 'AD': 0.0113934109, 'AE': 0.0800836841,
'AF': 0.0447861394, 'AG': 0.0006662778, 'AH': 0.0000000000, 'AI': 0.0000000000, 'AJ': 0.0000000000,
'BA': 0.3797354409, 'BB': 0.4027343780, 'BC': 0.0514514583, 'BD': 0.0130393965, 'BE': 0.0537499632,
'BF': 0.0729905565, 'BG': 0.0111108787, 'BH': 0.0031275708, 'BI': 0.0023455783, 'BJ': 0.0097147786,
'CA': 0.1891072652, 'CB': 0.2946473434, 'CC': 0.2364917039, 'CD': 0.0880135677, 'CE': 0.0482965439,
'CF': 0.0787937752, 'CG': 0.0253698253, 'CH': 0.0093373920, 'CI': 0.0079141512, 'CJ': 0.0220284323,
'DA': 0.0483595172, 'DB': 0.1123182379, 'DC': 0.2360030135, 'DD': 0.2671864603, 'DE': 0.1599897154,
'DF': 0.0747110269, 'DG': 0.0361994272, 'DH': 0.0235341217, 'DI': 0.0110460936, 'DJ': 0.0306523864,
'EA': 0.0030489270, 'EB': 0.0087452230, 'EC': 0.0317065591, 'ED': 0.1116373161, 'EE': 0.4596786075,
'EF': 0.2363462525, 'EG': 0.0415730522, 'EH': 0.0317221005, 'EI': 0.0337705984, 'EJ': 0.0417713636,
'FA': 0.0000714776, 'FB': 0.0004392777, 'FC': 0.0009404778, 'FD': 0.0009616656, 'FE': 0.0639382437,
'FF': 0.5571701149, 'FG': 0.2334875141, 'FH': 0.0436448940, 'FI': 0.0534054070, 'FJ': 0.0459409276,
'GA': 0.0000180339, 'GB': 0.0000262194, 'GC': 0.0000259210, 'GD': 0.0000098376, 'GE': 0.0042547348,
'GF': 0.1786430741, 'GG': 0.4287903160, 'GH': 0.2529733491, 'GI': 0.0794771637, 'GJ': 0.0557813503,
'HA': 0.0000000000, 'HB': 0.0000000000, 'HC': 0.0000113795, 'HD': 0.0000343601, 'HE': 0.0003227387,
'HF': 0.0451293553, 'HG': 0.1388347538, 'HH': 0.4625775773, 'HI': 0.2553865689, 'HJ': 0.0977032665,
'IA': 0.0000005762, 'IB': 0.0000008643, 'IC': 0.0000005762, 'ID': 0.0000006722, 'IE': 0.0000020166,
'IF': 0.0158936915, 'IG': 0.0439601226, 'IH': 0.1003621276, 'II': 0.4252538557, 'IJ': 0.4145254970,
'JA': 0.0000000000, 'JB': 0.0000000000, 'JC': 0.0000000000, 'JD': 0.0000000000, 'JE': 0.0000000000,
'JF': 0.0000667048, 'JG': 0.0021741947, 'JH': 0.0123694595, 'JI': 0.1601589181, 'JJ': 0.8252307230
}

NUM_ACTIONS = len(ACTIONS)

# Classes (from your existing code)
class InfoSetActionData:
    def __init__(self, initStratVal):
        self.strategy = initStratVal
        self.util = None
        self.cumulativeGain = initStratVal
        self.cumulativeStrategy = 0

class InfoSetData:
    def __init__(self):
        self.actions = {
            "k": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "c": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "b": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "f": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
        }
        self.beliefs = defaultdict(float)
        self.expectedUtil = None
        self.likelihood = 0

infoSets: dict[str, InfoSetData] = {}  # global
sortedInfoSets = [] # global

class InfoSetActionData:
    def __init__(self, initStratVal):
        self.strategy = initStratVal
        self.util = None
        self.cumulativeGain = initStratVal #initialize it to be consistent with the initial strategies... not sure if this is necessary though
        self.cumulativeStrategy = 0


@lru_cache(maxsize=None)
def getAncestralInfoSetStrs(infoSetStr) -> list[str]:
    # given an infoSet, return all opponent infoSets that can lead to it (e.g. given 'Bpb', return ['Ap','Bp','Cp',...])
    if len(infoSetStr) == 1:
        raise ValueError(f'no ancestors of infoSet={infoSetStr}')
    
    if '/' in infoSetStr:  # flop
        if len(infoSetStr) < 3 or infoSetStr[0] not in RANKS_TURN or infoSetStr[1] not in RANKS_RIVER:
            print(f'Error! getAncestralInfoSetStrs()::invalid infoSetStr: {infoSetStr}')
            sys.exit(-1)
        actionStr = infoSetStr[2:]
        suffix = ''
        if len(actionStr) > 2:
            if actionStr[-1] == '/': 
                suffix = actionStr[:-3] if actionStr[-2] == 'k' else actionStr[:-2]
                return [oppPocket1 + suffix for oppPocket1 in RANKS_TURN]
            else:
                suffix = infoSetStr[2:-1]
        else:
            suffix = infoSetStr[2:-1]
        # Precompute concatenated strings
        return [oppPocket1 + oppPocket2 + suffix for oppPocket1 in RANKS_TURN for oppPocket2 in RANKS_RIVER]
    else:  # preflop
        suffix = infoSetStr[1:-1]
        return [oppPocket + suffix for oppPocket in RANKS_TURN]


def getDescendantInfoSetStrs(infoSetStr, action):
  # given an infoSet and an action to perform at that infoSet, return all opponent infoSets that can result from it 
  # e.g. given infoSetStr='Bpb' and action='p', return ['Apbp','Bpbp','Cpbp',...]
  if '/' in infoSetStr:  # flop
    if len(infoSetStr) < 2 or infoSetStr[0] not in RANKS_TURN or infoSetStr[1] not in RANKS_RIVER:
      print('Error! getAncestralInfoSetStrs()::invalid infoSetStr: {}'.format(infoSetStr))
      sys.exit(-1)
    actionStr = infoSetStr[2:]+action

    return [oppPocket1+oppPocket2+actionStr for oppPocket1 in RANKS_TURN for oppPocket2 in RANKS_RIVER]
  else:  # preflop
    actionStr = infoSetStr[1:]+action
    if actionStr in TERMINAL_ACTION_STRS_TURN:
      if actionStr == 'bc' or actionStr == 'bbbc' or actionStr == 'kbbc' or actionStr == 'kbbbbc' or actionStr == 'kk':
        return [oppPocket + oppPocket2 +actionStr + '/' for oppPocket in RANKS_TURN for oppPocket2 in RANKS_RIVER]
      else:
        return [oppPocket + oppPocket2 +actionStr + 'k/' for oppPocket in RANKS_TURN for oppPocket2 in RANKS_RIVER]
    return [oppPocket+actionStr for oppPocket in RANKS_TURN]


def calcUtilityAtTerminalNode(pocket1, pocket2, action1, playerIdx_, totalBets, playerIdx2return):
  if action1 == 'f':
    return -totalBets if playerIdx2return == playerIdx_ else totalBets
  else:  # showdown
    if RANK2NUM[pocket1] > RANK2NUM[pocket2]:
       return totalBets if playerIdx2return == 0 else -totalBets
    elif RANK2NUM[pocket1] == RANK2NUM[pocket2]: # TODO: better tie breaker?
       return 0
    else:
       return -totalBets if playerIdx2return == 0 else totalBets


def initInfoSets():
  # initialize the infoSet objects.
  for actionsStrs in sorted(INFOSET_ACTION_STRS_TURN_RIVER, key=lambda x:len(x)):
    if '/' in actionsStrs: # flop
      for rank1 in RANKS_TURN:
        for rank2 in RANKS_RIVER:
            infoSetStr = rank1 + rank2 + actionsStrs
            infoSets[infoSetStr] = InfoSetData()
            sortedInfoSets.append(infoSetStr)
    else: # preflop
      for rank in RANKS_TURN:
        infoSetStr = rank + actionsStrs
        infoSets[infoSetStr] = InfoSetData()
        sortedInfoSets.append(infoSetStr)

def initStrategy():
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    actionstr = infoSetStr[1:] if '/' not in infoSetStr else infoSetStr[2:]
    allelgalactions = INFOSET_LEGAL_ACTIONS_TURN_RIVER[actionstr]
    numlegalactions = len(allelgalactions)
    for action in allelgalactions:
      infoSet.actions[action].strategy = 1/numlegalactions


def updateBeliefs():
    for infoSetStr in sortedInfoSets:
        infoSet = infoSets[infoSetStr]
        infoSet.beliefs = defaultdict(float)
        if len(infoSetStr) == 1:
            for oppPocket in RANKS_TURN:
              infoSet.beliefs[oppPocket] = TURN_BUCKET_PROBS[oppPocket] # natural prob of occuring: pre-computed lookup table
        else:
            if infoSetStr[-1] != '/':
              ancestralInfoSetStrs = getAncestralInfoSetStrs(infoSetStr) 
              lastAction = infoSetStr[-1]
              tot = 0  # normalizing factor for strategy (last action)
              all0 = True
              for oppInfoSetStr in ancestralInfoSetStrs:
                  oppInfoSet=infoSets[oppInfoSetStr]
                  # try:
                  #    oppInfoSet=infoSets[oppInfoSetStr]
                  # except KeyError:
                  #    print('infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))
                  if oppInfoSet.actions[lastAction].strategy != 0:
                      all0 = False
                  tot += oppInfoSet.actions[lastAction].strategy * TURN_BUCKET_PROBS[oppInfoSetStr[0]]
              # if all0:
              #   print('Error! updateBeliefs()::all0 | infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))
              #   sys.exit(-1)
              
              # if tot == 0:
              #   print('Error! updateBeliefs()::tot=0 | infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))
              #   print(oppInfoSet.actions[lastAction].strategy)
              #   sys.exit(-1)
              # else:
              #   print(infoSetStr)
              #   print("success")
              if tot == 0:
                for oppo in RANKS_TURN:
                  infoSet.beliefs[oppo] = 0.1
              else:
                for oppInfoSetStr in ancestralInfoSetStrs:
                    oppInfoSet=infoSets[oppInfoSetStr]
                    oppPocket = oppInfoSetStr[1] if '/' in oppInfoSetStr else oppInfoSetStr[0]  # TODO: include both buckets?
                    infoSet.beliefs[oppPocket]+=oppInfoSet.actions[lastAction].strategy * TURN_BUCKET_PROBS[oppInfoSetStr[0]] / tot
            else:
              ancestralInfoSetStrs = getAncestralInfoSetStrs(infoSetStr) 
              lastAction = infoSetStr[-2] if (infoSetStr[-2] == 'c' or (infoSetStr[-3] == 'k' and infoSetStr[-2] == 'k')) else infoSetStr[-3]
              tot = 0  # normalizing factor for strategy (last action)
              tembeliefs = defaultdict(float)
              for oppInfoSetStr in ancestralInfoSetStrs:
                  oppInfoSet=infoSets[oppInfoSetStr]
                  # try:
                  #    oppInfoSet=infoSets[oppInfoSetStr]
                  # except KeyError:
                  #    print('infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))

                  tot += oppInfoSet.actions[lastAction].strategy * TURN_BUCKET_PROBS[oppInfoSetStr[0]]
              if tot == 0:
                for oppo in RANKS_RIVER:
                  infoSet.beliefs[oppo] = 0.1
              else:
                for oppInfoSetStr in ancestralInfoSetStrs:
                    oppInfoSet=infoSets[oppInfoSetStr]
                    oppPocket = oppInfoSetStr[1] if '/' in oppInfoSetStr else oppInfoSetStr[0]  # TODO: include both buckets?
                    tembeliefs[oppPocket]=oppInfoSet.actions[lastAction].strategy * TURN_BUCKET_PROBS[oppInfoSetStr[0]] / tot
                for oppPocket2 in RANKS_RIVER:
                  totbeliefs = 0
                  for oppPocket1 in RANKS_TURN:
                    oppPocket = oppPocket1 + oppPocket2
                    totbeliefs += tembeliefs[oppPocket1] * BUCKET67_PROBS[oppPocket]
                  infoSet.beliefs[oppPocket2] = totbeliefs
    return


def updateUtilitiesForInfoSetStr(infoSetStr):  # infoSetStr example: "Kpb"
    # update the expected utility for the infoSet
    # first note the process is bottom-up, so we need to calculate the expected utility of the infoSet's descendants first
    # we need to go down 2 levels along the game tree, to ensure we get the infoset of same player, that's also why we are adding 
    # utils towards utilsFromInfoSets. Along this way, we also need to consider the terminal nodes
    street = 0
    descendtime = 0
    flopActionStr = ''
    if '/' in infoSetStr:
      street = 1
      flopActionStr = infoSetStr.split('/')[1]
      
    playerIdx = (len(infoSetStr)-1)%2 if street == 0 else len(flopActionStr)%2
    infoSet = infoSets[infoSetStr]
    beliefs = infoSet.beliefs
    cur_actionstr = infoSetStr[1:] if street == 0 else infoSetStr[2:]  #dont consider whether this actionstr is terminal or not?
    for action in INFOSET_LEGAL_ACTIONS_TURN_RIVER[cur_actionstr]:
        utilFromInfoSets,utilFromTerminalNodes=0,0
        actionStr=(infoSetStr[1:]+action) if street == 0 else (infoSetStr[2:]+action)
        if actionStr in TERMINAL_ACTION_STRS_TURN:
          if actionStr == 'bbbc' or actionStr == 'bc' or actionStr == 'kbbc' or actionStr == 'kbbbbc' or actionStr == 'kk':
            actionStr += '/'
          else:
            actionStr += 'k/'
        
        for descendentInfoSetStr in getDescendantInfoSetStrs(infoSetStr,action): # go down the game tree: (infoSetStr='Kpb', action='p') --> ['Qpbp','Jpbp']
            if descendentInfoSetStr[0] not in beliefs and descendentInfoSetStr[1] not in beliefs:
              print('Error! updateUtilitiesForInfoSetStr()::invalid descendentInfoSetStr: {}'.format(descendentInfoSetStr))
              sys.exit(-1)
            probOfThisInfoSet = beliefs[descendentInfoSetStr[0]] if street == 0 else beliefs[descendentInfoSetStr[1]]
            
            # we use pockets when we invoke calcUtilityAtTerminalNode below, 
            # we need to switch the order of the pockets when we're calculating player 2's payoff  
            # also: calcUtilityAtTerminalNode always returns [util_p1, utils_p2] regardless of playerIdx (acting player's index)
            
            if playerIdx == 0:
              pockets=[infoSetStr[0],descendentInfoSetStr[0]] if street == 0 else [infoSetStr[1],descendentInfoSetStr[1]]
            else: # if this is player 2's turn..
              pockets=[descendentInfoSetStr[0],infoSetStr[0]] if street == 0 else [descendentInfoSetStr[1],infoSetStr[1]] 
            
            if actionStr in TERMINAL_ACTION_STRS_TURN_RIVER:
                # choosing this action moves us to a terminal node
                utilFromTerminalNodes+=probOfThisInfoSet*calcUtilityAtTerminalNode(*pockets, actionStr[-1], playerIdx, TERMINAL_CHIPCOUNT_TURN_RIVER[actionStr], playerIdx)
            else:
                # choosing this action moves us to an opponent infoSet where they will choose an action 
                # The opponent's strategy is the same as OURS bc this is self-play
                descendentInfoSet = infoSets[descendentInfoSetStr]
                if actionStr not in INFOSET_LEGAL_ACTIONS_TURN_RIVER:
                  print('Error! updateUtilitiesForInfoSetStr()::invalid actionStr: {}'.format(actionStr))
                  print('infoSetStr: {} | action: {} | descendentInfoSetStr: {}'.format(infoSetStr, action, descendentInfoSetStr))
                  sys.exit(-1)
                for oppAction in INFOSET_LEGAL_ACTIONS_TURN_RIVER[actionStr]:
                    probOfOppAction = descendentInfoSet.actions[oppAction].strategy
                    destinationInfoSetStr = infoSetStr[0] + actionStr + oppAction if street == 0 else infoSetStr[0:2] + actionStr + oppAction
                    destinationActionStr = destinationInfoSetStr[2:] if street == 1 else destinationInfoSetStr[1:]
                    if destinationActionStr in TERMINAL_ACTION_STRS_TURN_RIVER:
                        # our opponent choosing that action moves us to a terminal node
                        utilFromTerminalNodes+=probOfThisInfoSet*probOfOppAction*\
                          calcUtilityAtTerminalNode(*pockets,destinationActionStr[-1], (playerIdx+1)%2, TERMINAL_CHIPCOUNT_TURN_RIVER[destinationActionStr], playerIdx)
                    else:
                        # it's another infoSet, and we've already calculated the expectedUtility of this infoSet
                        # ^^ the utility must've been computed as we are traversing the game tree from bottom up
                        
                        if destinationInfoSetStr[1:] in TERMINAL_ACTION_STRS_TURN:
                          if destinationInfoSetStr[1:] == 'bc' or destinationInfoSetStr[1:] == 'bbbc' or destinationInfoSetStr[1:] == 'kbbc' or destinationInfoSetStr[1:] == 'kbbbbc' or destinationInfoSetStr[1:] == 'kk':
                            destinationInfoSetStr += '/'
                            for oppPocket in RANKS_RIVER:
                              tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                              utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET67_PROBS[infoSetStr[0]+oppPocket]
                          else:
                            destinationInfoSetStr += 'k/'
                            for oppPocket in RANKS_RIVER:
                              tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                              utilFromInfoSets-=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET67_PROBS[infoSetStr[0]+oppPocket]
                        elif destinationInfoSetStr[1:-1] in TERMINAL_ACTION_STRS_TURN and destinationInfoSetStr[-1] == 'k':
                          destinationInfoSetStr += '/'
                          for oppPocket in RANKS_RIVER:
                            tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                            utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET67_PROBS[infoSetStr[0]+oppPocket]
                        if '/' in destinationInfoSetStr and street == 0:
                          if destinationInfoSetStr[-1] != '/' and destinationInfoSetStr[-2] != '/':
                            print('Error! updateUtilitiesForInfoSetStr()::invalid destinationInfoSetStr: {}'.format(destinationInfoSetStr))
                            sys.exit(-1)
                          for oppPocket in RANKS_RIVER:
                            tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                            utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET67_PROBS[infoSetStr[0]+oppPocket]
                        
                        else:
                          utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[destinationInfoSetStr].expectedUtil
        infoSet.actions[action].util=utilFromInfoSets+utilFromTerminalNodes
    
    infoSet.expectedUtil = 0 # Start from nothing, neglecting illegal actions
    for action in INFOSET_LEGAL_ACTIONS_TURN_RIVER[cur_actionstr]:
        actionData = infoSet.actions[action]
        infoSet.expectedUtil+=actionData.strategy*actionData.util  # weighted sum of utils associated with each action
    return descendtime


def calcInfoSetLikelihoods():
  # calculate the likelihood (aka "reach probability") of reaching each infoSet assuming the infoSet "owner" (the player who acts at that infoSet) tries to get there 
  # (and assuming the other player simply plays according to the current strategy)
  
  #for infosets in preflop
  for infoSetStr in sortedInfoSets:
    infoSet=infoSets[infoSetStr]
    infoSet.likelihood=0 #reset it to zero on each iteration so the likelihoods donnot continually grow (bc we're using += below)
    if '/' not in infoSetStr:
      if len(infoSetStr)==1:
        # the likelihood of the top-level infoSets (A, B, C,...) is determined solely by precomputed natural probs.
        infoSet.likelihood=TURN_BUCKET_PROBS[infoSetStr[0]]
      elif len(infoSetStr)==2:  # P2's perspective
        # the second-tier infoSet likelihoods. Note, the second-tier infoSet, e.g., 'Bb', may have resulted from the top-tier infoSets 'A', 'B',...
        # depending on which hand tier player 1 has. The likelihood of 'Bb' is therefore the multiplication of the likelihood along each of these possible paths
        for oppPocket in RANKS_TURN:
          oppInfoSet = infoSets[oppPocket]
          infoSet.likelihood+=oppInfoSet.actions[infoSetStr[-1]].strategy*TURN_BUCKET_PROBS[infoSetStr[0]]*\
            TURN_BUCKET_PROBS[oppPocket]  # once again this is natural prob
      else:
        # For infoSets on the third-tier and beyond, we can use the likelihoods of the infoSets two levels before to calculate their likelihoods.
        # Note, we can't simply use the infoSet one tier before because that's the opponent's infoSet, and the calculation of likelihoods 
        # assumes that the infoSet's "owner" is trying to reach the infoSet. Therefore, when calculating a liklihood for player 1's infoSet, 
        # we can only use the likelihood of an ancestral infoSet if the ancestral infoSet is also "owned" by player 1, and the closest such infoSet is 2 levels above.
        # Note also, that although there can be multiple ancestral infoSets one tier before, there is only one ancestral infoSet two tiers before. 
        # For example, 'Bbc' has one-tier ancestors 'Ab' and 'Bb', but only a single two-tier ancestor: 'B'

        infoSetTwoLevelsAgo = infoSets[infoSetStr[:-2]] # grab the closest ancestral infoSet with the same owner as the infoSet for which we seek to calculate likelihood
        for oppPocket in RANKS_TURN:
          oppInfoSet = infoSets[oppPocket + infoSetStr[1:-1]]
          infoSet.likelihood+=infoSetTwoLevelsAgo.likelihood*infoSetTwoLevelsAgo.actions[infoSetStr[-2]].strategy *TURN_BUCKET_PROBS[oppPocket]*\
            oppInfoSet.actions[infoSetStr[-1]].strategy 
          # ^^ note, each oppInfoSet is essentially slicing up the infoSetTwoLevelsAgo because they're each assuming a specific oppPocket. 
          # ^^ Therefore, we must account for the prob. of each opponent pocket
    else:
      if infoSetStr[-1] == '/':
        # at beginning of flop, the likelihood is determined solely by precomputed transitional probs times its ancestral infoset.
        actionStr = infoSetStr[2:]
        if not (actionStr[-2] == 'k' and actionStr != 'kk') and actionStr[:-1] not in TERMINAL_ACTION_STRS_TURN:
          print('Error! calcInfoSetLikelihoods()::invalid infoSetStr: {}'.format(infoSetStr))
          sys.exit(-1)

        if actionStr[-2] == 'k' and actionStr != 'kk':
          temstr = infoSetStr[0] + infoSetStr[2:-3]
          infoSet.likelihood += infoSets[temstr].likelihood * infoSets[temstr].actions[actionStr[-3]].strategy
          infoSet.likelihood *= BUCKET67_PROBS[infoSetStr[:2]]
        elif actionStr[:-1] in TERMINAL_ACTION_STRS_TURN:
          temstr = infoSetStr[0] + infoSetStr[2:-3]
          for oppPocket1 in RANKS_TURN:            
            temstr2 = oppPocket1 + actionStr[:-2]
            infoSet.likelihood += infoSets[temstr].likelihood * infoSets[temstr].actions[actionStr[-3]].strategy *\
              infoSets[temstr2].actions[actionStr[-2]].strategy * TURN_BUCKET_PROBS[oppPocket1]
          infoSet.likelihood *= BUCKET67_PROBS[infoSetStr[:2]]
        # we are still looking at ancestor two levels ago, but here we need to eliminate the '/' char.
      else:
        actionStr = infoSetStr[2:-1]
        flopActionStr = infoSetStr.split('/')[1]
        if len(flopActionStr) == 1:
          # same reason as preflop round, we should specify the likelihood of first-tier flop infoSets
          actionprob = 0 # prob only considering action str without bucket
          anstralInfoSets = getAncestralInfoSetStrs(infoSetStr)
          for oppoInfoSetStr in anstralInfoSets:
            actionprob += infoSets[oppoInfoSetStr].actions[flopActionStr[-1]].strategy * infoSets[oppoInfoSetStr].likelihood
          infoSet.likelihood = actionprob * TURN_BUCKET_PROBS[infoSetStr[0]] * BUCKET67_PROBS[infoSetStr[:2]]
        else:
          # for second-tier flop infoSets and beyond, we can use the likelihoods of the infoSets two levels before to calculate their likelihoods.
          if infoSetStr[:-2] not in infoSets:
            print('Error! calcInfoSetLikelihoods()::invalid infoSetStr: {}'.format(infoSetStr))
            sys.exit(-1)
          infoSetTwoLevelsAgoFlop = infoSets[infoSetStr[:-2]]
          if infoSetTwoLevelsAgoFlop.likelihood == None:
            print('Error! calcInfoSetLikelihoods()::infoSetTwoLevelsAgoFlop.likelihood is None')
            print('infoSetStr: {}'.format(infoSetStr[:-2]))
            sys.exit(-1)

          for oppPocket1 in RANKS_TURN:
            for oppPocket2 in RANKS_RIVER:
              rank_tem = oppPocket1 + oppPocket2
              oppInfoSet = infoSets[rank_tem + actionStr]
              infoSet.likelihood += infoSetTwoLevelsAgoFlop.likelihood * infoSetTwoLevelsAgoFlop.actions[flopActionStr[-2]].strategy * RIVER_BUCKET_PROBS[oppPocket2] *\
                TURN_BUCKET_PROBS[oppPocket1] * oppInfoSet.actions[flopActionStr[-1]].strategy


def calcGains(cur_t, alpha = 0.5, beta = 2.0):
  # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
  # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
  # we note that in first several iterations the gains are very large, and thus in later iterations its hard to change the strategy since denominator is large
  # so we use alpha to scale the gains, and thus make the strategy converge faster and more to 0
  totAddedGain=0.0
  max_now = 0.0
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    for action in INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[1:]]:
      utilForActionPureStrat = infoSet.actions[action].util 
      gain = max(0, utilForActionPureStrat-infoSet.expectedUtil)
      totAddedGain += gain
      max_now = max(gain, max_now)
      if infoSet.actions[action].cumulativeGain > 0:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, alpha) / (math.pow(cur_t, alpha) + 1)) + gain
      else:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, beta) / (math.pow(cur_t, beta) + 1)) + gain
  return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more) # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)


def updateStrategy(cur_t, gamma = 2.0):
  # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    allLegalActions = INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[1:]] if '/' not in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[2:]]
    totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
    for action in ACTIONS:
        infoSet.actions[action].strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0



# Set up only the critical components needed for optimization
def initialize_test_components():
    infoSets = {}
    sortedInfoSets = []
    for rank in RANKS_TURN:
        infoSetStr = rank
        infoSets[infoSetStr] = InfoSetData()
        sortedInfoSets.append(infoSetStr)
    return infoSets, sortedInfoSets


def run_cfr_with_alpha_beta(alpha, beta=None, num_iterations=3000):
    """
    Run the complete CFR algorithm with the given alpha parameter and return the total gain.
    
    Args:
        alpha: The alpha parameter for calcGains function
        beta: The beta parameter for calcGains function (if None, use alpha/2)
        num_iterations: Number of CFR iterations to run
    
    Returns:
        The final total gain value
    """
    if beta is None:
        beta = alpha / 2.0  # Default relationship between alpha and beta
    
    # Initialize infosets
    global infoSets, sortedInfoSets
    infoSets = {}
    sortedInfoSets = []
    
    # Initialize the CFR algorithm structures
    initInfoSets()
    initStrategy()
    
    # Run CFR for specified iterations
    totGains = []
    for t in range(1, num_iterations + 1):
        updateBeliefs()
        
        # Update utilities by traversing the game tree bottom-up
        for infoSetStr in reversed(sortedInfoSets):
            updateUtilitiesForInfoSetStr(infoSetStr)
        
        # Calculate infoset likelihoods
        calcInfoSetLikelihoods()
        
        # Calculate gains for this iteration - use the passed alpha parameter
        totGain = calcGains(t, alpha=alpha, beta=beta)
        totGains.append(totGain)
        
        # Update strategy for next iteration
        updateStrategy(t)
        
        # Print progress
        if t % 100 == 0:
            print(f"Iteration {t}/{num_iterations}, Alpha: {alpha:.4f}, Beta: {beta:.4f}, Tot_Gain: {totGain:.6f}")
    
    # Return the final total gain
    return totGains[-1]


def test_alpha_values():
    """Test various alpha values."""
    alpha_values = {
        "Low Alpha (1.0)": 1.0,
        "Medium Alpha (2.5)": 2.5,
        "Default Alpha (4.0)": 4.0,
    }
    
    results = {}
    for name, alpha in alpha_values.items():
        print(f"\nTesting {name}")
        start_time = time.time()
        gain = run_cfr_with_alpha_beta(alpha, num_iterations=800)
        elapsed = time.time() - start_time
        results[name] = gain
        print(f"Alpha: {alpha}")
        print(f"Total gain: {gain:.6f}")
        print(f"Time taken: {elapsed:.2f} seconds")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    plt.bar(names, values)
    plt.ylabel('Total Gain (lower is better)')
    plt.title('Comparison of Different Alpha Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('alpha_comparison_full_cfr.png')
    
    # Return the best alpha value
    best_alpha_name = min(results, key=results.get)
    best_alpha = alpha_values[best_alpha_name]
    return best_alpha, results[best_alpha_name]

def print_distribution(distribution, gain=None):
    """
    Print a distribution in a formatted way, with bucket letter labels.
    
    Args:
        distribution: The probability distribution array
        gain: Optional gain value associated with this distribution
    """
    print("{")
    for i, rank in enumerate(RANKS_TURN):
        comma = "," if i < len(RANKS_TURN) - 1 else ""
        print(f"    '{rank}': {distribution[i]:.6f}{comma}")
    print("}")
    
    if gain is not None:
        print(f"Total gain: {gain:.6f}")


def run_bayesian_optimization_for_alpha(initial_alpha=4.0, num_calls=20, optimize_beta=False):
    """
    Run Bayesian optimization to find the optimal alpha (and optionally beta) parameter.
    
    Args:
        initial_alpha: Initial alpha value to start from
        num_calls: Number of optimization iterations
        optimize_beta: Whether to also optimize beta separately (if False, beta = alpha/2)
    
    Returns:
        Optimal alpha (and beta if optimize_beta=True) value and its gain
    """
    print("\n" + "=" * 60)
    print("RUNNING BAYESIAN OPTIMIZATION FOR ALPHA PARAMETER")
    print("=" * 60)
    
    # Define search space with bounds
    if optimize_beta:
        # Optimize both alpha and beta
        space = [
            Real(0.05, 10.0, name='alpha'),
            Real(0.1, 50.0, name='beta')
        ]
    else:
        # Optimize just alpha
        space = [Real(0.5, 10.0, name='alpha')]
    
    # Track the best result found
    best_alpha = initial_alpha
    best_beta = initial_alpha / 2.0
    best_gain = float('inf')
    
    # Define the objective function for Bayesian optimization
    @use_named_args(space)
    def objective_bayesian(**params):
        nonlocal best_alpha, best_beta, best_gain
        
        # Extract parameters
        alpha = params['alpha']
        
        if optimize_beta:
            beta = params['beta']
        else:
            beta = alpha / 2.0  # Default relationship
        
        # Print current parameters
        print(f"\nTesting Alpha: {alpha:.4f}, Beta: {beta:.4f}")
        print(f"Current best - Alpha: {best_alpha:.4f}, Beta: {best_beta:.4f}, Gain: {best_gain:.6f}")
        
        # Run CFR with these parameters
        gain = run_cfr_with_alpha_beta(alpha, beta, num_iterations=2000)
        
        # Update best if improved
        if gain < best_gain:
            best_alpha = alpha
            best_beta = beta
            best_gain = gain
            print(f"\nNEW BEST PARAMETERS FOUND! Alpha: {best_alpha:.4f}, Beta: {best_beta:.4f}, Gain: {best_gain:.6f}")
        else:
            print(f"\nNo improvement. Current best gain: {best_gain:.6f}")
        
        return gain
    
    # Set initial point
    if optimize_beta:
        x0 = [initial_alpha, initial_alpha / 2.0]
    else:
        x0 = [initial_alpha]
    
    # Run Bayesian optimization
    result = gp_minimize(
        objective_bayesian,
        space,
        x0=x0,
        n_calls=num_calls,
        random_state=42,
        verbose=True,
        n_initial_points=3
    )
    
    # Get the best parameters
    if optimize_beta:
        best_alpha, best_beta = result.x
    else:
        best_alpha = result.x[0]
        best_beta = best_alpha / 2.0
    
    # Run a final evaluation with more iterations
    print("\n" + "=" * 60)
    print("RUNNING FINAL EVALUATION WITH 4000 ITERATIONS")
    print("=" * 60)
    
    print(f"\nBest parameters - Alpha: {best_alpha:.4f}, Beta: {best_beta:.4f}")
    
    # Final evaluation with more iterations
    final_gain = run_cfr_with_alpha_beta(best_alpha, best_beta, num_iterations=4000)
    
    # Create plot of the optimization progress
    plot_convergence(result)
    plt.savefig('alpha_optimization_convergence.png')
    
    if optimize_beta:
        return (best_alpha, best_beta), final_gain
    else:
        return best_alpha, final_gain

def main():
    print("=" * 60)
    print("ALPHA PARAMETER OPTIMIZATION FOR CFR ALGORITHM")
    print("=" * 60)
    
    # First, test standard alpha values
    print("\nTesting standard alpha values to find a good starting point...")
    best_standard_alpha, best_standard_gain = test_alpha_values()
    
    print("\nBest standard alpha value:")
    print(f"Alpha: {best_standard_alpha}")
    print(f"Total gain: {best_standard_gain:.6f}")
    
    # Ask if user wants to optimize beta separately as well
    optimize_beta = False  # Change to True if you want to optimize beta separately
    
    # Use Bayesian optimization to find the best alpha value
    if optimize_beta:
        (best_alpha, best_beta), final_gain = run_bayesian_optimization_for_alpha(
            initial_alpha=best_standard_alpha, 
            num_calls=15,
            optimize_beta=True
        )
        
        print("\n" + "=" * 60)
        print("FINAL OPTIMIZED PARAMETERS:")
        print(f"Alpha: {best_alpha:.4f}")
        print(f"Beta: {best_beta:.4f}")
    else:
        best_alpha, final_gain = run_bayesian_optimization_for_alpha(
            initial_alpha=best_standard_alpha, 
            num_calls=15,
            optimize_beta=False
        )
        
        best_beta = best_alpha / 2.0  # Default relationship
        
        print("\n" + "=" * 60)
        print("FINAL OPTIMIZED PARAMETERS:")
        print(f"Alpha: {best_alpha:.4f}")
        print(f"Beta: {best_beta:.4f} (= Alpha/2)")
    
    print(f"Final total gain with optimized parameters: {final_gain:.6f}")
    
    # Compare with default parameters
    default_alpha = 4.0
    default_beta = 2.0
    default_gain = run_cfr_with_alpha_beta(default_alpha, default_beta, num_iterations=4000)
    print(f"Total gain with default parameters (Alpha=4.0, Beta=2.0): {default_gain:.6f}")
    
    if default_gain > 0 and final_gain > 0:
        improvement = (default_gain - final_gain) / default_gain * 100
        print(f"Improvement: {improvement:.2f}%")
    
    # Save the optimized parameters to a file
    optimized_params = {
        'alpha': best_alpha,
        'beta': best_beta,
        'gain': final_gain
    }
    with open('optimized_alpha_beta_params.pkl', 'wb') as f:
        pickle.dump(optimized_params, f)
    print("Optimized parameters saved to 'optimized_alpha_beta_params.pkl'")

if __name__ == "__main__":
    main()