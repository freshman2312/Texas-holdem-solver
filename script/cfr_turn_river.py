# solve turn and river streets together. 
# turns out to be a little worse than solving them separately.
# maybe a better tie breaker is needed
# also consider using more actions, or accelerating the cfr process, since the current will take more than 1 sec for one iteration

from __future__ import annotations
# from copy import deepcopy
from compute_score import show_cards, card2rank, rank2str, card2row, str2rank, compute_score
# import cProfile
# https://stackoverflow.com/questions/10326936/sort-cprofile-output-by-percall-when-profiling-a-python-script
# cd /home/test/Documents/GZ-texasholdem241202/CFR
# python3 -m cProfile -s tottime good_cfr_texas.py
# visualize profiling results with snakeviz: 
# installation: python3 -m pip install snakeviz
# https://gist.github.com/matthewfeickert/dd6e7a5fda1ed1e3498219dafe5f9ea1
# snakeviz good_cfr_texas.prof
from random import randint, choice, shuffle
import sys
from itertools import combinations
from collections import defaultdict
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
import math




descendtime = 0
infoSets: dict[str, InfoSetData] = {}  # global
sortedInfoSets = [] # global
gainhistory: dict[tuple[str, str], list[int]] = defaultdict(list[int])  # global

# ASSUMPTION: consider PREFLOP and FLOP **only**; That is: game ENDS on FLOP

# RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# RANKS_RIVER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
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



NUM_ACTIONS = len(ACTIONS)

# PREFLOP_WR = {
#   '22': 51, '33': 55, '44': 58, '55': 61, '66': 64, '77': 67, '88': 69, '99': 72, 'TT': 75, 'JJ': 78, 'QQ': 80, 'KK': 83, 'AA': 85,
#   '23': 35, '24': 36, '25': 37, '26': 37, '27': 37, '28': 40, '29': 42, '2T': 44, '2J': 47, '2Q': 49, '2K': 53, '2A': 57, 
#   '34': 38, '35': 39, '36': 39, '37': 39, '38': 40, '39': 43, '3T': 45, '3J': 48, '3Q': 50, '3K': 54, '3A': 58, 
#   '45': 41, '46': 41, '47': 41, '48': 42, '49': 43, '4T': 46, '4J': 48, '4Q': 51, '4K': 54, '4A': 59, 
#   '56': 43, '57': 43, '58': 44, '59': 45, '5T': 47, '5J': 49, '5Q': 52, '5K': 55, '5A': 60, 
#   '67': 45, '68': 46, '69': 47, '6T': 48, '6J': 50, '6Q': 53, '6K': 56, '6A': 59, 
#   '78': 47, '79': 48, '7T': 50, '7J': 52, '7Q': 54, '7K': 57, '7A': 60, 
#   '89': 50, '8T': 52, '8J': 53, '8Q': 55, '8K': 58, '8A': 61, 
#   '9T': 53, '9J': 55, '9Q': 57, '9K': 59, '9A': 62, 
#   'TJ': 57, 'TQ': 59, 'TK': 61, 'TA': 64, 
#   'JQ': 59, 'JK': 62, 'JA': 65, 
#   'QK': 62, 'QA': 65, 'KA': 66, # ------------------------------------------------------------------------------------------------
#   '23s': 39, '24s': 40, '25s': 41, '26s': 40, '27s': 41, '28s': 43, '29s': 45, '2Ts': 47, '2Js': 50, '2Qs': 52, '2Ks': 55, '2As': 59,
#   '34s': 42, '35s': 43, '36s': 42, '37s': 43, '38s': 43, '39s': 46, '3Ts': 48, '3Js': 50, '3Qs': 53, '3Ks': 56, '3As': 60,
#   '45s': 44, '46s': 44, '47s': 45, '48s': 45, '49s': 46, '4Ts': 49, '4Js': 51, '4Qs': 54, '4Ks': 57, '4As': 61,
#   '56s': 46, '57s': 46, '58s': 47, '59s': 48, '5Ts': 49, '5Js': 52, '5Qs': 55, '5Ks': 58, '5As': 62,
#   '67s': 48, '68s': 49, '69s': 50, '6Ts': 51, '6Js': 53, '6Qs': 55, '6Ks': 58, '6As': 62,
#   '78s': 50, '79s': 51, '7Ts': 53, '7Js': 54, '7Qs': 56, '7Ks': 59, '7As': 63,
#   '89s': 53, '8Ts': 54, '8Js': 56, '8Qs': 58, '8Ks': 60, '8As': 63,
#   '9Ts': 56, '9Js': 57, '9Qs': 59, '9Ks': 61, '9As': 64,
#   'TJs': 59, 'TQs': 61, 'TKs': 63, 'TAs': 66,
#   'JQs': 61, 'JKs': 64, 'JAs': 66,
#   'QKs': 64, 'QAs': 67, 'KAs': 68,
# }


TURN_BUCKET_PROBS = {'A': 0.0084, 'B': 0.0499, 'C': 0.0999, 'D': 0.1604, 'E': 0.3452,
                    'F': 0.1183, 'G': 0.0801, 'H': 0.0694, 'I': 0.0356, 'J': 0.0328}

# TURN_BUCKET_PROBS = {'H': 0.1260, 'F': 0.1125, 'E': 0.0862, 'C': 0.0388, 
#                     'D': 0.0616, 'G': 0.2716, 'K': 0.0473, 'N': 0.0310, 
#                     'O': 0.0161, 'M': 0.0213, 'L': 0.0441, 'B': 0.0167, 
#                     'A': 0.0028, 'I': 0.0658, 'J': 0.0582} 
# Natural prob of each bucket for turn round. that is occurence of each bucket / total number of hands

RIVER_BUCKET_PROBS = {'A': 0.0530, 'B': 0.0714, 'C': 0.0752, 'D': 0.0910, 'E': 0.2005,
                    'F': 0.1894, 'G': 0.0965, 'H': 0.0773, 'I': 0.0651, 'J':0.0805}
# bucket prob of cubic prob distribution of 10bucs

# RIVER_BUCKET_PROBS = {'F': 0.0707, 'G': 0.0944, 'A': 0.0336, 'H': 0.2082, 'I': 0.0872, 
#     'M': 0.0397, 'B': 0.0406, 'C': 0.0503, 'D': 0.0427, 'K': 0.0512, 'L': 0.0506,
#     'E': 0.0529, 'N': 0.0491, 'J': 0.0721, 'O': 0.0569}


# conditional probability of each bucket given the previous bucket


# conditional probability of each bucket given the previous bucket

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


def load_pickled_data(file_path):
    """
    Load data from a pickle file.
    
    Args:
        file_path (str): The path to the pickle file.
        
    Returns:
        object: The deserialized object from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully loaded data from '{file_path}'.")
        print(f"Data type: {type(data)}")
        print(len(data))
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except pickle.UnpicklingError:
        print(f"Error: The file '{file_path}' could not be unpickled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
TRANS_PROB = load_pickled_data("transprob.pkl")

# the values we're updating are all indexed by the infoSet, so this class will store all the data for a particular infoSet in a single object
class InfoSetData:
    def __init__(self):
        # initialize the strategy for the infoSet to be uniform random (e.g. k: 1/4, c: 1/4, b: 1/4, f: 1/4)
        self.actions: dict[str, InfoSetActionData] = {
            "k": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "c": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "b": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "f": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
        }
        self.beliefs: dict[str, float] = defaultdict(float)  # opponent pocket card -> belief
        self.expectedUtil: float = None
        self.likelihood: float = 0

    @staticmethod
    def printInfoSetDataTable(infoSets: dict[str,InfoSetData], client_hand_rank_, client_pos_):
        print()
        # print the various values for the infoSets in a nicely formatted table
        rows=[]
        for infoSetStr in sortedInfoSets:
          if '/' not in infoSetStr:
            if infoSetStr[0] == client_hand_rank_ and (len(infoSetStr) + 1) % 2 == client_pos_\
                and len(INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[1:]]) > 1:
                infoSet = infoSets[infoSetStr]
                row=[infoSetStr,*infoSet.getStrategyTableData(),
                     infoSetStr,f'{infoSet.expectedUtil:.2f}',f'{infoSet.likelihood*100:.2f}%',
                     infoSetStr,*infoSet.getGainTableData()]
          else:
            if infoSetStr[0] == client_hand_rank_ and (len(infoSetStr) + 1) % 2 == client_pos_\
                and len(INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[2:]]) > 1:
                infoSet = infoSets[infoSetStr]
                row=[infoSetStr,*infoSet.getStrategyTableData(),
                     infoSetStr,f'{infoSet.expectedUtil:.2f}',f'{infoSet.likelihood*100:.2f}%',
                     infoSetStr,*infoSet.getGainTableData()]

                rows.append(row)
        
        headers = ["InfoSet","Actn:Check", "Actn:Call", "Actn:Bet", "Actn:Fold", 
                   "---", "ExpectedUtil","Likelihood",
                   "---","totGain:Check","totGain:Call","totGain:Bet","totGain:Fold"]

        # Calculate maximum width for each column
        max_widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

        # Print headers
        header_line = "   ".join(header.ljust(width) for header, width in zip(headers, max_widths))
        print(header_line)

        # Print separator
        separator_line = "-" * (sum(max_widths)+3*len(headers))
        print(separator_line)

        # Print rows
        for row in rows:
            row_line = "   ".join(str(cell).ljust(width) for cell, width in zip(row, max_widths))
            print(row_line)


    def getStrategyTableData(self):
        return [f'{self.actions[action].strategy*100:.0f}%' for action in ACTIONS]
    

    def getUtilityTableData(self):
        return ['--' if self.actions[action].util is None else f'{self.actions[action].util:.2f}' for action in ACTIONS]
    

    def getGainTableData(self):
        return [f'{self.actions[action].cumulativeGain:.2f}' for action in ACTIONS]
    

    def getBeliefTableData(self):
        return [f'{self.beliefs[oppPocket]:.2f}' for oppPocket in self.beliefs.keys()]
    # ^^ all 4 are used by `printInfoSetDataTable()`


# Each infoSet has at least one action that a player can choose to perform at it, and the infoSet-action pairs have various values we're updating 
# This class will store all these action-specific values for the infoSet
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
      gainhistory[(infoSetStr, action)].append(gain)
      totAddedGain += gain
      max_now = max(gain, max_now)
      if infoSet.actions[action].cumulativeGain > 0:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, alpha) / (math.pow(cur_t, alpha) + 1)) + gain
      else:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, beta) / (math.pow(cur_t, beta) + 1)) + gain
  print(max_now)
  return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)

# def calcGains(cur_t, alpha = 1.5, beta = 0):
#   # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
#   # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
#   # we note that in first several iterations the gains are very large, and thus in later iterations its hard to change the strategy since denominator is large
#   # so we use alpha to scale the gains, and thus make the strategy converge faster and more to 0
#   totAddedGain=0.0
#   max_now = 0.0
#   for infoSetStr in sortedInfoSets:
#     infoSet = infoSets[infoSetStr]
#     for action in INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[1:]]:
#       utilForActionPureStrat = infoSet.actions[action].util 
#       gain = max(0, utilForActionPureStrat-infoSet.expectedUtil)
#       totAddedGain+=gain
#       max_now = max(gain, max_now)
#       infoSet.actions[action].cumulativeGain += gain
#   print(max_now)
#   return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)


# def updateStrategy(cur_t, gamma = 100.0):
#   # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
#   for infoSetStr in sortedInfoSets:
#     infoSet = infoSets[infoSetStr]
#     allLegalActions = INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[1:]]

#     totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
#     if totGains == 0:
#       print('Error! updateStrategy()::totGains=0 | infoSetStr: {}'.format(infoSetStr))
#       print(infoSet.expectedUtil)
#       for action in allLegalActions:
#         print(infoSet.actions[action].strategy)
#         print(infoSet.actions[action].util)      
#       sys.exit(-1)
#     all0 = True
#     likelihood = infoSet.likelihood
#     totStrategy = 0.0
#     for action in ACTIONS:
#         cur_strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0
#         infoSet.actions[action].cumulativeStrategy = infoSet.actions[action].cumulativeStrategy * math.pow((cur_t - 1) / cur_t, gamma) + cur_strategy * likelihood
#         # infoSet.actions[action].cumulativeStrategy = cur_strategy * likelihood

#         totStrategy += infoSet.actions[action].cumulativeStrategy
#         if infoSet.actions[action].cumulativeStrategy != 0:
#             all0 = False
#     if all0:
#       print('Error! updateStrategy()::all0 | infoSetStr: {}'.format(infoSetStr))
#       for action in allLegalActions:
#         print(likelihood)
#         print(infoSet.actions[action].strategy)
#         print(infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0)
#         print(infoSet.actions[action].cumulativeStrategy)
#       sys.exit(-1)
#     for action in ACTIONS:
#         infoSet.actions[action].strategy = infoSet.actions[action].cumulativeStrategy / totStrategy if action in allLegalActions else 0.0 #why using cumulativeGain instead of gain, aka counterfactual regret?
#         if infoSet.actions[action].strategy != 0:
#             all0 = False

def updateStrategy(cur_t, gamma = 2.0):
  # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    allLegalActions = INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[1:]] if '/' not in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN_RIVER[infoSetStr[2:]]
    totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
    for action in ACTIONS:
        infoSet.actions[action].strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0


def plot_strategy_evolution(iterations, first_strategy_history, second_strategy_history):
    """
    Plot the evolution of first-level and second-level strategies across iterations,
    showing 6 plots at a time with clear labeling.
    """
    ranks = RANKS_TURN
    actions = ['k', 'c', 'b', 'f']
    colors = {'k': 'blue', 'c': 'green', 'b': 'red', 'f': 'purple'}
    
    # Calculate how many pages we need
    plots_per_page = 3
    num_ranks = len(ranks)
    num_pages = (num_ranks + plots_per_page - 1) // plots_per_page
    
    for page in range(num_pages):
        # Calculate which ranks to show on this page
        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, num_ranks)
        page_ranks = ranks[start_idx:end_idx]
        
        # How many plots on this page
        num_plots = len(page_ranks)
        
        # Create figure with appropriate size
        fig, axes = plt.subplots(num_plots, 2, figsize=(16, 3 * num_plots))
        
        # Handle case when there's only one plot (axes is 1D)
        if num_plots == 1:
            axes = axes.reshape(1, 2)
        
        # Plot the strategies for each rank on this page
        for i, rank in enumerate(page_ranks):
            # Plot first-level strategy (left column)
            for action in actions:
                axes[i, 0].plot(iterations, first_strategy_history[rank][action],
                               label=f'{action}', color=colors[action], marker='o', markersize=3)
            
            axes[i, 0].set_title(f'Rank {rank} First-Level Strategy')
            axes[i, 0].set_xlabel('Iterations')
            axes[i, 0].set_ylabel('Strategy Probability')
            axes[i, 0].set_ylim(-0.05, 1.05)
            axes[i, 0].grid(True)
            axes[i, 0].legend()
            
            # Plot second-level strategy (right column)
            for action in actions:
                axes[i, 1].plot(iterations, second_strategy_history[rank][action],
                               label=f'{action}', color=colors[action], marker='o', markersize=3)
            
            axes[i, 1].set_title(f'Rank {rank} Second-Level Strategy')
            axes[i, 1].set_xlabel('Iterations')
            axes[i, 1].set_ylabel('Strategy Probability')
            axes[i, 1].set_ylim(-0.05, 1.05)
            axes[i, 1].grid(True)
            axes[i, 1].legend()
            
        plt.tight_layout()
        plt.savefig(f'rank_strategy_evolution_page{page+1}.png')
        plt.show()
        
        print(f"Page {page+1}/{num_pages}: Showing strategy evolution for ranks {', '.join(page_ranks)}")
        print(f"First-level strategies show infosets: {', '.join([rank for rank in page_ranks])}")
        print(f"Second-level strategies show responses to opponent bets/raises for ranks {', '.join(page_ranks)}")
        print("-" * 80)


def plot_infoset_strategies(iterations, infoset_action_str, strategy_history_dict):
    """
    Plot strategy evolution for all 15 buckets (A-O) for a specific infoset action string.
    
    Args:
        iterations: List of iteration numbers
        infoset_action_str: The action string part of the infoset (e.g., '', 'b', 'k')
        strategy_history_dict: Dictionary with strategy history for all ranks and actions
    """
    ranks = RANKS_TURN  # All 15 buckets from A to O
    actions = ['k', 'c', 'b', 'f']
    colors = {'k': 'blue', 'c': 'green', 'b': 'red', 'f': 'purple', 'r': 'orange'}
    
    # Create 5x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    
    # Flatten the axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Set title for the entire figure
    fig.suptitle(f'Strategy Evolution for Infoset Action "{infoset_action_str}"', fontsize=16, y=0.99)
    
    # Plot each bucket's strategy evolution
    for i, rank in enumerate(ranks):
        ax = axes_flat[i]
        infoset_str = rank + infoset_action_str
        
        # Plot strategy for each action
        for action in actions:
            if infoset_str in strategy_history_dict and action in strategy_history_dict[infoset_str]:
                ax.plot(iterations, strategy_history_dict[infoset_str][action], 
                        label=f'{action}', color=colors[action], marker='o', markersize=2)
        
        ax.set_title(f'Bucket {rank}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Strategy Probability')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend()
    
    # Remove any unused subplots
    for i in range(len(ranks), len(axes_flat)):
        fig.delaxes(axes_flat[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title
    plt.savefig(f'strategy_for_infoset_{infoset_action_str.replace("/", "_")}.png')
    plt.show()

def main():
    start = time.time()
    initInfoSets()
    # print('>>>SORTED INFO SETS>>>\n{}\n<<<'.format(sortedInfoSets))
    initStrategy()

    numIterations=1000  # 10k converges a lot better; 1k ~ 40s
    totGains = []
    iterations = []
    time_belief = 0
    time_util = 0
    time_likelihood = 0
    time_gain = 0
    time_strategy = 0
    descendtot = 0
    actions = ['k', 'c', 'b', 'f']
    ranks = RANKS_TURN
    first_strategy_history = {rank: {action: [] for action in actions} for rank in ranks}
    second_strategy_history = {rank: {action: [] for action in actions} for rank in ranks}
    iterationstra = []
    save_interval = 10
    
    strategy_history = {}
    for infoset_str in sortedInfoSets:
        strategy_history[infoset_str] = {action: [] for action in ACTIONS}
        
    # only plot the gain from every xth iteration (in order to lessen the amount of data that needs to be plotted)
    numGainsToPlot=100
    gainGrpSize = numIterations//numGainsToPlot 
    if gainGrpSize==0:
       gainGrpSize=1

    for i in range(numIterations):
        now = time.time()
        if i % 10 == 0:
            print(f'ITERATION: {i}/{numIterations} | Time elapsed: {now-start:.2f}s')
        print(f'ITERATION: {i}/{numIterations}')
        belief_start = time.time()
        updateBeliefs()
        time_belief += time.time() - belief_start

        util_start = time.time()
        for infoSetStr in reversed(sortedInfoSets):  # game tree: from bottom up
            temdescend = updateUtilitiesForInfoSetStr(infoSetStr)
            descendtot += temdescend
        time_util += time.time() - util_start

        likelihood_start = time.time()
        calcInfoSetLikelihoods()
        time_likelihood += time.time() - likelihood_start
        
        gain_start = time.time()
        cur_t = i + 1
        totGain = calcGains(cur_t)
        time_gain += time.time() - gain_start
        
        if i%gainGrpSize==0: # every 10 or 100 or x rounds, save off the gain so we can plot it afterwards and visually see convergence
            totGains.append(totGain)
            iterations.append(i)
            # print(f'TOT_GAIN {totGain: .3f}  @{i}/{numIterations}')
            
        strategy_start = time.time()
        updateStrategy(cur_t + 1)
        time_strategy += time.time() - strategy_start
        
        if i % save_interval == 0:
            iterationstra.append(i)
            
            for infoset_str in sortedInfoSets:
                infoset = infoSets[infoset_str]
                for action in ACTIONS:
                    strategy_history[infoset_str][action].append(infoset.actions[action].strategy)
          
        # if i % 10 == 0:
        #   levelprob = defaultdict(float)
        #   levelprob2 = defaultdict(float)
        #   for infoSetStr in sortedInfoSets:
        #       actionstr = infoSetStr[1:] if '/' not in infoSetStr else infoSetStr[2:]
        #       level = len(actionstr) if '/' not in infoSetStr else len(actionstr.split('/')[1]) + 10
        #       level2 = len(actionstr) if '/' not in infoSetStr else len(actionstr) + 10
        #       levelprob[level] += infoSets[infoSetStr].likelihood
        #       levelprob2[level2] += infoSets[infoSetStr].likelihood
        #       if '/' not in infoSetStr:
        #         print(infoSets[infoSetStr].beliefs)
        #       sum = 0
        #       for opp in RANKS_RIVER:
        #         sum += infoSets[infoSetStr].beliefs[opp]
        #       if sum > 1.05:
        #         print("too big sum")
        #         print(sum)
        #         sys.exit(-1)
        #   print(levelprob)
        #   print(levelprob2)


    # profiler.disable()
    # profiler.dump_stats("good_cfr_texas.prof")

    with open('infoSets_TURN_RIVER.pkl','wb') as f:
        pickle.dump(infoSets,f)
    print('>>>INFO SETS SAVED TO FILE>>>')
    print('\ntotGains:', totGains)
    
    print(descendtot)
    print(f'BELIEF: {time_belief:.2f}s | UTIL: {time_util:.2f}s | LIKELIHOOD: {time_likelihood:.2f}s | GAIN: {time_gain:.2f}s | STRATEGY: {time_strategy:.2f}s')
    print(f'TOTAL TIME: {time.time()-start:.2f}s')
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, totGains, label='Total Gain', marker='o')

    plt.xlabel('Iterations (in units of 100)')
    plt.ylabel('Total Gain')
    plt.title('Iterations vs Total Gain in CFR Algorithm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig('cfr_convergence.png')  # PNG format
    plt.show()
    
    # plot_strategy_evolution(iterationstra, first_strategy_history, second_strategy_history)
    infoset_action_strs = sorted(list(INFOSET_ACTION_STRS_TURN_RIVER))
    for action_str in infoset_action_strs:
        print(f"Plotting strategies for infoset action string: '{action_str}'")
        plot_infoset_strategies(iterationstra, action_str, strategy_history)


if __name__ == '__main__':
    main()
