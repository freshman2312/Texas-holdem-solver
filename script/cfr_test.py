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

RANKS_PREFLOP = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]  # based on preflop chart: reflects WR (in ascending)
RANKS_FLOP = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I", 10: "J"}
ACTIONS = ["k", "c", "b", "f"]  # {k: check, c: call, b: bet/raise/all-in, f: fold}

TERMINAL_ACTION_STRS_PREFLOP = {'bc', 'bbc', 'bbbc', 'bbbbc'}

TERMINAL_ACTION_STRS_PREFLOP_3b = {'bc', 'bbc', 'bbbc'}


TERMINAL_ACTION_STRS = {  # preflop + flop, total=5+60
  'f', 'bf', 'bbf', 'bbbf', 'bbbbf', 'bc/kk', 'bc/kbf', 'bc/kbc', 'bc/kbbc', 'bc/kbbf', 'bc/kbbbf', 'bc/kbbbc', 'bc/bf', 'bc/bc', 'bc/bbf', 'bc/bbc', 'bc/bbbf', 'bc/bbbc', 'bc/bbbbf', 'bc/bbbbc', 'bbck/kk', 'bbck/kbf', 'bbck/kbc', 'bbck/kbbc', 'bbck/kbbf', 'bbck/kbbbf', 'bbck/kbbbc', 'bbck/bf', 'bbck/bc', 'bbck/bbf', 'bbck/bbc', 'bbck/bbbf', 'bbck/bbbc', 'bbck/bbbbf', 'bbck/bbbbc', 'bbbc/kk', 'bbbc/kbf', 'bbbc/kbc', 'bbbc/kbbc', 'bbbc/kbbf', 'bbbc/kbbbf', 'bbbc/kbbbc', 'bbbc/bf', 'bbbc/bc', 'bbbc/bbf', 'bbbc/bbc', 'bbbc/bbbf', 'bbbc/bbbc', 'bbbc/bbbbf', 'bbbc/bbbbc', 'bbbbck/kk', 'bbbbck/kbf', 'bbbbck/kbc', 'bbbbck/kbbc', 'bbbbck/kbbf', 'bbbbck/kbbbf', 'bbbbck/kbbbc', 'bbbbck/bf', 'bbbbck/bc', 'bbbbck/bbf', 'bbbbck/bbc', 'bbbbck/bbbf', 'bbbbck/bbbc', 'bbbbck/bbbbf', 'bbbbck/bbbbc'
} # terminal action paths where all decisions have already been made (terminal nodes are NOT considered infoSets here, bc no decision needs to be made)
# TERMINAL_CHIPCOUNT = {
#   'f': 1, 'bf': 1, 'bbf': 2, 'bbbf': 4, 'bbbbf': 8, 'bc/kk': 2, 'bc/kbf': 2, 'bc/kbc': 4, 'bc/kbbc': 8, 'bc/kbbf': 4, 'bc/kbbbf': 8, 
#   'bc/kbbbc': 16, 'bc/bf': 2, 'bc/bc': 4, 'bc/bbf': 4, 'bc/bbc': 8, 'bc/bbbf': 8, 'bc/bbbc': 16, 'bc/bbbbf': 16, 'bc/bbbbc': 32, 'bbck/kk': 4,
#   'bbck/kbf': 4, 'bbck/kbc': 8, 'bbck/kbbc': 16, 'bbck/kbbf': 8, 'bbck/kbbbf': 16, 'bbck/kbbbc': 32, 'bbck/bf': 4, 'bbck/bc': 8, 'bbck/bbf': 8,
#   'bbck/bbc': 16, 'bbck/bbbf': 16, 'bbck/bbbc': 32, 'bbck/bbbbf': 32, 'bbck/bbbbc': 64, 'bbbc/kk': 8, 'bbbc/kbf': 8, 'bbbc/kbc': 16, 
#   'bbbc/kbbc': 32, 'bbbc/kbbf': 16, 'bbbc/kbbbf': 32, 'bbbc/kbbbc': 64, 'bbbc/bf': 8, 'bbbc/bc': 16, 'bbbc/bbf': 16, 'bbbc/bbc': 32, 
#   'bbbc/bbbf': 32, 'bbbc/bbbc': 64, 'bbbc/bbbbf': 64, 'bbbc/bbbbc': 128, 'bbbbck/kk': 16, 'bbbbck/kbf': 16, 'bbbbck/kbc': 32, 'bbbbck/kbbc': 64,
#   'bbbbck/kbbf': 32, 'bbbbck/kbbbf': 64, 'bbbbck/kbbbc': 128, 'bbbbck/bf': 16, 'bbbbck/bc': 32, 'bbbbck/bbf': 32, 'bbbbck/bbc': 64, 
#   'bbbbck/bbbf': 64, 'bbbbck/bbbc': 128, 'bbbbck/bbbbf': 128, 'bbbbck/bbbbc': 256
# }
TERMINAL_CHIPCOUNT = {
  'f': 1, 'bf': 1, 'bbf': 2, 'bbbf': 4, 'bbbbf': 8, 'bc/kk': 2, 'bc/kbf': 2, 'bc/kbc': 4, 'bc/kbbc': 6, 'bc/kbbf': 4, 'bc/kbbbf': 6, 
  'bc/kbbbc': 10, 'bc/bf': 2, 'bc/bc': 4, 'bc/bbf': 4, 'bc/bbc': 6, 'bc/bbbf': 6, 'bc/bbbc': 10, 'bc/bbbbf': 10, 'bc/bbbbc': 18, 'bbck/kk': 4,
  'bbck/kbf': 4, 'bbck/kbc': 6, 'bbck/kbbc': 8, 'bbck/kbbf': 6, 'bbck/kbbbf': 8, 'bbck/kbbbc': 12, 'bbck/bf': 4, 'bbck/bc': 6, 'bbck/bbf': 6,
  'bbck/bbc': 8, 'bbck/bbbf': 8, 'bbck/bbbc': 12, 'bbck/bbbbf': 12, 'bbck/bbbbc': 20, 'bbbc/kk': 8, 'bbbc/kbf': 8, 'bbbc/kbc': 10, 
  'bbbc/kbbc': 12, 'bbbc/kbbf': 10, 'bbbc/kbbbf': 12, 'bbbc/kbbbc': 16, 'bbbc/bf': 8, 'bbbc/bc': 10, 'bbbc/bbf': 10, 'bbbc/bbc': 12, 
  'bbbc/bbbf': 12, 'bbbc/bbbc': 16, 'bbbc/bbbbf': 16, 'bbbc/bbbbc': 24, 'bbbbck/kk': 16, 'bbbbck/kbf': 16, 'bbbbck/kbc': 18, 'bbbbck/kbbc': 20,
  'bbbbck/kbbf': 18, 'bbbbck/kbbbf': 20, 'bbbbck/kbbbc': 24, 'bbbbck/bf': 16, 'bbbbck/bc': 18, 'bbbbck/bbf': 18, 'bbbbck/bbc': 20, 
  'bbbbck/bbbf': 20, 'bbbbck/bbbc': 24, 'bbbbck/bbbbf': 24, 'bbbbck/bbbbc': 32
}
INFOSET_ACTION_STRS = { # preflop + flop, total=5+36
  '', 'b', 'bb', 'bbb', 'bbbb', 'bc/', 'bc/b', 'bc/bb', 'bc/bbb', 'bc/bbbb', 'bc/k', 'bc/kb', 'bc/kbb', 'bc/kbbb', 'bbck/', 'bbck/b', 'bbck/bb', 
  'bbck/bbb', 'bbck/bbbb', 'bbck/k', 'bbck/kb', 'bbck/kbb', 'bbck/kbbb', 'bbbc/', 'bbbc/b', 'bbbc/bb', 'bbbc/bbb', 'bbbc/bbbb', 'bbbc/k', 'bbbc/kb', 
  'bbbc/kbb', 'bbbc/kbbb', 'bbbbck/', 'bbbbck/b', 'bbbbck/bb', 'bbbbck/bbb', 'bbbbck/bbbb', 'bbbbck/k', 'bbbbck/kb', 'bbbbck/kbb', 'bbbbck/kbbb'
} # action paths where a decision still needs to be made by one of the players (i.e. actions paths that end on an infoSet)

INFOSET_ACTION_STRS_3b = {
    '', 'b', 'bb', 'bbb', 'bc/', 'bc/b', 'bc/bb', 'bc/bbb', 'bc/k', 'bc/kb', 'bc/kbb', 'bc/kbbb', 'bbck/',
    'bbck/b', 'bbck/bb', 'bbck/bbb', 'bbck/k', 'bbck/kb', 'bbck/kbb', 'bbck/kbbb', 'bbbc/', 'bbbc/b', 'bbbc/bb',
    'bbbc/bbb', 'bbbc/k', 'bbbc/kb', 'bbbc/kbb', 'bbbc/kbbb'
}

INFOSET_LEGAL_ACTIONS = { # Is 'b' really an infoset? should it be 'c' instead? (UtG's first action)
  '': ['b', 'f'], 'b': ['c', 'b', 'f'], 'bb': ['c', 'b', 'f'], 'bbb': ['c', 'b', 'f'], 'bbbb': ['c', 'f'], 'bc/': ['k', 'b'], 
  'bc/b': ['c', 'b', 'f'], 'bc/bb': ['c', 'b', 'f'], 'bc/bbb': ['c', 'b', 'f'], 'bc/bbbb': ['c', 'f'], 'bc/k': ['k', 'b'], 
  'bc/kb': ['f', 'c', 'b'], 'bc/kbb': ['f', 'c', 'b'], 'bc/kbbb': ['f', 'c'], 'bbck/': ['k', 'b'], 'bbck/b': ['c', 'b', 'f'], 
  'bbck/bb': ['c', 'b', 'f'], 'bbck/bbb': ['c', 'b', 'f'], 'bbck/bbbb': ['c', 'f'], 'bbck/k': ['k', 'b'], 'bbck/kb': ['f', 'c', 'b'], 
  'bbck/kbb': ['f', 'c', 'b'], 'bbck/kbbb': ['f', 'c'], 'bbbc/': ['k', 'b'], 'bbbc/b': ['c', 'b', 'f'], 'bbbc/bb': ['c', 'b', 'f'], 
  'bbbc/bbb': ['c', 'b', 'f'], 'bbbc/bbbb': ['c', 'f'], 'bbbc/k': ['k', 'b'], 'bbbc/kb': ['f', 'c', 'b'], 'bbbc/kbb': ['f', 'c', 'b'], 
  'bbbc/kbbb': ['f', 'c'], 'bbbbck/': ['k', 'b'], 'bbbbck/b': ['c', 'b', 'f'], 'bbbbck/bb': ['c', 'b', 'f'], 'bbbbck/bbb': ['c', 'b', 'f'], 
  'bbbbck/bbbb': ['c', 'f'], 'bbbbck/k': ['k', 'b'], 'bbbbck/kb': ['f', 'c', 'b'], 'bbbbck/kbb': ['f', 'c', 'b'], 'bbbbck/kbbb': ['f', 'c']
} # ATTN::convert to TERMINAL: infosets which yield only **1** legal action

INFOSET_LEGAL_ACTIONS_3b = {
  '': ['b', 'f'], 'b': ['c', 'b', 'f'], 'bb': ['c', 'b', 'f'], 'bbb': ['c', 'f'], 'bc/': ['k', 'b'], 
  'bc/b': ['c', 'b', 'f'], 'bc/bb': ['c', 'b', 'f'], 'bc/bbb': ['c', 'f'], 'bc/k': ['k', 'b'], 
  'bc/kb': ['f', 'c', 'b'], 'bc/kbb': ['f', 'c', 'b'], 'bc/kbbb': ['f', 'c'], 'bbck/': ['k', 'b'], 'bbck/b': ['c', 'b', 'f'], 
  'bbck/bb': ['c', 'b', 'f'], 'bbck/bbb': ['c', 'f'], 'bbck/k': ['k', 'b'], 'bbck/kb': ['f', 'c', 'b'], 
  'bbck/kbb': ['f', 'c', 'b'], 'bbck/kbbb': ['f', 'c'], 'bbbc/': ['k', 'b'], 'bbbc/b': ['c', 'b', 'f'], 'bbbc/bb': ['c', 'b', 'f'],
  'bbbc/bbb': ['c', 'f'], 'bbbc/k': ['k', 'b'], 'bbbc/kb': ['f', 'c', 'b'], 'bbbc/kbb': ['f', 'c', 'b'],
  'bbbc/kbbb': ['f', 'c']
}

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

# PREFLOP_BUCKETS = {w: (PREFLOP_WR[w]-30)//5 for w in PREFLOP_WR}
PREFLOP_BUCKETS = { # Set 'AA' to 10 (instead of 11) so that 11 buckets --> 10
  '22': 4, '33': 5, '44': 5, '55': 6, '66': 6, '77': 7, '88': 7, '99': 8, 'TT': 9, 'JJ': 9, 'QQ': 10, 'KK': 10, 'AA': 10, '23': 1, '24': 1, '25': 1, '26': 1, '27': 1, '28': 2, '29': 2, '2T': 2, '2J': 3, '2Q': 3, '2K': 4, '2A': 5, '34': 1, '35': 1, '36': 1, '37': 1, '38': 2, '39': 2, '3T': 3, '3J': 3, '3Q': 4, '3K': 4, '3A': 5, '45': 2, '46': 2, '47': 2, '48': 2, '49': 2, '4T': 3, '4J': 3, '4Q': 4, '4K': 4, '4A': 5, '56': 2, '57': 2, '58': 2, '59': 3, '5T': 3, '5J': 3, '5Q': 4, '5K': 5, '5A': 6, '67': 3, '68': 3, '69': 3, '6T': 3, '6J': 4, '6Q': 4, '6K': 5, '6A': 5, '78': 3, '79': 3, '7T': 4, '7J': 4, '7Q': 4, '7K': 5, '7A': 6, '89': 4, '8T': 4, '8J': 4, '8Q': 5, '8K': 5, '8A': 6, '9T': 4, '9J': 5, '9Q': 5, '9K': 5, '9A': 6, 'TJ': 5, 'TQ': 5, 'TK': 6, 'TA': 6, 'JQ': 5, 'JK': 6, 'JA': 7, 'QK': 6, 'QA': 7, 'KA': 7, '23s': 1, '24s': 2, '25s': 2, '26s': 2, '27s': 2, '28s': 2, '29s': 3, '2Ts': 3, '2Js': 4, '2Qs': 4, '2Ks': 5, '2As': 5, '34s': 2, '35s': 2, '36s': 2, '37s': 2, '38s': 2, '39s': 3, '3Ts': 3, '3Js': 4, '3Qs': 4, '3Ks': 5, '3As': 6, '45s': 2, '46s': 2, '47s': 3, '48s': 3, '49s': 3, '4Ts': 3, '4Js': 4, '4Qs': 4, '4Ks': 5, '4As': 6, '56s': 3, '57s': 3, '58s': 3, '59s': 3, '5Ts': 3, '5Js': 4, '5Qs': 5, '5Ks': 5, '5As': 6, '67s': 3, '68s': 3, '69s': 4, '6Ts': 4, '6Js': 4, '6Qs': 5, '6Ks': 5, '6As': 6, '78s': 4, '79s': 4, '7Ts': 4, '7Js': 4, '7Qs': 5, '7Ks': 5, '7As': 6, '89s': 4, '8Ts': 4, '8Js': 5, '8Qs': 5, '8Ks': 6, '8As': 6, '9Ts': 5, '9Js': 5, '9Qs': 5, '9Ks': 6, '9As': 6, 'TJs': 5, 'TQs': 6, 'TKs': 6, 'TAs': 7, 'JQs': 6, 'JKs': 6, 'JAs': 7, 'QKs': 6, 'QAs': 7, 'KAs': 7
}

# PREFLOP_BUCKET_CARDS = {}
# for _ in PREFLOP_BUCKETS: 
#    PREFLOP_BUCKET_CARDS.setdefault(PREFLOP_BUCKETS[_],[]).append(_)

PREFLOP_BUCKET_PROBS = {"A": 0.08446455505279035, "B": 0.15384615384615385, "C": 0.18099547511312217, "D": 0.1885369532428356, "E": 0.19306184012066366, 
                        "F": 0.12368024132730016, "G": 0.048265460030165915, "H": 0.004524886877828055, "I":  0.00904977375565611, "J": 0.013574660633484163}
FLOP_BUCKET_PROBS = {"A": 0.0134, "B": 0.1158, "C": 0.2188, "D": 0.3700, "E": 0.1234, 
                    "F": 0.0840, "G": 0.0488, "H": 0.0242, "I": 0.0015}
BUCKET25_PROBS = {
   'AA': 0.33719934402332363, 'AB': 0.2569442419825073, 'AC': 0.05844752186588921, 'AD': 0.03264030612244898, 'AE': 0.21580539358600584, 
   'AF': 0.05802113702623907, 'AG': 0.00551567055393586, 'AH': 0.020847303206997084, 'AI': 0.01457908163265306, 
   'BA': 0.0879891956782713, 'BB': 0.41976990796318525, 'BC': 0.10606442577030813, 'BD': 0.05745498199279712, 'BE': 0.14904261704681873, 
   'BF': 0.12170168067226891, 'BG': 0.02101440576230492, 'BH': 0.021278511404561825, 'BI': 0.015684273709483793, 
   'CA': 0.00548469387755102, 'CB': 0.26926445578231295, 'CC': 0.29563690476190474, 'CD': 0.07862925170068027, 'CE': 0.10094387755102041, 
   'CF': 0.14133163265306123, 'CG': 0.07071428571428572, 'CH': 0.021181972789115645, 'CI': 0.01681292517006803, 
   'DA': 0.00012979591836734695, 'DB': 0.05738204081632653, 'DC': 0.3583567346938776, 'DD': 0.20609061224489797, 'DE': 0.09154040816326531, 
   'DF': 0.10433142857142857, 'DG': 0.13237224489795918, 'DH': 0.02908734693877551, 'DI': 0.02070938775510204, 
   'EA': 0.000001, 'EB': 0.005493463010204082, 'EC': 0.1541143176020408, 'ED': 0.36061782525510205, 'EE': 0.15639508928571427, 
   'EF': 0.0883952487244898, 'EG': 0.12627311862244897, 'EH': 0.0829639668367347, 'EI': 0.025746970663265305, 
   'FA': 0.000001, 'FB': 0.0001679940268790443, 'FC': 0.03045918367346939, 'FD': 0.32020408163265307, 'FE': 0.25925584868093576, 
   'FF': 0.11713290194126431, 'FG': 0.09623195619711299, 'FH': 0.14322424091587854, 'FI': 0.033323792931806866, 
   'GA': 0.000001, 'GB': 0.000001, 'GC': 0.0031377551020408162, 'GD': 0.13598533163265306, 'GE': 0.3301434948979592, 
   'GF': 0.19708545918367346, 'GG': 0.09248405612244898, 'GH': 0.19310586734693877, 'GI': 0.04805803571428571, 
   'HA': 0.000001, 'HB': 0.000001, 'HC': 0.000001, 'HD': 0.0009183673469387755, 'HE': 0.03857142857142857, 
   'HF': 0.346734693877551, 'HG': 0.49785714285714283, 'HH': 0.02469387755102041, 'HI': 0.09122448979591836, 
   'IA': 0.000001, 'IB': 0.000001, 'IC': 0.000001, 'ID': 0.000001, 'IE': 0.007959183673469388, 
   'IF': 0.13066326530612246, 'IG': 0.6404591836734694, 'IH': 0.12887755102040815, 'II': 0.0920408163265306, 
   'JA': 0.000001, 'JB': 0.000001, 'JC': 0.000001, 'JD': 0.000001, 'JE': 0.0007482993197278912, 
   'JF': 0.021496598639455782, 'JG': 0.27568027210884355, 'JH': 0.5941496598639455, 'JI': 0.10792517006802721
}

s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
num2letter = {}
for n in range(1, 53):
    num2letter[n] = s[n-1]

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
        self.beliefs: dict[str, float] = {}
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
                and len(INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]) > 1:
                infoSet = infoSets[infoSetStr]
                row=[infoSetStr,*infoSet.getStrategyTableData(),
                     infoSetStr,f'{infoSet.expectedUtil:.2f}',f'{infoSet.likelihood*100:.2f}%',
                     infoSetStr,*infoSet.getGainTableData()]
          else:
            if infoSetStr[0] == client_hand_rank_ and (len(infoSetStr) + 1) % 2 == client_pos_\
                and len(INFOSET_LEGAL_ACTIONS[infoSetStr[2:]]) > 1:
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
        self.cumulativeStrategy = 0.0


@lru_cache(maxsize=None)
def getAncestralInfoSetStrs(infoSetStr) -> list[str]:
    # given an infoSet, return all opponent infoSets that can lead to it (e.g. given 'Bpb', return ['Ap','Bp','Cp',...])
    if len(infoSetStr) == 1:
        raise ValueError(f'no ancestors of infoSet={infoSetStr}')
    
    if '/' in infoSetStr:  # flop
        if len(infoSetStr) < 3 or infoSetStr[0] not in RANKS_PREFLOP or infoSetStr[1] not in RANKS_FLOP:
            print(f'Error! getAncestralInfoSetStrs()::invalid infoSetStr: {infoSetStr}')
            sys.exit(-1)
        actionStr = infoSetStr[2:]
        suffix = ''

        if actionStr[-1] == '/': 
            suffix = actionStr[:-3] if actionStr[-2] == 'k' else actionStr[:-2]
            return [oppPocket1 + suffix for oppPocket1 in RANKS_PREFLOP]
        else:
            suffix = infoSetStr[2:-1]

        # Precompute concatenated strings
        return [oppPocket1 + oppPocket2 + suffix for oppPocket1 in RANKS_PREFLOP for oppPocket2 in RANKS_FLOP]
    else:  # preflop
        suffix = infoSetStr[1:-1]
        return [oppPocket + suffix for oppPocket in RANKS_PREFLOP]
  

def getDescendantInfoSetStrs(infoSetStr, action):
  # given an infoSet and an action to perform at that infoSet, return all opponent infoSets that can result from it 
  # e.g. given infoSetStr='Bpb' and action='p', return ['Apbp','Bpbp','Cpbp',...]
  if '/' in infoSetStr:  # flop
    if len(infoSetStr) < 2 or infoSetStr[0] not in RANKS_PREFLOP or infoSetStr[1] not in RANKS_FLOP:
      print('Error! getAncestralInfoSetStrs()::invalid infoSetStr: {}'.format(infoSetStr))
      sys.exit(-1)
    actionStr = infoSetStr[2:]+action

    return [oppPocket1+oppPocket2+actionStr for oppPocket1 in RANKS_PREFLOP for oppPocket2 in RANKS_FLOP]
  else:  # preflop
    actionStr = infoSetStr[1:]+action
    if actionStr in TERMINAL_ACTION_STRS_PREFLOP:
      if actionStr == 'bc' or actionStr == 'bbbc':
        return [oppPocket + oppPocket2 +actionStr + '/' for oppPocket in RANKS_PREFLOP for oppPocket2 in RANKS_FLOP]
      else:
        return [oppPocket + oppPocket2 +actionStr + 'k/' for oppPocket in RANKS_PREFLOP for oppPocket2 in RANKS_FLOP]
    return [oppPocket+actionStr for oppPocket in RANKS_PREFLOP]


def calcUtilityAtTerminalNode(pocket1, pocket2, action1, playerIdx_, totalBets, playerIdx2return):
  if action1 == 'f':
    return -totalBets if playerIdx2return == playerIdx_ else totalBets
  else:  # showdown
    if RANK2NUM[pocket1] > RANK2NUM[pocket2]:
      return totalBets if playerIdx2return == 0 else -totalBets
    elif RANK2NUM[pocket1] == RANK2NUM[pocket2]: # TODO: better tie breaker?
      return 0
      # return totalBets if randint(0, 100) < 50 else -totalBets
    else:
      return -totalBets if playerIdx2return == 0 else totalBets


def initInfoSets():
  # initialize the infoSet objects.
  for actionsStrs in sorted(INFOSET_ACTION_STRS, key=lambda x:len(x)):
    if '/' in actionsStrs: # flop
      for rank1 in RANKS_PREFLOP:
        for rank2 in RANKS_FLOP:
            infoSetStr = rank1 + rank2 + actionsStrs
            infoSets[infoSetStr] = InfoSetData()
            sortedInfoSets.append(infoSetStr)
    else: # preflop
      for rank in RANKS_PREFLOP:
        infoSetStr = rank + actionsStrs
        infoSets[infoSetStr] = InfoSetData()
        sortedInfoSets.append(infoSetStr)


def updateBeliefs():
    for infoSetStr in sortedInfoSets:
        infoSet = infoSets[infoSetStr]
        for opp in RANKS_PREFLOP:
            infoSet.beliefs[opp] = 0.0
        if len(infoSetStr) == 1:
            for oppPocket in RANKS_PREFLOP:
              infoSet.beliefs[oppPocket] = PREFLOP_BUCKET_PROBS[oppPocket] # natural prob of occuring: pre-computed lookup table
        else:
            if infoSetStr[-1] != '/':
              ancestralInfoSetStrs = getAncestralInfoSetStrs(infoSetStr) 
              lastAction = infoSetStr[-1]
              tot = 0  # normalizing factor for strategy (last action)
              for oppInfoSetStr in ancestralInfoSetStrs:
                  oppInfoSet=infoSets[oppInfoSetStr]
                  # try:
                  #    oppInfoSet=infoSets[oppInfoSetStr]
                  # except KeyError:
                  #    print('infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))

                  tot += oppInfoSet.actions[lastAction].strategy * PREFLOP_BUCKET_PROBS[oppInfoSetStr[0]]
              temdict = defaultdict(float)
              if tot == 0:
                for opp in RANKS_PREFLOP:
                  infoSet.beliefs[opp] = 1 / len(RANKS_PREFLOP)
              else:
                if '/' not in infoSetStr:
                  for oppInfoSetStr in ancestralInfoSetStrs:
                      oppInfoSet=infoSets[oppInfoSetStr]
                      oppPocket = oppInfoSetStr[0]  # TODO: include both buckets?
                      infoSet.beliefs[oppPocket]=(oppInfoSet.actions[lastAction].strategy * PREFLOP_BUCKET_PROBS[oppPocket] / tot)
                else:
                  for oppInfoSetStr in ancestralInfoSetStrs:
                      oppInfoSet=infoSets[oppInfoSetStr]
                      oppPocket = oppInfoSetStr[1] # TODO: include both buckets?
                      temdict[oppPocket]+=(oppInfoSet.actions[lastAction].strategy * PREFLOP_BUCKET_PROBS[oppPocket] / tot)
                  for oppPocket2 in RANKS_FLOP:
                    infoSet.beliefs[oppPocket2] = temdict[oppPocket2]
            else:
              ancestralInfoSetStr = infoSetStr[0] + infoSetStr[2:-2]
              ancestralInfoSetStrs = getAncestralInfoSetStrs(infoSetStr) 
              lastAction = ancestralInfoSetStr[-1]
              tot = 0  # normalizing factor for strategy (last action)
              tembeliefs = defaultdict(float)
              for oppInfoSetStr in ancestralInfoSetStrs:
                  oppInfoSet=infoSets[oppInfoSetStr]
                  # try:
                  #    oppInfoSet=infoSets[oppInfoSetStr]
                  # except KeyError:
                  #    print('infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))

                  tot += oppInfoSet.actions[lastAction].strategy * PREFLOP_BUCKET_PROBS[oppInfoSetStr[0]]
              if tot == 0:
                for opp in RANKS_FLOP:
                  infoSet.beliefs[opp] = 1 / len(RANKS_FLOP)
              else:
                for oppInfoSetStr in ancestralInfoSetStrs:
                    oppInfoSet=infoSets[oppInfoSetStr]
                    oppPocket = oppInfoSetStr[1] if '/' in oppInfoSetStr else oppInfoSetStr[0]  # TODO: include both buckets?
                    tembeliefs[oppPocket]=oppInfoSet.actions[lastAction].strategy * PREFLOP_BUCKET_PROBS[oppPocket] / tot
                for oppPocket2 in RANKS_FLOP:
                  totbeliefs = 0
                  for oppPocket1 in RANKS_PREFLOP:
                    oppPocket = oppPocket1 + oppPocket2
                    totbeliefs += tembeliefs[oppPocket1] * BUCKET25_PROBS[oppPocket]
                  infoSet.beliefs[oppPocket2] += totbeliefs
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
    for action in INFOSET_LEGAL_ACTIONS[cur_actionstr]:
        utilFromInfoSets,utilFromTerminalNodes=0,0
        actionStr=(infoSetStr[1:]+action) if street == 0 else (infoSetStr[2:]+action)
        if actionStr in TERMINAL_ACTION_STRS_PREFLOP:
          if actionStr == 'bbbc' or actionStr == 'bc':
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
            
            if actionStr in TERMINAL_ACTION_STRS:
                # choosing this action moves us to a terminal node
                utilFromTerminalNodes+=probOfThisInfoSet*calcUtilityAtTerminalNode(*pockets, actionStr[-1], playerIdx, TERMINAL_CHIPCOUNT[actionStr], playerIdx)
            else:
                # choosing this action moves us to an opponent infoSet where they will choose an action 
                # The opponent's strategy is the same as OURS bc this is self-play
                descendentInfoSet = infoSets[descendentInfoSetStr]
                if actionStr not in INFOSET_LEGAL_ACTIONS:
                  print('Error! updateUtilitiesForInfoSetStr()::invalid actionStr: {}'.format(actionStr))
                  print('infoSetStr: {} | action: {} | descendentInfoSetStr: {}'.format(infoSetStr, action, descendentInfoSetStr))
                  sys.exit(-1)
                for oppAction in INFOSET_LEGAL_ACTIONS[actionStr]:
                    probOfOppAction = descendentInfoSet.actions[oppAction].strategy
                    destinationInfoSetStr = infoSetStr[0] + actionStr + oppAction if street == 0 else infoSetStr[0:2] + actionStr + oppAction
                    destinationActionStr = destinationInfoSetStr[2:] if street == 1 else destinationInfoSetStr[1:]
                    if destinationActionStr in TERMINAL_ACTION_STRS:
                        # our opponent choosing that action moves us to a terminal node
                        utilFromTerminalNodes+=probOfThisInfoSet*probOfOppAction*\
                          calcUtilityAtTerminalNode(*pockets,destinationActionStr[-1], (playerIdx+1)%2, TERMINAL_CHIPCOUNT[destinationActionStr], playerIdx)
                    else:
                        # it's another infoSet, and we've already calculated the expectedUtility of this infoSet
                        # ^^ the utility must've been computed as we are traversing the game tree from bottom up
                        
                        if destinationInfoSetStr[1:] in TERMINAL_ACTION_STRS_PREFLOP:
                          if destinationInfoSetStr[1:] == 'bc' or destinationInfoSetStr[1:] == 'bbbc':
                            destinationInfoSetStr += '/'
                            for oppPocket in RANKS_FLOP:
                              tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                              utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET25_PROBS[infoSetStr[0]+oppPocket]
                          else:
                            destinationInfoSetStr += 'k/'
                            for oppPocket in RANKS_FLOP:
                              tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                              utilFromInfoSets-=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET25_PROBS[infoSetStr[0]+oppPocket]
                        if '/' in destinationInfoSetStr and street == 0:
                          for oppPocket in RANKS_FLOP:
                            tem = infoSetStr[0] + oppPocket + destinationInfoSetStr[1:]
                            utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[tem].expectedUtil*BUCKET25_PROBS[infoSetStr[0]+oppPocket]
                        
                        else:
                          utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[destinationInfoSetStr].expectedUtil
        infoSet.actions[action].util=utilFromInfoSets+utilFromTerminalNodes
    
    infoSet.expectedUtil = 0 # Start from nothing, neglecting illegal actions
    for action in INFOSET_LEGAL_ACTIONS[cur_actionstr]:
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
        infoSet.likelihood=PREFLOP_BUCKET_PROBS[infoSetStr[0]]
      elif len(infoSetStr)==2:  # P2's perspective
        # the second-tier infoSet likelihoods. Note, the second-tier infoSet, e.g., 'Bb', may have resulted from the top-tier infoSets 'A', 'B',...
        # depending on which hand tier player 1 has. The likelihood of 'Bb' is therefore the multiplication of the likelihood along each of these possible paths
        for oppPocket in RANKS_PREFLOP:
          oppInfoSet = infoSets[oppPocket]
          infoSet.likelihood+=oppInfoSet.actions[infoSetStr[-1]].strategy*PREFLOP_BUCKET_PROBS[infoSetStr[0]]*PREFLOP_BUCKET_PROBS[oppPocket]  # once again this is natural prob
      else:
        # For infoSets on the third-tier and beyond, we can use the likelihoods of the infoSets two levels before to calculate their likelihoods.
        # Note, we can't simply use the infoSet one tier before because that's the opponent's infoSet, and the calculation of likelihoods 
        # assumes that the infoSet's "owner" is trying to reach the infoSet. Therefore, when calculating a liklihood for player 1's infoSet, 
        # we can only use the likelihood of an ancestral infoSet if the ancestral infoSet is also "owned" by player 1, and the closest such infoSet is 2 levels above.
        # Note also, that although there can be multiple ancestral infoSets one tier before, there is only one ancestral infoSet two tiers before. 
        # For example, 'Bbc' has one-tier ancestors 'Ab' and 'Bb', but only a single two-tier ancestor: 'B'

        infoSetTwoLevelsAgo = infoSets[infoSetStr[:-2]] # grab the closest ancestral infoSet with the same owner as the infoSet for which we seek to calculate likelihood
        for oppPocket in RANKS_PREFLOP:
          oppInfoSet = infoSets[oppPocket + infoSetStr[1:-1]]
          infoSet.likelihood+=infoSetTwoLevelsAgo.likelihood*PREFLOP_BUCKET_PROBS[oppPocket]*oppInfoSet.actions[infoSetStr[-1]].strategy *\
            infoSetTwoLevelsAgo.actions[infoSetStr[-2]].strategy
          # ^^ note, each oppInfoSet is essentially slicing up the infoSetTwoLevelsAgo because they're each assuming a specific oppPocket. 
          # ^^ Therefore, we must account for the prob. of each opponent pocket
    else:
      if infoSetStr[-1] == '/':
        # at beginning of flop, the likelihood is determined solely by precomputed transitional probs times its ancestral infoset.
        actionStr = infoSetStr[2:]
        if actionStr[-2] == 'k':
          temstr = oppPocket1 + infoSetStr[2:-3]
          infoSet.likelihood += infoSets[temstr].likelihood * infoSets[temstr].actions[actionStr[-3]].strategy
          infoSet.likelihood *= BUCKET25_PROBS[infoSetStr[:2]]
        elif actionStr[:-1] in TERMINAL_ACTION_STRS_PREFLOP:
          temstr = infoSetStr[0] + infoSetStr[2:-3]
          for oppPocket1 in RANKS_FLOP:
            temstr2 = oppPocket1 + actionStr[:-2]
            infoSet.likelihood += infoSets[temstr].likelihood * infoSets[temstr].actions[actionStr[-3]].strategy *\
              infoSets[temstr2].actions[actionStr[-2]].strategy * FLOP_BUCKET_PROBS[oppPocket1]
          infoSet.likelihood *= BUCKET25_PROBS[infoSetStr[:2]]
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
          infoSet.likelihood = actionprob * BUCKET25_PROBS[infoSetStr[:2]] * PREFLOP_BUCKET_PROBS[infoSetStr[0]]
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

          for oppPocket1 in RANKS_PREFLOP:
            for oppPocket2 in RANKS_FLOP:
              rank_tem = oppPocket1 + oppPocket2
              if BUCKET25_PROBS[rank_tem] > 0:
                oppInfoSet = infoSets[rank_tem + actionStr]
                infoSet.likelihood += infoSetTwoLevelsAgoFlop.likelihood * FLOP_BUCKET_PROBS[oppPocket2] * PREFLOP_BUCKET_PROBS[oppPocket1] *\
                  oppInfoSet.actions[flopActionStr[-1]].strategy


# def calcGains(cur_t, alpha = 0.5, beta = 1.0):
#   # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
#   # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
#   maxgain = 0.0
#   totAddedGain=0.0
#   for infoSetStr in sortedInfoSets:
#     infoSet = infoSets[infoSetStr]
#     for action in (INFOSET_LEGAL_ACTIONS[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]):
#       utilForActionPureStrat = infoSet.actions[action].util 
#       gain = max(0,utilForActionPureStrat-infoSet.expectedUtil)
#       totAddedGain+=gain
#       infoSet.actions[action].cumulativeGain+=gain * infoSet.likelihood
#       maxgain = max(gain, maxgain)
#   print(maxgain)
#   return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)

def calcGains(cur_t, alpha = 0.4, beta = 1.2):
  # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
  # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
  # we note that in first several iterations the gains are very large, and thus in later iterations its hard to change the strategy since denominator is large
  # so we use alpha to scale the gains, and thus make the strategy converge faster and more to 0
  totAddedGain=0.0
  max_now = 0.0
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    for action in INFOSET_LEGAL_ACTIONS[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]:
      utilForActionPureStrat = infoSet.actions[action].util 
      gain = max(0, utilForActionPureStrat-infoSet.expectedUtil)

      totAddedGain += gain
      max_now = max(gain, max_now)
      if gain > 0:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, alpha) / (math.pow(cur_t, alpha) + 1)) + gain
      else:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, beta) / (math.pow(cur_t, beta) + 1)) + gain
  print(max_now)
  return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)


def updateStrategy(cur_t, gamma = 100.0):
  # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    allLegalActions = INFOSET_LEGAL_ACTIONS[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]
    totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
    likelihood = infoSet.likelihood
    totStrategy = 0.0
    for action in ACTIONS:
        cur_strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0
        # infoSet.actions[action].cumulativeStrategy = infoSet.actions[action].cumulativeStrategy * math.pow((cur_t - 1) / cur_t, gamma) + cur_strategy * likelihood
        infoSet.actions[action].cumulativeStrategy = cur_strategy * likelihood
        totStrategy += infoSet.actions[action].cumulativeStrategy
    for action in ACTIONS:
        infoSet.actions[action].strategy = infoSet.actions[action].cumulativeStrategy / totStrategy if action in allLegalActions else 0.0 #why using cumulativeGain instead of gain, aka counterfactual regret?

# def updateStrategy(cur_t, gamma = 6.0):
#   # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
#   for infoSetStr in sortedInfoSets:
#     infoSet = infoSets[infoSetStr]
#     allLegalActions = INFOSET_LEGAL_ACTIONS[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]

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
#     if all0:
#       print('Error! updateStrategy()::all0 | infoSetStr: {}'.format(infoSetStr))
#       for action in allLegalActions:
#         print(infoSet.actions[action].strategy)
#         print(infoSet.actions[action].cumulativeStrategy)
#       sys.exit(-1)       


def calcHoleCardsRankNum(hole_cards_):
  if len(hole_cards_) != 2:
    print("ERROR! Illegal number of hole cards")
    sys.exit(-1)
  hole_card_ranks = [card2rank[hc] for hc in hole_cards_]
  hole_card_ranks.sort()
  hole_cards_str = ''
  for hcr in hole_card_ranks:
    hole_cards_str += rank2str[hcr]
  if card2row[hole_cards_[0]] == card2row[hole_cards_[1]]:
    hole_cards_str += 's' # mark suited
  return PREFLOP_BUCKETS[hole_cards_str]


if __name__ == "__main__":
    start = time.time()
    client_pos = 0
    hole_cards = [40, 7]  #7hAs
    # hole_cards = [45, 24] #6sJd
    # hole_cards = [16, 34] #3d8c
    # hole_cards = [42, 17] #3s4d
    street = 0 # 0: preflop; 1: flop; 2: turn; 3: river
    community_cards_all = [20, 9, 44, 13, 45]
    community_cards = community_cards_all[:street+2] if street > 0 else []
    client_hand_rank_num = calcHoleCardsRankNum(hole_cards)
    client_hand_rank = NUM2RANK[client_hand_rank_num]

    if len(community_cards):
      print('Community cards: ', end='')
      show_cards(community_cards)
    print('Client hole cards: ', end='')
    show_cards(hole_cards)
    print('Estimated client hand WR: {}-{}% | position: {}\n'.format(
       30+5*client_hand_rank_num, 34+5*client_hand_rank_num,
       'FIRST' if client_pos == 0 else 'SECOND'))

    # profiler = cProfile.Profile()
    # profiler.enable()

    # calcTransitionProbs(6,1)
    

    # profiler.disable()
    # profiler.dump_stats("tProbs.prof")

    initInfoSets()
    # print('>>>SORTED INFO SETS>>>\n{}\n<<<'.format(sortedInfoSets))

    numIterations=1000  # 10k converges a lot better; 1k ~ 40s
    totGains = []
    iterations = []
    time_belief = 0
    time_util = 0
    time_likelihood = 0
    time_gain = 0
    time_strategy = 0
    descendtot = 0

    # only plot the gain from every xth iteration (in order to lessen the amount of data that needs to be plotted)
    numGainsToPlot=200
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
        totGain = calcGains(i + 1)
        time_gain += time.time() - gain_start
        
        if i%gainGrpSize==0: # every 10 or 100 or x rounds, save off the gain so we can plot it afterwards and visually see convergence
            totGains.append(totGain)
            iterations.append(i)
            # print(f'TOT_GAIN {totGain: .3f}  @{i}/{numIterations}')
            
        strategy_start = time.time()
        updateStrategy(i + 1)
        time_strategy += time.time() - strategy_start

    # profiler.disable()
    # profiler.dump_stats("good_cfr_texas.prof")

    InfoSetData.printInfoSetDataTable(infoSets, client_hand_rank, client_pos)
    print('\ntotGains:', totGains)
    
    with open('infoSets_PREFLOP_FLOP.pkl', 'wb') as f:
        pickle.dump(infoSets, f)
    
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
    plt.savefig('cfr_preflop_flop_de.png')
    plt.show()
