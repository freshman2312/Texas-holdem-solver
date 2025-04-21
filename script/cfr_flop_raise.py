# solve flop street 
# 

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


RANKS_TURN = ["B", "C", "D", "E", "F", "G", "H", "I", "J"]
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I", 10: "J"}
ACTIONS = ["k", "c", "b", "f", "r"]  # {k: check, c: call, b: bet/raise/all-in, f: fold}

TERMINAL_ACTION_STRS_TURN = {
    'bc', 'bbc', 'bbbc', 'bbbbc', 'f', 'bf', 'bbf', 'bbbf', 'bbbbf', 'kbc', 'kbbc', 'kbbbc', 'kbbbbc', 'kbf', 'kbbf', 'kbbbf', 'kbbbbf', 'kk',
    'rc', 'rrc', 'rrrc', 'rrrrc', 'f', 'rf', 'rrf', 'rrrf', 'rrrrf', 'krc', 'krrc', 'krrrc', 'krrrrc', 'krf', 'krrf', 'krrrf', 'krrrrf', 'kk',
    'rbc', 'brc', 'rbbc', 'brbc', 'bbrc', 'rrbc', 'rbrc', 'brrc', 'rbbbc', 'brbbc', 'bbrbc', 'bbbrc', 'rrbbc', 'rbrbc', 'brbrc', 'rbbrc', 'brrbc', 'bbrrc', 'rrrbc',
    'rrbrc', 'rbrrc', 'brrrc', 'rbf', 'brf', 'rbbf', 'brbf', 'bbrf', 'rrbf', 'rbrf', 'brrf', 'bbrrf', 'rbbbf', 'brbbf', 'bbrbf', 'bbbrf', 'rrbbf', 'rbrbf', 'brbrf',
    'rbbrf', 'brrbf', 'rrrbf', 'rrbrf', 'rbrrf', 'brrrf', 'krbc', 'kbrc', 'krbbc', 'kbrbc', 'kbbrc', 'krrbc', 'krbrc', 'kbrrc', 'krbbbc', 'kbrbbc', 'kbbrbc', 
    'kbbbrc', 'krrbbc', 'krbrbc', 'kbrbrc', 'krbbrc', 'kbrrbc', 'kbbrrc', 'krrrbc', 'krrbrc', 'krbrrc', 'kbrrrc', 'krbf', 'kbrf', 'krbbf', 'kbrbf', 'kbbrf', 'krrbf', 'krbrf', 'kbrrf', 
    'krbbbf', 'kbrbbf', 'kbbrbf', 'kbbbrf', 'krrbbf', 'krbrbf', 'kbrbrf', 'krbbrf', 'kbrrbf', 'kbbrrf', 'krrrbf', 'krrbrf', 'krbrrf', 'kbrrrf'}

TERMINAL_CHIPCOUNT_TURN = {
    # Original terminal states with 'b' (doubles) and 'k' (check)
    'bc': 2, 'bbc': 4, 'bbbc': 8, 'bbbbc': 16, 
    'f': 1, 'bf': 1, 'bbf': 2, 'bbbf': 4, 'bbbbf': 8, 
    'kbc': 2, 'kbbc': 4, 'kbbbc': 8, 'kbbbbc': 16, 
    'kbf': 1, 'kbbf': 2, 'kbbbf': 4, 'kbbbbf': 8, 'kk': 1,
    
    # Terminal states with 'r' (triples)
    'rc': 3, 'rrc': 9, 'rrrc': 27, 'rrrrc': 81,
    'rf': 1, 'rrf': 3, 'rrrf': 9, 'rrrrf': 27,
    'krc': 3, 'krrc': 9, 'krrrc': 27, 'krrrrc': 81,
    'krf': 1, 'krrf': 3, 'krrrf': 9, 'krrrrf': 27,
    
    # Mixed terminal states with both 'r' and 'b'
    'rbc': 6, 'brc': 6, 'rbbc': 12, 'brbc': 12, 'bbrc': 12, 
    'rrbc': 18, 'rbrc': 18, 'brrc': 18, 
    'rbbbc': 24, 'brbbc': 24, 'bbrbc': 24, 'bbbrc': 24, 
    'rrbbc': 36, 'rbrbc': 36, 'rbbrc': 36, 'brrbc': 36, 'brbrc':36, 'bbrrc': 36,
    'rrrbc': 54, 'rrbrc': 54, 'rbrrc': 54, 'brrrc': 54,
    
    # Fold variants of mixed terminal states
    'rbf': 3, 'brf': 2, 'rbbf': 6, 'brbf': 6, 'bbrf': 4, 
    'rrbf': 9, 'rbrf': 6, 'brrf': 6, 
    'rbbbf': 12, 'brbbf': 12, 'bbrbf': 12, 'bbbrf': 8, 
    'rrbbf': 18, 'rbrbf': 18, 'rbbrf': 12, 'brrbf': 18, 'brbrf': 12, 'bbrrf': 12,
    'rrrbf': 27, 'rrbrf': 18, 'rbrrf': 18, 'brrrf': 18,
    
    # With 'k' prefix (check first)
    'krbc': 6, 'kbrc': 6, 'krbbc': 12, 'kbrbc': 12, 'kbbrc': 12, 
    'krrbc': 18, 'krbrc': 18, 'kbrrc': 18, 
    'krbbbc': 24, 'kbrbbc': 24, 'kbbrbc': 24, 'kbbbrc': 24, 
    'krrbbc': 36, 'krbrbc': 36, 'krbbrc': 36, 'kbrrbc': 36, 'kbrbrc': 36, 'kbbrrc': 36,
    'krrrbc': 54, 'krrbrc': 54, 'krbrrc': 54, 'kbrrrc': 54,
    
    # Fold variants with 'k' prefix
    'krbf': 3, 'kbrf': 2, 'krbbf': 6, 'kbrbf': 6, 'kbbrf': 4, 
    'krrbf': 9, 'krbrf': 6, 'kbrrf': 6, 
    'krbbbf': 12, 'kbrbbf': 12, 'kbbrbf': 12, 'kbbbrf': 8, 
    'krrbbf': 18, 'krbrbf': 18, 'krbbrf': 12, 'kbrrbf': 18, 'kbrbrf': 12, 'kbbrrf': 12,
    'krrrbf': 27, 'krrbrf': 18, 'krbrrf': 18, 'kbrrrf': 18
}
INFOSET_ACTION_STRS_TURN = { # preflop + flop, total=5+36
  # Original strings with only 'b's
  '', 'b', 'bb', 'bbb', 'bbbb', 'k', 'kb', 'kbb', 'kbbb', 'kbbbb',
  
  # Add 'r' sequences
  'r', 'rr', 'rrr', 'rrrr',
  'kr', 'krr', 'krrr', 'krrrr',
  
  # Mixed b and r sequences (where sum ≤ 4)
  'br', 'rb', 'brr', 'rbr', 'rrb', 'brb', 'rbb', 'bbr', 
  'brrr', 'rbrr', 'rrbr', 'rrrb', 'brbr', 'brrb', 'rbrb', 'rrbb', 'rbbr', 'bbrr',
  'bbbr', 'bbrb', 'brbb', 'rbbb',
  
  # With 'k' prefix
  'kbr', 'krb', 'kbrr', 'krbr', 'krrb', 'kbrb', 'krbb', 'kbbr',
  'kbrrr', 'krbrr', 'krrbr', 'krrrb', 'kbrbr', 'kbrrb', 'krbrb', 'krrbb', 'krbbr', 'kbbrr',
  'kbbbr', 'kbbrb', 'kbrbb', 'krbbb'
} # action paths where a decision still needs to be made

INFOSET_LEGAL_ACTIONS_TURN = { 
  # Original actions
  '': ['b', 'f', 'k', 'r'], 'b': ['b', 'c', 'f', 'r'], 'bb': ['b', 'c', 'f', 'r'], 'bbb': ['b', 'c', 'f', 'r'], 'bbbb': ['c', 'f'], 
  'k': ['b', 'k', 'r'], 'kb': ['b', 'c', 'f', 'r'], 'kbb': ['b', 'c', 'f', 'r'], 'kbbb': ['b', 'c', 'f', 'r'], 'kbbbb': ['c', 'f'],
  
  # 'r' sequences
  'r': ['b', 'c', 'f', 'r'], 'rr': ['b', 'c', 'f', 'r'], 'rrr': ['b', 'c', 'f', 'r'], 'rrrr': ['c', 'f'], 'kr': ['b', 'c', 'f', 'r'],
  'krr': ['b', 'c', 'f', 'r'], 'krrr': ['b', 'c', 'f', 'r'], 'krrrr': ['c', 'f'],
  
  # Mixed b and r sequences (ensuring total actions ≤ 4)
  'br': ['b', 'c', 'f', 'r'], 'rb': ['b', 'c', 'f', 'r'], 'brr': ['b', 'c', 'f', 'r'], 'rbr': ['b', 'c', 'f', 'r'], 'rrb': ['b', 'c', 'f', 'r'],
  'brb': ['b', 'c', 'f', 'r'], 'rbb': ['b', 'c', 'f', 'r'], 'bbr': ['b', 'c', 'f', 'r'], 'brrr': ['c', 'f'], 'rbrr': ['c', 'f'], 'rrbr': ['c', 'f'], 'rrrb': ['c', 'f'], 
  'brbr': ['c', 'f'], 'brrb': ['c', 'f'], 'rbrb': ['c', 'f'], 'rbbr': ['c', 'f'], 'rrbb': ['c', 'f'], 'bbrr': ['c', 'f'], 'bbbr': ['c', 'f'], 
  'bbrb': ['c', 'f'], 'brbb': ['c', 'f'], 'rbbb': ['c', 'f'],
  
  # With 'k' prefix
  'kbr': ['b', 'c', 'f', 'r'], 'krb': ['b', 'c', 'f', 'r'], 'kbrr': ['b', 'c', 'f', 'r'], 'krbr': ['b', 'c', 'f', 'r'],
  'krrb': ['b', 'c', 'f', 'r'], 'kbbr': ['b', 'c', 'f', 'r'], 'kbrb': ['b', 'c', 'f', 'r'], 'krbb': ['b', 'c', 'f', 'r'], 'krbr': ['b', 'c', 'f', 'r'], 'kbrrr': ['c', 'f'], 
  'krbrr': ['c', 'f'], 'krrbr': ['c', 'f'], 'krrrb': ['c', 'f'], 'kbrbr': ['c', 'f'], 'kbrrb': ['c', 'f'], 'krbrb': ['c', 'f'],
  'krbbr': ['c', 'f'], 'krrbb': ['c', 'f'], 'kbbrr': ['c', 'f'], 'kbbbr': ['c', 'f'], 'kbbrb': ['c', 'f'], 'kbrbb': ['c', 'f'], 
  'krbbb': ['c', 'f']
}

NUM_ACTIONS = len(ACTIONS)


FLOP_BUCKET_PROBS = {"B": 0.0134, "C": 0.1158, "D": 0.2188, "E": 0.3700, "F": 0.1234, 
                    "G": 0.0840, "H": 0.0488, "I": 0.0242, "J": 0.0015}
# Natural prob of each bucket for turn round. that is occurence of each bucket / total number of hands
BUCKET56_PROBS = {
    'AA': 0.8480948471267484, 'AB': 0.0198374356142587, 'AC': 0.0018954500300452, 'AD': 0.0137805352690935, 'AE': 0.0733869823283358, 
    'AF': 0.0381839275017294, 'AG': 0.0045974147504137, 'AH': 0.0002037362893674, 'AI': 0.0000000050000000, 'AJ': 0.0000196710900079,
    'BA': 0.5409506783009161, 'BB': 0.2823608917580871, 'BC': 0.0163728254836688, 'BD': 0.0148323794031578, 'BE': 0.0288427267721925, 
    'BG': 0.0274130620957111, 'BH': 0.0148714494645764, 'BI': 0.0043238106337811, 'BJ': 0.0146636646559344, 'CA': 0.2111039102533692, 
    'CB': 0.3755744446422428, 'CC': 0.1858126498685885, 'CD': 0.0429359330861157, 'CE': 0.0234336153363996, 'CF': 0.0369033395087988, 
    'CG': 0.0340914217738775, 'CH': 0.0375812176648383, 'CI': 0.0178024253989985, 'CJ': 0.0347610424667711, 'DA': 0.0438021139842402, 
    'DB': 0.1558863205545636, 'DC': 0.3171353328969181, 'DD': 0.2203334184749630, 'DE': 0.0786478234568574, 'DF': 0.0342095193520985, 
    'DG': 0.0278758816247700, 'DH': 0.0398927710264859, 'DI': 0.0412402107827371, 'DJ': 0.0409766078463661, 'EA': 0.0054172864587393, 
    'EB': 0.0315142609284356, 'EC': 0.1217606571616269, 'ED': 0.3311994640351739, 'EE': 0.2446679171274790, 'EF': 0.0841961416050733, 
    'EG': 0.0338476460440790, 'EH': 0.0395193524199272, 'EI': 0.0505620779432019, 'EJ': 0.0573151962762639, 'FA': 0.0030832814840459,
    'FB': 0.0031192050457181, 'FC': 0.0157175692382392, 'FD': 0.0882097542278018, 'FE': 0.3335996672116175, 'FF': 0.2851789579952091,
    'FG': 0.1056155317269551, 'FH': 0.0486431678853232, 'FI': 0.0419990355652867, 'FJ': 0.0748338296198034, 'GA': 0.0017614330207370,
    'GB': 0.0018167151097690, 'GC': 0.0013709843641195, 'GD': 0.0109109797825371, 'GE': 0.0608231069007049, 'GF': 0.2784603755375059,
    'GG': 0.3373126237855998, 'GH': 0.1778870515832816, 'GI': 0.0433688029327229, 'GJ': 0.0862879269830223, 'HA': 0.0006811380696226,
    'HB': 0.0000489491951693, 'HC': 0.0000493785740743, 'HD': 0.0005533978453868, 'HE': 0.0045914351032444, 'HF': 0.0465313804461127,
    'HG': 0.2020513016608087, 'HH': 0.4348425994886354, 'HI': 0.2073619106483431, 'HJ': 0.1032885089686027, 'IA': 0.0007529466956181,
    'IB': 0.0000095230215328, 'IC': 0.0000218954510832, 'ID': 0.0001968528525895, 'IE': 0.0015897228501421, 'IF': 0.0055108675828167,
    'IG': 0.0354611171062892, 'IH': 0.1663113102858424, 'II': 0.5346767206945462, 'IJ': 0.2554690434595397, 'JA': 0.0011544720790074,
    'JB': 0.0000546340372169, 'JC': 0.0000207561626108, 'JD': 0.0000133901106498, 'JE': 0.0001352232183882, 'JF': 0.0015761412761846,
    'JG': 0.0031808371869954, 'JH': 0.0282554446816730, 'JI': 0.0794721159121937, 'JJ': 0.8861369853350802, 'BF': 0.0553685114319749
}
# conditional probability of each bucket given the previous bucket


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
            "r": InfoSetActionData(initStratVal=1 / NUM_ACTIONS)
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
                and len(INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]) > 1:
                infoSet = infoSets[infoSetStr]
                row=[infoSetStr,*infoSet.getStrategyTableData(),
                     infoSetStr,f'{infoSet.expectedUtil:.2f}',f'{infoSet.likelihood*100:.2f}%',
                     infoSetStr,*infoSet.getGainTableData()]
          else:
            if infoSetStr[0] == client_hand_rank_ and (len(infoSetStr) + 1) % 2 == client_pos_\
                and len(INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[2:]]) > 1:
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
    suffix = infoSetStr[1:-1]
    return [oppPocket + suffix for oppPocket in RANKS_TURN]


def getDescendantInfoSetStrs(infoSetStr, action):
  # given an infoSet and an action to perform at that infoSet, return all opponent infoSets that can result from it 
  # e.g. given infoSetStr='Bpb' and action='p', return ['Apbp','Bpbp','Cpbp',...]
  actionStr = infoSetStr[1:]+action
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
  for actionsStrs in sorted(INFOSET_ACTION_STRS_TURN, key=lambda x:len(x)):
    for rank in RANKS_TURN:
      infoSetStr = rank + actionsStrs
      infoSets[infoSetStr] = InfoSetData()
      sortedInfoSets.append(infoSetStr)

def initStrategy():
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    actionstr = infoSetStr[1:] if '/' not in infoSetStr else infoSetStr[2:]
    allelgalactions = INFOSET_LEGAL_ACTIONS_TURN[actionstr]
    numlegalactions = len(allelgalactions)
    for action in allelgalactions:
      infoSet.actions[action].strategy = 1/numlegalactions


def updateBeliefs():
    for infoSetStr in sortedInfoSets:
      infoSet = infoSets[infoSetStr]
      if len(infoSetStr) == 1:
        for oppPocket in RANKS_TURN:
          infoSet.beliefs[oppPocket] = FLOP_BUCKET_PROBS[oppPocket] # natural prob of occuring: pre-computed lookup table
      else:
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
            tot += oppInfoSet.actions[lastAction].strategy * FLOP_BUCKET_PROBS[oppInfoSetStr[0]]
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
              infoSet.beliefs[oppPocket]=oppInfoSet.actions[lastAction].strategy * FLOP_BUCKET_PROBS[oppInfoSetStr[0]] / tot
    return


def updateUtilitiesForInfoSetStr(infoSetStr):
    playerIdx = (len(infoSetStr)-1)%2
    infoSet = infoSets[infoSetStr]
    beliefs = infoSet.beliefs
    
    for action in INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]:
        utilFromInfoSets,utilFromTerminalNodes=0,0
        actionStr=infoSetStr[1:]+action
        
        for descendentInfoSetStr in getDescendantInfoSetStrs(infoSetStr,action): # go down the game tree: (infoSetStr='Kpb', action='p') --> ['Qpbp','Jpbp']
            probOfThisInfoSet = beliefs[descendentInfoSetStr[0]]
            
            # we use pockets when we invoke calcUtilityAtTerminalNode below, 
            # we need to switch the order of the pockets when we're calculating player 2's payoff  
            # also: calcUtilityAtTerminalNode always returns [util_p1, utils_p2] regardless of playerIdx (acting player's index)
            
            if playerIdx == 0:
               pockets=[infoSetStr[0],descendentInfoSetStr[0]]
            else: # if this is player 2's turn..
               pockets=[descendentInfoSetStr[0],infoSetStr[0]]
            
            if actionStr in TERMINAL_ACTION_STRS_TURN:
                # choosing this action moves us to a terminal node
                utilFromTerminalNodes+=probOfThisInfoSet*calcUtilityAtTerminalNode(*pockets, actionStr[-1], playerIdx, TERMINAL_CHIPCOUNT_TURN[actionStr], playerIdx)
            else:
                # choosing this action moves us to an opponent infoSet where they will choose an action (depending on their strategy, 
                # which is also OUR strategy bc this is self-play)
                descendentInfoSet = infoSets[descendentInfoSetStr]
                for oppAction in INFOSET_LEGAL_ACTIONS_TURN[actionStr]:
                    probOfOppAction = descendentInfoSet.actions[oppAction].strategy
                    destinationInfoSetStr = infoSetStr+action+oppAction
                    destinationActionStr = destinationInfoSetStr[1:]
                    if destinationActionStr in TERMINAL_ACTION_STRS_TURN:
                        # our opponent choosing that action moves us to a terminal node
                        utilFromTerminalNodes+=probOfThisInfoSet*probOfOppAction*\
                          calcUtilityAtTerminalNode(*pockets,destinationActionStr[-1], (playerIdx+1)%2, TERMINAL_CHIPCOUNT_TURN[destinationActionStr], playerIdx)
                    else:
                        # it's another infoSet, and we've already calculated the expectedUtility of this infoSet
                        # ^^ the utility must've been computed as we are traversing the game tree from bottom up
                        utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[destinationInfoSetStr].expectedUtil

        infoSet.actions[action].util=utilFromInfoSets+utilFromTerminalNodes
    
    infoSet.expectedUtil = 0 # Start from nothing, neglecting illegal actions
    for action in INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]:
        actionData = infoSet.actions[action]
        infoSet.expectedUtil+=actionData.strategy*actionData.util  # weighted sum of utils associated with each action


def calcInfoSetLikelihoods():
  # calculate the likelihood (aka "reach probability") of reaching each infoSet assuming the infoSet "owner" (the player who acts at that infoSet) tries to get there 
  # (and assuming the other player simply plays according to the current strategy)
  
  #for infosets in preflop
  for infoSetStr in sortedInfoSets:
    infoSet=infoSets[infoSetStr]
    infoSet.likelihood=0 #reset it to zero on each iteration so the likelihoods donnot continually grow (bc we're using += below)
    if len(infoSetStr)==1:
      # the likelihood of the top-level infoSets (A, B, C,...) is determined solely by precomputed natural probs.
      infoSet.likelihood=FLOP_BUCKET_PROBS[infoSetStr[0]]
    elif len(infoSetStr)==2:  # P2's perspective
      # the second-tier infoSet likelihoods. Note, the second-tier infoSet, e.g., 'Bb', may have resulted from the top-tier infoSets 'A', 'B',...
      # depending on which hand tier player 1 has. The likelihood of 'Bb' is therefore the multiplication of the likelihood along each of these possible paths
      for oppPocket in RANKS_TURN:
        oppInfoSet = infoSets[oppPocket]
        infoSet.likelihood+=oppInfoSet.actions[infoSetStr[-1]].strategy*FLOP_BUCKET_PROBS[infoSetStr[0]]*\
          FLOP_BUCKET_PROBS[oppPocket]  # once again this is natural prob
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
        infoSet.likelihood+=infoSetTwoLevelsAgo.likelihood*infoSetTwoLevelsAgo.actions[infoSetStr[-2]].strategy *FLOP_BUCKET_PROBS[oppPocket]*\
          oppInfoSet.actions[infoSetStr[-1]].strategy 
        # ^^ note, each oppInfoSet is essentially slicing up the infoSetTwoLevelsAgo because they're each assuming a specific oppPocket. 
        # ^^ Therefore, we must account for the prob. of each opponent pocket


def calcGains(cur_t, alpha = 2.0061, beta = 1.0031):
  # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
  # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
  # we note that in first several iterations the gains are very large, and thus in later iterations its hard to change the strategy since denominator is large
  # so we use alpha to scale the gains, and thus make the strategy converge faster and more to 0
  totAddedGain=0.0
  max_now = 0.0
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    for action in INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]:
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
#     for action in INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]:
#       utilForActionPureStrat = infoSet.actions[action].util 
#       gain = max(0, utilForActionPureStrat-infoSet.expectedUtil)
#       totAddedGain+=gain
#       max_now = max(gain, max_now)
#       infoSet.actions[action].cumulativeGain += gain * infoSet.likelihood
#   print(max_now)
#   return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)


# def updateStrategy(cur_t, gamma = 0.0):
#   # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
#   for infoSetStr in sortedInfoSets:
#     infoSet = infoSets[infoSetStr]
#     allLegalActions = INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]

#     totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
#     likelihood = infoSet.likelihood
#     totStrategy = 0.0
#     for action in ACTIONS:
#         cur_strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0
#         infoSet.actions[action].cumulativeStrategy = infoSet.actions[action].cumulativeStrategy * math.pow((cur_t - 1) / cur_t, gamma) + cur_strategy * likelihood
#         # infoSet.actions[action].cumulativeStrategy = cur_strategy * likelihood

#         totStrategy += infoSet.actions[action].cumulativeStrategy
#         if infoSet.actions[action].cumulativeStrategy != 0:
#             all0 = False
#     for action in ACTIONS:
#         infoSet.actions[action].strategy = infoSet.actions[action].cumulativeStrategy / totStrategy if action in allLegalActions else 0.0 #why using cumulativeGain instead of gain, aka counterfactual regret?
#         if infoSet.actions[action].strategy != 0:
#             all0 = False     

def updateStrategy(cur_t, gamma = 2.0):
  # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    allLegalActions = INFOSET_LEGAL_ACTIONS_TURN[infoSetStr[1:]]
    totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
    for action in ACTIONS:
        infoSet.actions[action].strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0

def plot_strategy_evolution(iterations, first_strategy_history, second_strategy_history):
    """
    Plot the evolution of strategies based on saved history data.
    """
    ranks = RANKS_TURN
    actions = ['k', 'c', 'b', 'f', 'r']
    colors = {'k': 'blue', 'c': 'green', 'b': 'red', 'f': 'purple', 'r': 'orange'}
    
    for rank in ranks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot first-level strategy
        for action in actions:
            ax1.plot(iterations, first_strategy_history[rank][action], 
                    label=f'{action}', color=colors[action], marker='o', markersize=3)
        
        ax1.set_title(f'Rank {rank} First-Level Strategy')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Strategy Probability')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True)
        ax1.legend()
        
        # Plot second-level strategy
        for action in actions:
            ax2.plot(iterations, second_strategy_history[rank][action], 
                    label=f'{action}', color=colors[action], marker='o', markersize=3)
        
        ax2.set_title(f'Rank {rank} Second-Level Strategy')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Strategy Probability')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'rank_{rank}_strategy_evolution.png')
        plt.show()

def plot_infoset_strategies(iterations, infoset_action_str, strategy_history_dict):
    """
    Plot strategy evolution for all 15 buckets (A-O) for a specific infoset action string.
    
    Args:
        iterations: List of iteration numbers
        infoset_action_str: The action string part of the infoset (e.g., '', 'b', 'k')
        strategy_history_dict: Dictionary with strategy history for all ranks and actions
    """
    ranks = RANKS_TURN  # All 15 buckets from A to O
    actions = ['k', 'c', 'b', 'f', 'r']
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
    plt.show()

def main():
    start = time.time()


    # profiler = cProfile.Profile()
    # profiler.enable()

    # calcTransitionProbs(6,1)
    

    # profiler.disable()
    # profiler.dump_stats("tProbs.prof")

    initInfoSets()
    # print('>>>SORTED INFO SETS>>>\n{}\n<<<'.format(sortedInfoSets))
    initStrategy()

    numIterations=8000  # 10k converges a lot better; 1k ~ 40s
    totGains = []
    iterations = []
    time_belief = 0
    time_util = 0
    time_likelihood = 0
    time_gain = 0
    time_strategy = 0
    descendtot = 0
    ranks = RANKS_TURN
    actions = ['k', 'c', 'b', 'f', 'r']
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
            updateUtilitiesForInfoSetStr(infoSetStr)
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
            
            # Save first-level strategy
            for rank in ranks:
                infoset = infoSets[rank]
                for action in actions:
                    first_strategy_history[rank][action].append(infoset.actions[action].strategy)
            
            # Calculate and save second-level strategy
            for rank in ranks:
                secondstra = defaultdict(float)
                for rank2 in RANKS_TURN:
                    secondstra['k'] += infoSets[rank + 'k'].actions['k'].strategy * infoSets[rank2].actions['k'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['c'] += infoSets[rank + 'b'].actions['c'].strategy * infoSets[rank2].actions['b'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['c'] += infoSets[rank + 'r'].actions['c'].strategy * infoSets[rank2].actions['r'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['b'] += infoSets[rank + 'r'].actions['b'].strategy * infoSets[rank2].actions['r'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['b'] += infoSets[rank + 'b'].actions['b'].strategy * infoSets[rank2].actions['b'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['r'] += infoSets[rank + 'r'].actions['r'].strategy * infoSets[rank2].actions['r'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['r'] += infoSets[rank + 'b'].actions['r'].strategy * infoSets[rank2].actions['b'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['f'] += infoSets[rank + 'r'].actions['f'].strategy * infoSets[rank2].actions['r'].strategy * FLOP_BUCKET_PROBS[rank2]
                    secondstra['f'] += infoSets[rank + 'b'].actions['f'].strategy * infoSets[rank2].actions['b'].strategy * FLOP_BUCKET_PROBS[rank2]
                    # ... other second-level strategy calculations ...
                
                for action in actions:
                    second_strategy_history[rank][action].append(secondstra[action])
    
        
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

    with open('infoSets_FLOP_raise.pkl','wb') as f:
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
    
    infoset_action_strs = sorted(list(INFOSET_ACTION_STRS_TURN))
    for action_str in infoset_action_strs:
        print(f"Plotting strategies for infoset action string: '{action_str}'")
        plot_infoset_strategies(iterationstra, action_str, strategy_history)
        
    # plot_strategy_evolution(iterationstra, first_strategy_history, second_strategy_history)
    
    
    # for rank in RANKS_TURN:
    #     secondstra = defaultdict(float)
    #     infoset = infoSets[rank]
    #     print(rank)
    #     print(infoset.actions['k'].strategy)
    #     print(infoset.actions['c'].strategy)
    #     print(infoset.actions['b'].strategy)
    #     print(infoset.actions['f'].strategy)
    #     print(infoset.actions['r'].strategy)
    #     for rank2 in RANKS_TURN:
    #         secondstra['k'] += infoSets[rank + 'k'].actions['k'].strategy * infoSets[rank2].actions['k'].strategy
    #         secondstra['c'] += infoSets[rank + 'b'].actions['c'].strategy * infoSets[rank2].actions['b'].strategy
    #         secondstra['c'] += infoSets[rank + 'r'].actions['c'].strategy * infoSets[rank2].actions['r'].strategy
    #         secondstra['b'] += infoSets[rank + 'r'].actions['b'].strategy * infoSets[rank2].actions['r'].strategy
    #         secondstra['b'] += infoSets[rank + 'b'].actions['b'].strategy * infoSets[rank2].actions['b'].strategy
    #         secondstra['r'] += infoSets[rank + 'r'].actions['r'].strategy * infoSets[rank2].actions['r'].strategy
    #         secondstra['r'] += infoSets[rank + 'b'].actions['r'].strategy * infoSets[rank2].actions['b'].strategy
    #         secondstra['f'] += infoSets[rank + 'r'].actions['f'].strategy * infoSets[rank2].actions['r'].strategy
    #         secondstra['f'] += infoSets[rank + 'b'].actions['f'].strategy * infoSets[rank2].actions['b'].strategy
    #     print(secondstra['k'])
    #     print(secondstra['c'])
    #     print(secondstra['b'])
    #     print(secondstra['f'])
    #     print(secondstra['r'])
    #     print(sum(secondstra.values()))

if __name__ == '__main__':
    main()
