# use bayesian optimization to find the best alpha and beta values for CFR algorithm
# also tried to optimize the distribution of buckets, but failed since it turns out that hands will concentrate on some buckets and 
# thus the distribution of buckets is not uniform, and thus the optimization is not effective.

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

# Constants (existing code)
# RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
# RANKS_TURN = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T"}
ACTIONS = ["k", "c", "b", "f", "r"]  # {k: check, c: call, b: bet/raise/all-in, f: fold}

TERMINAL_ACTION_STRS_RIVER = {
    'bc', 'bbc', 'bbbc', 'bbbbc', 'f', 'bf', 'bbf', 'bbbf', 'bbbbf', 'kbc', 'kbbc', 'kbbbc', 'kbbbbc', 'kbf', 'kbbf', 'kbbbf', 'kbbbbf', 'kk',
    'rc', 'rrc', 'rrrc', 'rrrrc', 'f', 'rf', 'rrf', 'rrrf', 'rrrrf', 'krc', 'krrc', 'krrrc', 'krrrrc', 'krf', 'krrf', 'krrrf', 'krrrrf', 'kk',
    'rbc', 'brc', 'rbbc', 'brbc', 'bbrc', 'rrbc', 'rbrc', 'brrc', 'rbbbc', 'brbbc', 'bbrbc', 'bbbrc', 'rrbbc', 'rbrbc', 'brbrc', 'rbbrc', 'brrbc', 'bbrrc', 'rrrbc',
    'rrbrc', 'rbrrc', 'brrrc', 'rbf', 'brf', 'rbbf', 'brbf', 'bbrf', 'rrbf', 'rbrf', 'brrf', 'bbrrf', 'rbbbf', 'brbbf', 'bbrbf', 'bbbrf', 'rrbbf', 'rbrbf', 'brbrf',
    'rbbrf', 'brrbf', 'rrrbf', 'rrbrf', 'rbrrf', 'brrrf', 'krbc', 'kbrc', 'krbbc', 'kbrbc', 'kbbrc', 'krrbc', 'krbrc', 'kbrrc', 'krbbbc', 'kbrbbc', 'kbbrbc', 
    'kbbbrc', 'krrbbc', 'krbrbc', 'kbrbrc', 'krbbrc', 'kbrrbc', 'kbbrrc', 'krrrbc', 'krrbrc', 'krbrrc', 'kbrrrc', 'krbf', 'kbrf', 'krbbf', 'kbrbf', 'kbbrf', 'krrbf', 'krbrf', 'kbrrf', 
    'krbbbf', 'kbrbbf', 'kbbrbf', 'kbbbrf', 'krrbbf', 'krbrbf', 'kbrbrf', 'krbbrf', 'kbrrbf', 'kbbrrf', 'krrrbf', 'krrbrf', 'krbrrf', 'kbrrrf'}

TERMINAL_CHIPCOUNT_RIVER = {
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
INFOSET_ACTION_STRS_RIVER = { # preflop + flop, total=5+36
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

INFOSET_LEGAL_ACTIONS_RIVER = { 
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

RIVER_BUCKET_PROBS = {'A': 0.0309, 'B': 0.0279, 'C': 0.0307, 'D': 0.0406, 'E': 0.0412,
                    'F': 0.0389, 'G': 0.0559, 'H': 0.0458, 'I': 0.0723, 'J': 0.1069,
                    'K': 0.1041, 'L': 0.0650, 'M': 0.0532, 'N': 0.0491, 'O': 0.0391,
                    "P": 0.0415, "Q": 0.0325, "R": 0.0369, "S": 0.0428, "T": 0.0446}


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
            "r": InfoSetActionData(initStratVal=1 / NUM_ACTIONS)
        }
        self.beliefs = defaultdict(float)
        self.expectedUtil = None
        self.likelihood = 0

infoSets: dict[str, InfoSetData] = {}  # global
sortedInfoSets = [] # global

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
  for actionsStrs in sorted(INFOSET_ACTION_STRS_RIVER, key=lambda x:len(x)):
    for rank in RANKS_TURN:
      infoSetStr = rank + actionsStrs
      infoSets[infoSetStr] = InfoSetData()
      sortedInfoSets.append(infoSetStr)

def initStrategy():
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    actionstr = infoSetStr[1:] if '/' not in infoSetStr else infoSetStr[2:]
    allelgalactions = INFOSET_LEGAL_ACTIONS_RIVER[actionstr]
    numlegalactions = len(allelgalactions)
    for action in allelgalactions:
      infoSet.actions[action].strategy = 1/numlegalactions


def updateBeliefs():
    for infoSetStr in sortedInfoSets:
      infoSet = infoSets[infoSetStr]
      if len(infoSetStr) == 1:
        for oppPocket in RANKS_TURN:
          infoSet.beliefs[oppPocket] = RIVER_BUCKET_PROBS[oppPocket] # natural prob of occuring: pre-computed lookup table
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
            tot += oppInfoSet.actions[lastAction].strategy * RIVER_BUCKET_PROBS[oppInfoSetStr[0]]
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
              infoSet.beliefs[oppPocket]=oppInfoSet.actions[lastAction].strategy * RIVER_BUCKET_PROBS[oppInfoSetStr[0]] / tot
    return


def updateUtilitiesForInfoSetStr(infoSetStr):
    playerIdx = (len(infoSetStr)-1)%2
    infoSet = infoSets[infoSetStr]
    beliefs = infoSet.beliefs
    
    for action in INFOSET_LEGAL_ACTIONS_RIVER[infoSetStr[1:]]:
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
            
            if actionStr in TERMINAL_ACTION_STRS_RIVER:
                # choosing this action moves us to a terminal node
                utilFromTerminalNodes+=probOfThisInfoSet*calcUtilityAtTerminalNode(*pockets, actionStr[-1], playerIdx, TERMINAL_CHIPCOUNT_RIVER[actionStr], playerIdx)
            else:
                # choosing this action moves us to an opponent infoSet where they will choose an action (depending on their strategy, 
                # which is also OUR strategy bc this is self-play)
                descendentInfoSet = infoSets[descendentInfoSetStr]
                for oppAction in INFOSET_LEGAL_ACTIONS_RIVER[actionStr]:
                    probOfOppAction = descendentInfoSet.actions[oppAction].strategy
                    destinationInfoSetStr = infoSetStr+action+oppAction
                    destinationActionStr = destinationInfoSetStr[1:]
                    if destinationActionStr in TERMINAL_ACTION_STRS_RIVER:
                        # our opponent choosing that action moves us to a terminal node
                        utilFromTerminalNodes+=probOfThisInfoSet*probOfOppAction*\
                          calcUtilityAtTerminalNode(*pockets,destinationActionStr[-1], (playerIdx+1)%2, TERMINAL_CHIPCOUNT_RIVER[destinationActionStr], playerIdx)
                    else:
                        # it's another infoSet, and we've already calculated the expectedUtility of this infoSet
                        # ^^ the utility must've been computed as we are traversing the game tree from bottom up
                        utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[destinationInfoSetStr].expectedUtil

        infoSet.actions[action].util=utilFromInfoSets+utilFromTerminalNodes
    
    infoSet.expectedUtil = 0 # Start from nothing, neglecting illegal actions
    for action in INFOSET_LEGAL_ACTIONS_RIVER[infoSetStr[1:]]:
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
      infoSet.likelihood=RIVER_BUCKET_PROBS[infoSetStr[0]]
    elif len(infoSetStr)==2:  # P2's perspective
      # the second-tier infoSet likelihoods. Note, the second-tier infoSet, e.g., 'Bb', may have resulted from the top-tier infoSets 'A', 'B',...
      # depending on which hand tier player 1 has. The likelihood of 'Bb' is therefore the multiplication of the likelihood along each of these possible paths
      for oppPocket in RANKS_TURN:
        oppInfoSet = infoSets[oppPocket]
        infoSet.likelihood+=oppInfoSet.actions[infoSetStr[-1]].strategy*RIVER_BUCKET_PROBS[infoSetStr[0]]*\
          RIVER_BUCKET_PROBS[oppPocket]  # once again this is natural prob
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
        infoSet.likelihood+=infoSetTwoLevelsAgo.likelihood*infoSetTwoLevelsAgo.actions[infoSetStr[-2]].strategy *RIVER_BUCKET_PROBS[oppPocket]*\
          oppInfoSet.actions[infoSetStr[-1]].strategy 
        # ^^ note, each oppInfoSet is essentially slicing up the infoSetTwoLevelsAgo because they're each assuming a specific oppPocket. 
        # ^^ Therefore, we must account for the prob. of each opponent pocket


def calcGains(cur_t, alpha = 4.0, beta = 2.0):
  # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
  # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
  # we note that in first several iterations the gains are very large, and thus in later iterations its hard to change the strategy since denominator is large
  # so we use alpha to scale the gains, and thus make the strategy converge faster and more to 0
  totAddedGain=0.0
  max_now = 0.0
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    for action in INFOSET_LEGAL_ACTIONS_RIVER[infoSetStr[2:]] if '/' in infoSetStr else INFOSET_LEGAL_ACTIONS_RIVER[infoSetStr[1:]]:
      utilForActionPureStrat = infoSet.actions[action].util 
      gain = max(0, utilForActionPureStrat-infoSet.expectedUtil)
      totAddedGain += gain
      max_now = max(gain, max_now)
      if infoSet.actions[action].cumulativeGain > 0:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, alpha) / (math.pow(cur_t, alpha) + 1)) + gain
      else:
        infoSet.actions[action].cumulativeGain = infoSet.actions[action].cumulativeGain * (math.pow(cur_t, beta) / (math.pow(cur_t, beta) + 1)) + gain
  return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)


def updateStrategy(cur_t, gamma = 2.0):
  # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    allLegalActions = INFOSET_LEGAL_ACTIONS_RIVER[infoSetStr[1:]]
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
        gain = run_cfr_with_alpha_beta(alpha, num_iterations=2000)
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