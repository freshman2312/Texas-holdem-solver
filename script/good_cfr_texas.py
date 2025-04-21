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


infoSets: dict[str, InfoSetData] = {}  # global
sortedInfoSets = [] # global

# ASSUMPTION: consider PREFLOP **only**
# TODO: [1] compute prob of occurence for BUCKETS; [2] filter out illegal actions; [3] compare two hands of the same tier (how?)

RANKS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]  # based on preflop chart: reflects WR
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I"}
ACTIONS = ["k", "c", "b", "f"]  # {k: check, c: call, b: bet/raise/all-in, f: fold}

s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
num2letter = {}
for n in range(1, 53):
    num2letter[n] = s[n-1]

TERMINAL_ACTION_STRS = {
  'f', 'bf', 'bc', 'bbf', 'bbc', 'bbbf', 'bbbc', 'bbbbf', 'bbbbc'
} # terminal action paths where all decisions have already been made (terminal nodes are NOT considered infoSets here, bc no decision needs to be made)
TERMINAL_CHIPCOUNT = {
  'f': 1, 'bf': 1, 'bc': 2, 'bbf': 2, 'bbc': 4, 'bbbf': 4, 'bbbc': 8, 'bbbbf': 8, 'bbbbc': 16
}
INFOSET_ACTION_STRS = {
  '', 'b', 'bb', 'bbb', 'bbbb'
} # action paths where a decision still needs to be made by one of the players (i.e. actions paths that end on an infoSet)
INFOSET_LEGAL_ACTIONS = {
  '': ['f', 'b'], 'b': ['c', 'f', 'b'], 'bb': ['c', 'f', 'b'], 'bbb': ['c', 'f', 'b'], 'bbbb': ['c', 'f']
} # TODO: separate infosets which yield only **1** legal action: can they be considered as leaf nodes?

NUM_RANKS = len(RANKS)
NUM_ACTIONS = len(ACTIONS)

PREFLOP_WR = {
   '22': 51, '33': 55, '44': 58, '55': 61, '66': 64, '77': 67, '88': 69, '99': 72, 'TT': 75, 'JJ': 78, 'QQ': 80, 'KK': 83, 'AA': 84,
   '23o': 35, '24o': 36, '25o': 37, '26o': 37, '27o': 37, '28o': 40, '29o': 42, '2To': 44, '2Jo': 47, '2Qo': 49, '2Ko': 53, '2Ao': 57,
   '34o': 38, '35o': 39, '36o': 39, '37o': 39, '38o': 40, '39o': 43, '3To': 45, '3Jo': 48, '3Qo': 50, '3Ko': 54, '3Ao': 58,
   '45o': 41, '46o': 41, '47o': 41, '48o': 42, '49o': 43, '4To': 46, '4Jo': 48, '4Qo': 51, '4Ko': 54, '4Ao': 59,
   '56o': 43, '57o': 43, '58o': 44, '59o': 45, '5To': 47, '5Jo': 49, '5Qo': 52, '5Ko': 55, '5Ao': 60,
   '67o': 45, '68o': 46, '69o': 47, '6To': 48, '6Jo': 50, '6Qo': 53, '6Ko': 56, '6Ao': 59,
   '78o': 47, '79o': 48, '7To': 50, '7Jo': 52, '7Qo': 54, '7Ko': 57, '7Ao': 60,
   '89o': 50, '8To': 52, '8Jo': 53, '8Qo': 55, '8Ko': 58, '8Ao': 61,
   '9To': 53, '9Jo': 55, '9Qo': 57, '9Ko': 59, '9Ao': 62,
   'TJo': 57, 'TQo': 59, 'TKo': 61, 'TAo': 64,
   'JQo': 59, 'JKo': 62, 'JAo': 65,
   'QKo': 62, 'QAo': 65, 'KAo': 66, # ------------------------------------------------------------------------------------------------
   '23s': 39, '24s': 40, '25s': 41, '26s': 40, '27s': 41, '28s': 43, '29s': 45, '2Ts': 47, '2Js': 50, '2Qs': 52, '2Ks': 55, '2As': 59,
   '34s': 42, '35s': 43, '36s': 42, '37s': 43, '38s': 43, '39s': 46, '3Ts': 48, '3Js': 50, '3Qs': 53, '3Ks': 56, '3As': 60,
   '45s': 44, '46s': 44, '47s': 45, '48s': 45, '49s': 46, '4Ts': 49, '4Js': 51, '4Qs': 54, '4Ks': 57, '4As': 61,
   '56s': 46, '57s': 46, '58s': 47, '59s': 48, '5Ts': 49, '5Js': 52, '5Qs': 55, '5Ks': 58, '5As': 62,
   '67s': 48, '68s': 49, '69s': 50, '6Ts': 51, '6Js': 53, '6Qs': 55, '6Ks': 58, '6As': 62,
   '78s': 50, '79s': 51, '7Ts': 53, '7Js': 54, '7Qs': 56, '7Ks': 59, '7As': 63,
   '89s': 53, '8Ts': 54, '8Js': 56, '8Qs': 58, '8Ks': 60, '8As': 63,
   '9Ts': 56, '9Js': 57, '9Qs': 59, '9Ks': 61, '9As': 64,
   'TJs': 59, 'TQs': 61, 'TKs': 63, 'TAs': 66,
   'JQs': 61, 'JKs': 64, 'JAs': 66,
   'QKs': 64, 'QAs': 67, 'KAs': 68,
}

PREFLOP_BUCKETS = {w: (PREFLOP_WR[w]-30)//5 for w in PREFLOP_WR}
# PREFLOP_BUCKETS = { # Set 'AA' to 10 (instead of 11) so that 11 buckets --> 10
#    '22': 4, '33': 5, '44': 5, '55': 6, '66': 6, '77': 7, '88': 7, '99': 8, 'TT': 8, 'JJ': 10, 'QQ': 9, 'KK': 9, 'AA': 9, '23o': 1, '24o': 1, '25o': 1, '26o': 1, '27o': 1, '28o': 2, '29o': 2, '2To': 2, '2Jo': 3, '2Qo': 3, '2Ko': 4, '2Ao': 5, '34o': 1, '35o': 1, '36o': 1, '37o': 1, '38o': 2, '39o': 2, '3To': 3, '3Jo': 3, '3Qo': 4, '3Ko': 4, '3Ao': 5, '45o': 2, '46o': 2, '47o': 2, '48o': 2, '49o': 2, '4To': 3, '4Jo': 3, '4Qo': 4, '4Ko': 4, '4Ao': 5, '56o': 2, '57o': 2, '58o': 2, '59o': 3, '5To': 3, '5Jo': 3, '5Qo': 4, '5Ko': 5, '5Ao': 6, '67o': 3, '68o': 3, '69o': 3, '6To': 3, '6Jo': 4, '6Qo': 4, '6Ko': 5, '6Ao': 5, '78o': 3, '79o': 3, '7To': 4, '7Jo': 4, '7Qo': 4, '7Ko': 5, '7Ao': 6, '89o': 4, '8To': 4, '8Jo': 4, '8Qo': 5, '8Ko': 5, '8Ao': 6, '9To': 4, '9Jo': 5, '9Qo': 5, '9Ko': 5, '9Ao': 6, 'TJo': 5, 'TQo': 5, 'TKo': 6, 'TAo': 6, 'JQo': 5, 'JKo': 6, 'JAo': 7, 'QKo': 6, 'QAo': 7, 'KAo': 7, '23s': 1, '24s': 2, '25s': 2, '26s': 2, '27s': 2, '28s': 2, '29s': 3, '2Ts': 3, '2Js': 4, '2Qs': 4, '2Ks': 5, '2As': 5, '34s': 2, '35s': 2, '36s': 2, '37s': 2, '38s': 2, '39s': 3, '3Ts': 3, '3Js': 4, '3Qs': 4, '3Ks': 5, '3As': 6, '45s': 2, '46s': 2, '47s': 3, '48s': 3, '49s': 3, '4Ts': 3, '4Js': 4, '4Qs': 4, '4Ks': 5, '4As': 6, '56s': 3, '57s': 3, '58s': 3, '59s': 3, '5Ts': 3, '5Js': 4, '5Qs': 5, '5Ks': 5, '5As': 6, '67s': 3, '68s': 3, '69s': 4, '6Ts': 4, '6Js': 4, '6Qs': 5, '6Ks': 5, '6As': 6, '78s': 4, '79s': 4, '7Ts': 4, '7Js': 4, '7Qs': 5, '7Ks': 5, '7As': 6, '89s': 4, '8Ts': 4, '8Js': 5, '8Qs': 5, '8Ks': 6, '8As': 6, '9Ts': 5, '9Js': 5, '9Qs': 5, '9Ks': 6, '9As': 6, 'TJs': 5, 'TQs': 6, 'TKs': 6, 'TAs': 7, 'JQs': 6, 'JKs': 6, 'JAs': 7, 'QKs': 6, 'QAs': 7, 'KAs': 7
# }

# PREFLOP_BUCKET_CARDS = {}
# for _ in PREFLOP_BUCKETS: 
#    PREFLOP_BUCKET_CARDS.setdefault(PREFLOP_BUCKETS[_],[]).append(_)
PREFLOP_BUCKET_CARDS = {
   4: ['22', '2Ko', '3Qo', '3Ko', '4Qo', '4Ko', '5Qo', '6Jo', '6Qo', '7To', '7Jo', '7Qo', '89o', '8To', '8Jo', '9To', '2Js', '2Qs', '3Js', '3Qs', '4Js', '4Qs', '5Js', '69s', '6Ts', '6Js', '78s', '79s', '7Ts', '7Js', '89s', '8Ts'], 5: ['33', '44', '2Ao', '3Ao', '4Ao', '5Ko', '6Ko', '6Ao', '7Ko', '8Qo', '8Ko', '9Jo', '9Qo', '9Ko', 'TJo', 'TQo', 'JQo', '2Ks', '2As', '3Ks', '4Ks', '5Qs', '5Ks', '6Qs', '6Ks', '7Qs', '7Ks', '8Js', '8Qs', '9Ts', '9Js', '9Qs', 'TJs'], 6: ['55', '66', '5Ao', '7Ao', '8Ao', '9Ao', 'TKo', 'TAo', 'JKo', 'QKo', '3As', '4As', '5As', '6As', '7As', '8Ks', '8As', '9Ks', '9As', 'TQs', 'TKs', 'JQs', 'JKs', 'QKs'], 7: ['77', '88', 'JAo', 'QAo', 'KAo', 'TAs', 'JAs', 'QAs', 'KAs'], 8: ['99', 'TT', 'JJ'], 9: ['QQ', 'KK', 'AA'], 1: ['23o', '24o', '25o', '26o', '27o', '34o', '35o', '36o', '37o', '23s'], 2: ['28o', '29o', '2To', '38o', '39o', '45o', '46o', '47o', '48o', '49o', '56o', '57o', '58o', '24s', '25s', '26s', '27s', '28s', '34s', '35s', '36s', '37s', '38s', '45s', '46s'], 3: ['2Jo', '2Qo', '3To', '3Jo', '4To', '4Jo', '59o', '5To', '5Jo', '67o', '68o', '69o', '6To', '78o', '79o', '29s', '2Ts', '39s', '3Ts', '47s', '48s', '49s', '4Ts', '56s', '57s', '58s', '59s', '5Ts', '67s', '68s']
}

# PREFLOP_BUCKET_PROBS = {}
# for _ in range(1, 11):
#     PREFLOP_BUCKET_PROBS[_] = sum(1 for v in PREFLOP_BUCKETS.values() if v == _)/169
BUCKET_PROBS = {"A": 0.05917159763313609, "B": 0.14792899408284024, "C": 0.17751479289940827, "D": 0.1893491124260355, "E": 0.1952662721893491, "F": 0.14201183431952663, "G": 0.05325443786982249, "H": 0.017751479289941, "I": 0.017751479289941}

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
        self.likelihood: float = None

    @staticmethod
    def printInfoSetDataTable(infoSets: dict[str,InfoSetData], client_hand_rank_, client_pos_):
        print()
        # print the various values for the infoSets in a nicely formatted table
        rows=[]
        for infoSetStr in sortedInfoSets:
            if infoSetStr[0] == client_hand_rank_ and (len(infoSetStr) + 1) % 2 == client_pos_\
                and len(INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]) > 1:
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


def getAncestralInfoSetStrs(infoSetStr) -> list[InfoSetData]:
    # given an infoSet, return all opponent infoSets that can lead to it (e.g. given 'Bpb', return ['Ap','Bp','Cp',...])
    if len(infoSetStr) == 1:
        raise ValueError(f'no ancestors of infoSet={infoSetStr}')
    
    return [oppPocket + infoSetStr[1:-1] for oppPocket in RANKS]


def getDescendantInfoSetStrs(infoSetStr, action):
  # given an infoSet and an action to perform at that infoSet, return all opponent infoSets that can result from it 
  # e.g. given infoSetStr='Bpb' and action='p', return ['Apbp','Bpbp','Cpbp',...]
  actionStr = infoSetStr[1:]+action
  return [oppPocket+actionStr for oppPocket in RANKS]

def calcUtilityAtTerminalNode(pocket1, pocket2, action1, playerIdx_, totalBets, playerIdx2return):
  if action1 == 'f':
    return -totalBets if playerIdx2return == playerIdx_ else totalBets
  else:  # showdown
    if RANK2NUM[pocket1] > RANK2NUM[pocket2]:
       return totalBets if playerIdx2return == 0 else -totalBets
    elif RANK2NUM[pocket1] == RANK2NUM[pocket2]: # TODO: better tie breaker?
       return totalBets if randint(1,100) <= 50 else -totalBets
    else:
       return -totalBets if playerIdx2return == 0 else totalBets


def initInfoSets():
    # initialize the infoSet objects.
    for actionsStrs in sorted(INFOSET_ACTION_STRS, key=lambda x:len(x)):
        for rank in RANKS:
            infoSetStr = rank + actionsStrs
            infoSets[infoSetStr] = InfoSetData()
            sortedInfoSets.append(infoSetStr)


def updateBeliefs():
    for infoSetStr in sortedInfoSets:
        infoSet = infoSets[infoSetStr]
        if len(infoSetStr) == 1:
            for oppPocket in RANKS:
              infoSet.beliefs[oppPocket] = BUCKET_PROBS[oppPocket] # natural prob of occuring: pre-computed lookup table
        else:
            ancestralInfoSetStrs = getAncestralInfoSetStrs(infoSetStr) 
            lastAction = infoSetStr[-1]
            tot = 0  # normalizing factor for strategy (last action)
            for oppInfoSetStr in ancestralInfoSetStrs:
                oppInfoSet=infoSets[oppInfoSetStr]
                # try:
                #    oppInfoSet=infoSets[oppInfoSetStr]
                # except KeyError:
                #    print('infoSetStr: {} | ancestralInfoSetStrs: {} | lastAction: {}'.format(infoSetStr, ancestralInfoSetStrs, lastAction))

                tot += oppInfoSet.actions[lastAction].strategy
            for oppInfoSetStr in ancestralInfoSetStrs:
                oppInfoSet=infoSets[oppInfoSetStr]
                oppPocket = oppInfoSetStr[0]
                infoSet.beliefs[oppPocket]=oppInfoSet.actions[lastAction].strategy / tot
    return

def updateUtilitiesForInfoSetStr(infoSetStr):
    playerIdx = (len(infoSetStr)-1)%2
    infoSet = infoSets[infoSetStr]
    beliefs = infoSet.beliefs
    
    for action in INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]:
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
            
            if actionStr in TERMINAL_ACTION_STRS:
                # choosing this action moves us to a terminal node
                utilFromTerminalNodes+=probOfThisInfoSet*calcUtilityAtTerminalNode(*pockets, actionStr[-1], playerIdx, TERMINAL_CHIPCOUNT[actionStr], playerIdx)
            else:
                # choosing this action moves us to an opponent infoSet where they will choose an action (depending on their strategy, 
                # which is also OUR strategy bc this is self-play)
                descendentInfoSet = infoSets[descendentInfoSetStr]
                for oppAction in INFOSET_LEGAL_ACTIONS[actionStr]:
                    probOfOppAction = descendentInfoSet.actions[oppAction].strategy
                    destinationInfoSetStr = infoSetStr+action+oppAction
                    destinationActionStr = destinationInfoSetStr[1:]
                    if destinationActionStr in TERMINAL_ACTION_STRS:
                        # our opponent choosing that action moves us to a terminal node
                        utilFromTerminalNodes+=probOfThisInfoSet*probOfOppAction*\
                          calcUtilityAtTerminalNode(*pockets,destinationActionStr[-1], (playerIdx+1)%2, TERMINAL_CHIPCOUNT[destinationActionStr], playerIdx)
                    else:
                        # it's another infoSet, and we've already calculated the expectedUtility of this infoSet
                        # ^^ the utility must've been computed as we are traversing the game tree from bottom up
                        utilFromInfoSets+=probOfThisInfoSet*probOfOppAction*infoSets[destinationInfoSetStr].expectedUtil

        infoSet.actions[action].util=utilFromInfoSets+utilFromTerminalNodes
    
    infoSet.expectedUtil = 0 # Start from nothing, neglecting illegal actions
    for action in INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]:
        actionData = infoSet.actions[action]
        infoSet.expectedUtil+=actionData.strategy*actionData.util  # weighted sum of utils associated with each action


def calcInfoSetLikelihoods():
  # calculate the likelihood (aka "reach probability") of reaching each infoSet assuming the infoSet "owner" (the player who acts at that infoSet) tries to get there 
  # (and assuming the other player simply plays according to the current strategy)
  for infoSetStr in sortedInfoSets:
    infoSet=infoSets[infoSetStr]
    infoSet.likelihood=0 #reset it to zero on each iteration so the likelihoods donnot continually grow (bc we're using += below)
    if len(infoSetStr)==1:
      # the likelihood of the top-level infoSets (A, B, C,...) is determined solely by precomputed natural probs.
      infoSet.likelihood=BUCKET_PROBS[infoSetStr[0]]
    elif len(infoSetStr)==2:  # P2's perspective
      # the second-tier infoSet likelihoods. Note, the second-tier infoSet, e.g., 'Bb', may have resulted from the top-tier infoSets 'A', 'B',...
      # depending on which hand tier player 1 has. The likelihood of 'Bb' is therefore the multiplication of the likelihood along each of these possible paths
      for oppPocket in RANKS:
        oppInfoSet = infoSets[oppPocket]
        infoSet.likelihood+=oppInfoSet.actions[infoSetStr[-1]].strategy*BUCKET_PROBS[infoSetStr[0]]*BUCKET_PROBS[oppPocket]  # once again this is natural prob
    else:
      # For infoSets on the third-tier and beyond, we can use the likelihoods of the infoSets two levels before to calculate their likelihoods.
      # Note, we can't simply use the infoSet one tier before because that's the opponent's infoSet, and the calculation of likelihoods 
      # assumes that the infoSet's "owner" is trying to reach the infoSet. Therefore, when calculating a liklihood for player 1's infoSet, 
      # we can only use the likelihood of an ancestral infoSet if the ancestral infoSet is also "owned" by player 1, and the closest such infoSet is 2 levels above.
      # Note also, that although there can be multiple ancestral infoSets one tier before, there is only one ancestral infoSet two tiers before. 
      # For example, 'Bbc' has one-tier ancestors 'Ab' and 'Bb', but only a single two-tier ancestor: 'B'

      infoSetTwoLevelsAgo = infoSets[infoSetStr[:-2]] # grab the closest ancestral infoSet with the same owner as the infoSet for which we seek to calculate likelihood
      for oppPocket in RANKS:
        oppInfoSet = infoSets[oppPocket + infoSetStr[1:-1]]
        infoSet.likelihood+=infoSetTwoLevelsAgo.likelihood*BUCKET_PROBS[oppPocket]*oppInfoSet.actions[infoSetStr[-1]].strategy 
        # ^^ note, each oppInfoSet is essentially slicing up the infoSetTwoLevelsAgo because they're each assuming a specific oppPocket. 
        # ^^ Therefore, we must account for the prob. of each opponent pocket


def calcGains():
  # for each action at each infoSet, calc the gains for this round weighted by the likelihood (aka "reach probability")
  # and add these weighted gains for this round to the cumulative gains over all previous iterations for that infoSet-action pair
  totAddedGain=0.0
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    for action in INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]:
      utilForActionPureStrat = infoSet.actions[action].util 
      gain = max(0,utilForActionPureStrat-infoSet.expectedUtil)
      totAddedGain+=gain
      infoSet.actions[action].cumulativeGain+=gain * infoSet.likelihood
  return totAddedGain # return the totAddedGain as a rough measure of convergence (it should grow smaller as we iterate more)


def updateStrategy():
  # update the strategy for each infoSet-action pair to be proportional to the cumulative gain for that action over all previous iterations
  for infoSetStr in sortedInfoSets:
    infoSet = infoSets[infoSetStr]
    allLegalActions = INFOSET_LEGAL_ACTIONS[infoSetStr[1:]]
    totGains = sum([infoSet.actions[action].cumulativeGain for action in allLegalActions])
    for action in ACTIONS:
        infoSet.actions[action].strategy = infoSet.actions[action].cumulativeGain/totGains if action in allLegalActions else 0.0

def calcHoleCardsRankNum(hole_cards_):
  if len(hole_cards_) != 2:
    print("ERROR! Illegal number of hole cards")
    sys.exit(-1)
  hole_card_ranks = [card2rank[hc] for hc in hole_cards_]
  hole_card_ranks.sort()
  hole_cards_str = ''
  for hcr in hole_card_ranks:
    hole_cards_str += rank2str[hcr]
  if hole_card_ranks[0] != hole_card_ranks[1]:
    hole_cards_str += 's' if card2row[hole_cards_[0]] == card2row[hole_cards_[1]] else 'o' # suited vs off-suited
  return PREFLOP_BUCKETS[hole_cards_str]


def twoCardNumsFromStr(cardsStr):
  cardRanks = [str2rank[cs] for cs in cardsStr[:2]]
  cardNums = [1 if cr == 14 else cr for cr in cardRanks]
  randomSuit = randint(0, 3)
  if cardsStr[-1] == 's':
    randomSuit2 = randomSuit
  else:
    allSuits = [0,1,2,3]
    allSuits.remove(randomSuit)
    randomSuit2 = choice(allSuits)
  return [13*randomSuit+cardNums[0], 13*randomSuit2+cardNums[1]]


def calcTransitionProbs(pocket1RankNum, pocket2RankNum):
  res = {}
  deck = list(range(1, 53))
  i_cc3 = 0
  for cc3 in combinations(deck, 3): # Loop through ALL possible 3-card flop combos, and compute their wins/losses
    i_cc3 += 1
    if i_cc3 % 2210 == 0:
      print('calcTransitionProbs() progress: {:.0f}%'.format(i_cc3/221))
    deck3 = [d for d in deck if d not in cc3]
    w, l = 0, 0
    cc3_list = list(cc3)
    cc3_str = ''.join([num2letter[x] for x in cc3_list])

    for _ in range(5):
      pocket1Cards = twoCardNumsFromStr(choice(PREFLOP_BUCKET_CARDS[pocket1RankNum]))  # sample 2 hole cards for player#1
      tries = 0
      while (pocket1Cards[0] in cc3 or pocket1Cards[1] in cc3) and tries < 100:
        pocket1Cards = twoCardNumsFromStr(choice(PREFLOP_BUCKET_CARDS[pocket2RankNum]))
        tries += 1
      
      if tries >= 100:
        if pocket1Cards[0] in cc3 or pocket1Cards[1] in cc3:
          print('Error! At least one of pocket1 cards is in the 3-card flop')
          sys.exit(-1)

      pocket2Cards = twoCardNumsFromStr(choice(PREFLOP_BUCKET_CARDS[pocket2RankNum]))  # sample 2 hole cards for player#2
      tries = 0
      while (pocket2Cards[0] in pocket1Cards or pocket2Cards[1] in pocket1Cards\
             or pocket2Cards[0] in cc3 or pocket2Cards[1] in cc3) and tries < 100:
        pocket2Cards = twoCardNumsFromStr(choice(PREFLOP_BUCKET_CARDS[pocket2RankNum]))
        tries += 1
      
      if tries >= 100:
        if pocket2Cards[0] in pocket1Cards or pocket2Cards[1] in pocket1Cards\
          or pocket2Cards[0] in cc3 or pocket2Cards[1] in cc3:
          print('Error! At least one of pocket2 cards is either in pocket1 or in the 3-card flop')
          sys.exit(-1)

      p1Cards5 = pocket1Cards + cc3_list
      p2Cards5 = pocket2Cards + cc3_list
      deck7 = [d for d in deck3 if d not in pocket1Cards+pocket2Cards] # deep copy

      cc2_all = list(combinations(deck7, 2))  # len=990
      shuffle(cc2_all)
      for cc2 in cc2_all[:100]:
        cc2_list = list(cc2)
        p1Score = compute_score(p1Cards5 + cc2_list)
        p2Score = compute_score
        (p2Cards5 + cc2_list)
        if p1Score > p2Score:
           w += 1
        elif p1Score < p2Score:
           l += 1
  
    res[cc3_str] = (w, l)
  
  with open('tProbs'+str(pocket1RankNum)+str(pocket2RankNum)+'.pkl','wb') as g:
    pickle.dump(res,g,pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
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

    # only plot the gain from every xth iteration (in order to lessen the amount of data that needs to be plotted)
    numGainsToPlot=20
    gainGrpSize = numIterations//numGainsToPlot 
    if gainGrpSize==0:
       gainGrpSize=1

    for i in range(numIterations):
        updateBeliefs()

        for infoSetStr in reversed(sortedInfoSets):  # game tree: from bottom up
            updateUtilitiesForInfoSetStr(infoSetStr)

        calcInfoSetLikelihoods()
        totGain = calcGains()
        if i%gainGrpSize==0: # every 10 or 100 or x rounds, save off the gain so we can plot it afterwards and visually see convergence
            totGains.append(totGain)
            # print(f'TOT_GAIN {totGain: .3f}  @{i}/{numIterations}')
        updateStrategy()
        
        if i % 10 == 0:
          levelprob = defaultdict(float)
          for infoSetStr in sortedInfoSets:
              actionstr = infoSetStr[1:] if '/' not in infoSetStr else infoSetStr[2:]
              level = len(actionstr)
              levelprob[level] += infoSets[infoSetStr].likelihood
          print(levelprob)


    # profiler.disable()
    # profiler.dump_stats("good_cfr_texas.prof")

    InfoSetData.printInfoSetDataTable(infoSets, client_hand_rank, client_pos)
    print('\ntotGains:', totGains)
