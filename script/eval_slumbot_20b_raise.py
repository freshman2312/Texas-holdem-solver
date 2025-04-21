# program used to play with slumbot. 
# rules of using API can be looked up in www.slumbot.com
# use 15bucks and allow raise action.
# expected bb/100 is undecided since hands are not enough 

import requests
import sys
import argparse


sys.path.append('..')
# from texasholdem.env.game import GameEnv
# from texasholdem.evaluation.simulation import load_card_play_models
from random import choice, choices
from os.path import isfile
# from time import time
from re import sub
import pickle
from cfr_texas23 import getDBKey, calcHoleCardsRankNum, CLUSTERS01, CLUSTERS2
import multiprocessing as mp
import matplotlib.pyplot as plt
import urllib3
from typing import Dict, List, Tuple
from collections import defaultdict

RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T"}

INFOSET_LEGAL_ACTIONS = { 
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


class SignificantHand:
    def __init__(self, winnings: int, action: str, client_cards: List[str], 
                 slumbot_cards: List[str], client_pos: int, board: List[str]):
        self.winnings = winnings
        self.action = action
        self.client_cards = client_cards
        self.slumbot_cards = slumbot_cards
        self.client_pos = client_pos
        self.board = board
        self.playerBuc = ''
        self.slumbotBuc = ''
    
    def __str__(self):
        pos_str = "BB" if self.client_pos == 0 else "SB"
        return (f"Winnings: {self.winnings} | Position: {pos_str} | "
                f"My Cards: {self.client_cards} | Slumbot Cards: {self.slumbot_cards} | "
                f"Board: {self.board} | Action: {self.action}")

class MismatchAction:
    def __init__(self, expected_actions: List[str], actual_actions: List[str], 
                action_history: str, action_key: str, 
                street: int = -1, hand_rank: str = ""):
        self.expected_actions = expected_actions
        self.actual_actions = actual_actions
        self.action_history = action_history
        self.action_key = action_key
        self.street = street
        self.hand_rank = hand_rank
    
    def __str__(self):
        street_name = STREET_NAMES[self.street] if 0 <= self.street < 4 else "unknown"
        return (f"Mismatch on {street_name} | "
                f"Key: '{self.action_key}' | "
                f"Hand rank: {self.hand_rank} | "
                f"Expected: {sorted(self.expected_actions)} | "
                f"Actual: {sorted(self.actual_actions)} | "
                f"Action history: {self.action_history}")

PREFLOP_BUCKETS = { # Set 'AA' to 10 (instead of 11) so that 11 buckets --> 10
  '22': 4, '33': 5, '44': 5, '55': 6, '66': 6, '77': 7, '88': 7, '99': 8, 'TT': 9, 'JJ': 9, 'QQ': 10, 'KK': 10, 'AA': 10, '23': 1, '24': 1, '25': 1, '26': 1, '27': 1, '28': 2, '29': 2, '2T': 2, '2J': 3, '2Q': 3, '2K': 4, '2A': 5, '34': 1, '35': 1, '36': 1, '37': 1, '38': 2, '39': 2, '3T': 3, '3J': 3, '3Q': 4, '3K': 4, '3A': 5, '45': 2, '46': 2, '47': 2, '48': 2, '49': 2, '4T': 3, '4J': 3, '4Q': 4, '4K': 4, '4A': 5, '56': 2, '57': 2, '58': 2, '59': 3, '5T': 3, '5J': 3, '5Q': 4, '5K': 5, '5A': 6, '67': 3, '68': 3, '69': 3, '6T': 3, '6J': 4, '6Q': 4, '6K': 5, '6A': 5, '78': 3, '79': 3, '7T': 4, '7J': 4, '7Q': 4, '7K': 5, '7A': 6, '89': 4, '8T': 4, '8J': 4, '8Q': 5, '8K': 5, '8A': 6, '9T': 4, '9J': 5, '9Q': 5, '9K': 5, '9A': 6, 'TJ': 5, 'TQ': 5, 'TK': 6, 'TA': 6, 'JQ': 5, 'JK': 6, 'JA': 7, 'QK': 6, 'QA': 7, 'KA': 7, '23s': 1, '24s': 2, '25s': 2, '26s': 2, '27s': 2, '28s': 2, '29s': 3, '2Ts': 3, '2Js': 4, '2Qs': 4, '2Ks': 5, '2As': 5, '34s': 2, '35s': 2, '36s': 2, '37s': 2, '38s': 2, '39s': 3, '3Ts': 3, '3Js': 4, '3Qs': 4, '3Ks': 5, '3As': 6, '45s': 2, '46s': 2, '47s': 3, '48s': 3, '49s': 3, '4Ts': 3, '4Js': 4, '4Qs': 4, '4Ks': 5, '4As': 6, '56s': 3, '57s': 3, '58s': 3, '59s': 3, '5Ts': 3, '5Js': 4, '5Qs': 5, '5Ks': 5, '5As': 6, '67s': 3, '68s': 3, '69s': 4, '6Ts': 4, '6Js': 4, '6Qs': 5, '6Ks': 5, '6As': 6, '78s': 4, '79s': 4, '7Ts': 4, '7Js': 4, '7Qs': 5, '7Ks': 5, '7As': 6, '89s': 4, '8Ts': 4, '8Js': 5, '8Qs': 5, '8Ks': 6, '8As': 6, '9Ts': 5, '9Js': 5, '9Qs': 5, '9Ks': 6, '9As': 6, 'TJs': 5, 'TQs': 6, 'TKs': 6, 'TAs': 7, 'JQs': 6, 'JKs': 6, 'JAs': 7, 'QKs': 6, 'QAs': 7, 'KAs': 7
}
NUM_STREETS = 4
NUM_ACTIONS = 5
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
LETTER2SUIT = {'s': '\u2660', 'h': '\033[91m\u2665\033[0m', 'd': '\033[91m\u2666\033[0m', 'c': '\u2663'}
CARD2NUM = { 'Ah': 1, '2h': 2, '3h': 3, '4h': 4, '5h': 5, '6h': 6, '7h': 7, '8h': 8, '9h': 9, 'Th': 10, 'Jh': 11, 'Qh': 12, 'Kh': 13, 
            'Ad': 14, '2d': 15, '3d': 16, '4d': 17, '5d': 18, '6d': 19, '7d': 20, '8d': 21, '9d': 22, 'Td': 23, 'Jd': 24, 'Qd': 25, 'Kd': 26, 
            'Ac': 27, '2c': 28, '3c': 29, '4c': 30, '5c': 31, '6c': 32, '7c': 33, '8c': 34, '9c': 35, 'Tc': 36, 'Jc': 37, 'Qc': 38, 'Kc': 39, 
            'As': 40, '2s': 41, '3s': 42, '4s': 43, '5s': 44, '6s': 45, '7s': 46, '8s': 47, '9s': 48, 'Ts': 49, 'Js': 50, 'Qs': 51, 'Ks': 52 }
NUM2ACTION = { 0: 'xm', 1: 'dm', 2: 'k', 3: 'c', 4: 'b', 5: 'b_', 6: 'f' }
MAX_RANKS = {0: 10, 1: 9, 2: 10, 3: 10}
DISPLAY_INFO = 1
CLIENT_TYPE = 2  # 0: random, 1: deep, 2: CFR
CLIENT_NAME = ['RANDOM', 'DEEP', 'CFR']
STREET_NAMES = ['preflop', 'flop', 'turn', 'river']
FIXPROBS01 = 0  # 0: use preflop probs conditioned on action history; 1: use pre-computed preflop probs


class InfoSetActionData:
    def __init__(self, initStratVal):
        self.strategy = initStratVal
        self.util = None
        self.cumulativeGain = initStratVal


class InfoSetData:
    def __init__(self):
        # initialize the strategy for the infoSet to be uniform random (e.g. k: 1/4, c: 1/4, b: 1/4, f: 1/4)
        self.actions: dict[str, InfoSetActionData] = {
            "k": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "c": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "b": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "f": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
            "r": InfoSetActionData(initStratVal=1 / NUM_ACTIONS),
        }
        self.beliefs: dict[str, float] = {}
        self.expectedUtil: float = None
        self.likelihood: float = None


host = 'slumbot.com'
infoSets01: dict[str, InfoSetData] = {}
# cfr_solution_preflopNflop = './db/cfr_texas_45k.pkl'
cfr_solution_preflopNflop = 'infoSets_PREFLOP_FLOP.pkl'
# cfr_solution_preflopNflop = './db/cfr_texas_300k.pkl'
if isfile(cfr_solution_preflopNflop):
    with open(cfr_solution_preflopNflop, 'rb') as f:
        infoSets01 = pickle.load(f)
        f.close()
else:
    print('Error! preflop+flop GTO solution not found!')
    sys.exit(-1)

infoSets0: dict[str, InfoSetData] = {}
cfr_solution_preflop = 'infoSets_PREFLOP_raise.pkl'
if isfile(cfr_solution_preflop):
    with open(cfr_solution_preflop, 'rb') as f:
        infoSets0 = pickle.load(f)
        f.close()
else:
    print('Error! preflop GTO solution not found!')
    sys.exit(-1)

infoSets1: dict[str, InfoSetData] = {}
cfr_solution_flop = 'infoSets_FLOP_raise.pkl'
if isfile(cfr_solution_flop):
    with open(cfr_solution_flop, 'rb') as f:
        infoSets1 = pickle.load(f)
        f.close()
else:
    print('Error! flop GTO solution not found!')
    sys.exit(-1)

infoSets2: dict[str, InfoSetData] = {}
cfr_solution_turn = 'infoSets_TURN_20b_raise.pkl'
if isfile(cfr_solution_turn):
    with open(cfr_solution_turn, 'rb') as f:
        infoSets2 = pickle.load(f)
        f.close()

infoSets3: dict[str, InfoSetData] = {}
cfr_solution_river = 'infoSets_RIVER_20b_raise.pkl'
if isfile(cfr_solution_river):
    with open(cfr_solution_river, 'rb') as f:
        infoSets3 = pickle.load(f)
        f.close()

infoSets23: dict[str, InfoSetData] = {}
cfr_solution_turn_river = 'infoSets_TURN_RIVER.pkl'

bckt5: dict[str, str] = {}
bckt5_filename = 'bckt5s.pkl'
if isfile(bckt5_filename):
    with open(bckt5_filename, 'rb') as f:
        bckt5 = pickle.load(f)
        f.close()
else:
    print('Error! bckt5.pkl does NOT exist')
    sys.exit(-1)

bckt6 = {}
bckt6_filename = 'bckt6s-twe.pkl'
if isfile(bckt6_filename):
    with open(bckt6_filename, 'rb') as f:
        bckt6 = pickle.load(f)
        f.close()
else:
    print('Error! buckets6.pkl does NOT exist')
    sys.exit(-1)

bckt7 = {}
bckt7_filename = 'bckt7s-twe.pkl'
if isfile(bckt7_filename):
    with open(bckt7_filename, 'rb') as f:
        bckt7 = pickle.load(f)
        f.close()
else:
    print('Error! buckets7.pkl does NOT exist')
    sys.exit(-1)

slumbot_strategy = {
    'preflop': {},
    'flop': {},
    'turn': {},
    'river': {}
}   

recorded_street = defaultdict(int)

def parse_action(action):
    """
    Returns a dict with information about the action passed in.
    Returns a key "error" if there was a problem parsing the action.
    pos is returned as -1 if the hand is over; otherwise the position of the player next to act.
    street_last_bet_to only counts chips bet on this street, total_last_bet_to counts all
      chips put into the pot.
    Handles action with or without a final '/'; e.g., "ck" or "ck/".
    """
    st = 0  # street indexer
    street_last_bet_to = BIG_BLIND
    total_last_bet_to = BIG_BLIND
    last_bet_size = BIG_BLIND - SMALL_BLIND
    last_bettor = 0
    sz = len(action)
    pos = 1
    if sz == 0:
        return {
            'st': st,
            'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
        }

    check_or_call_ends_street = False
    i = 0
    
    while i < sz:
        if st >= NUM_STREETS:
            return {'error': 'Unexpected error'}

        c = action[i]
        i += 1
        if c == 'k':  # action: check
            if last_bet_size > 0:
                return {'error': 'Illegal check <- last_best_size is greater than 0'}

            if check_or_call_ends_street:
                # After a check that ends a pre-river street, expect either a '/' or end of string.
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != '/':
                        return {'error': 'Missing slash'}
                    i += 1
                if st == NUM_STREETS - 1:
                    # Reached showdown
                    pos = -1
                else:
                    pos = 0
                    st += 1  # progress to the next street
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2  # advance to the next player
                check_or_call_ends_street = True
        elif c == 'c':  # action: call
            if last_bet_size == 0:
                return {'error': 'Illegal call <- last_bet_size is 0'}
            if total_last_bet_to == STACK_SIZE:
                # Call of an all-in bet
                # Either allow no slashes, or slashes terminating all streets prior to the river.
                if i != sz:
                    for st1 in range(st, NUM_STREETS - 1):
                        if i == sz:
                            return {'error': 'Missing slash (end of string)'}
                        else:
                            c = action[i]
                            i += 1
                            if c != '/':
                                return {'error': 'Missing slash'}
                if i != sz:
                    return {'error': 'Extra characters at end of an action (call)'}
                st = NUM_STREETS - 1
                pos = -1
                last_bet_size = 0
                return {
                    'st': st,
                    'pos': pos,
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor,
                }
            if check_or_call_ends_street:
                # After a call that ends a pre-river street, expect either a '/' or end of string.
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != '/':
                        return {'error': 'Missing slash after a street-ending call'}
                    i += 1
                if st == NUM_STREETS - 1:
                    # Reached showdown
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            last_bet_size = 0
            last_bettor = -1
        elif c == 'f':  # action: fold
            if last_bet_size == 0:
                return {'error', 'Illegal fold <- last_bet_size is 0'}
            if i != sz:
                return {'error': 'Extra characters at end of action'}
            pos = -1
            return {
                'st': st,
                'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }
        elif c == 'b':  # action: bet
            j = i
            while i < sz and '0' <= action[i] <= '9':
                i += 1
            if i == j:
                return {'error': 'Missing bet size'}
            try:
                new_street_last_bet_to = int(action[j:i])
            except (TypeError, ValueError):
                return {'error': 'Bet size not an integer'}
            new_last_bet_size = new_street_last_bet_to - street_last_bet_to

            # Validate that the bet is legal
            remaining = STACK_SIZE - total_last_bet_to
            if last_bet_size > 0:
                min_bet_size = last_bet_size
                # Make sure minimum opening bet is the size of the big blind.
                if min_bet_size < BIG_BLIND:
                    min_bet_size = BIG_BLIND
            else:
                min_bet_size = BIG_BLIND
            # Can always go all-in
            if min_bet_size > remaining:
                min_bet_size = remaining
            if new_last_bet_size < min_bet_size:
                return {'error': 'Bet too small'}
            max_bet_size = remaining
            if new_last_bet_size > max_bet_size:
                return {'error': 'Bet too big'}
            last_bet_size = new_last_bet_size
            street_last_bet_to = new_street_last_bet_to
            total_last_bet_to += last_bet_size
            last_bettor = pos
            pos = (pos + 1) % 2
            check_or_call_ends_street = True
        else:
            return {'error': 'Unexpected character in action'}

    return {
        'st': st,
        'pos': pos,
        'street_last_bet_to': street_last_bet_to,
        'total_last_bet_to': total_last_bet_to,
        'last_bet_size': last_bet_size,
        'last_bettor': last_bettor,
    }
# END of parse_action()


def new_hand(token):
    data = {}
    if token:
        data['token'] = token
    # Use verify=false to avoid SSL Error
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/new_hand', headers={}, json=data)
    success = getattr(response, 'status_code') == 200
    if not success:
        print('new_hand()::Status code: %s' % repr(response.status_code))
        try:
            print('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        sys.exit(-1)

    return r


def act(token, action):
    data = {'token': token, 'incr': action}
    # Use verify=false to avoid SSL Error
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    # response = requests.post(f'https://{host}/api/act', headers={}, json=data, verify=False)
    response = requests.post(f'https://{host}/api/act', headers={}, json=data)
    success = getattr(response, 'status_code') == 200
    if not success:
        print('act()::Status code: %s' % repr(response.status_code))
        print('act()::action: {}'.format(action))
        try:
            print('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        sys.exit(-1)

    # if 'error_msg' in r:
    #     print('Error: %s' % r['error_msg'])
    #     sys.exit(-1)

    return r



def enrich_cards(cards):
    return [('10' if s[0] == 'T' else s[0]) + LETTER2SUIT[s[1]] for s in cards]


def play_hand2(token, client_pos=0):  # client_pos = first_player
    global slumbot_strategy
    global recorded_street
    sign_num = 0
    act_num = 0
    slumbot_hand_ranks = ['Z']*4  # preflop-flop-turn-river
    slumbot_hole_cards = []
    slumbot_actions_by_street = [[], [], [], []]  # Track actions for each street
    current_street = -1  # -1: not started, 0: preflop, 1: flop, 2: turn, 3: river
    street_action_history = ['', '', '', '']  # Action history by street
    
    r = new_hand(token)
    new_token = r.get('token')
    if new_token:
        token = new_token
    player_n_bets = {0: 100, 1: 0} if client_pos == 0 else {0: 100, 1: 200}
    slumbot_pos = (client_pos + 1) % 2
    client_hand_ranks = ['Z']*4 # preflop-flop-turn-river
    client_hole_cards = []
    community_cards = []
    action_path_turn = None
    bot_cards = None
    fold = 0 # 0: not folded; 1: agent folded; -1: slumbot folded
    afold = 0
    betting_round = 999
    while True:
        action = r.get('action')
        action_ = sub(r'\d+', '', action)
        raise_action = parse_action_simplified(action)
        if 'f' in action_:
            fold = -1 if afold == 0 else 1
        action_ = action_[1:] if client_pos == 0 else 'b'+action_  # consider the first 2 steps
        hole_cards = r.get('hole_cards')
        bot_cards = r.get('bot_hole_cards')
        board = r.get('board')
        board_size = len(board)
        
        # Determine current betting round
        new_street = -1
        if board_size == 0:                
            new_street = 0  # preflop
        elif board_size == 3:
            new_street = 1  # flop
        elif board_size == 4:
            new_street = 2  # turn
        elif board_size == 5:
            new_street = 3  # river
            
        # Process street change
        if new_street > current_street:
            current_street = new_street
            player_n_bets = {0: 100, 1: 0} if client_pos == 0 else {0: 100, 1: 200}
            # We don't compute Slumbot's bucket here as hole cards may not be available yet
        
        
        winnings = r.get('winnings')
        if len(client_hole_cards) == 0:
            client_hole_cards = [CARD2NUM[hc] for hc in hole_cards]

        if board_size == 0:                
            betting_round = 0 # preflop
            if client_hand_ranks[0] == 'Z':
                client_hand_ranks[0] = NUM2RANK[calcHoleCardsRankNum(client_hole_cards)]
        elif board_size == 3:
            betting_round = 1 # flop
            if len(community_cards) != 3:
                community_cards = [CARD2NUM[board[_]] for _ in range(board_size)]
            if client_hand_ranks[1] == 'Z':
                client_hand_ranks[1] = bckt5[getDBKey(client_hole_cards, community_cards)]
                
                
        elif board_size == 4:
            betting_round = 2 # turn
            if len(community_cards) != 4:
                community_cards = [CARD2NUM[board[_]] for _ in range(board_size)]
            if client_hand_ranks[2] == 'Z':
                client_hand_ranks[2] = bckt6[getDBKey(client_hole_cards, community_cards)]
                
                
        elif board_size == 5:
            betting_round = 3 # river
            if len(community_cards) != 5:
                community_cards = [CARD2NUM[board[_]] for _ in range(board_size)]
            if client_hand_ranks[3] == 'Z':
                client_hand_ranks[3] = bckt7[getDBKey(client_hole_cards, community_cards)]
                
        else:
            print('Unexpected board size: %i' % board_size)
            sys.exit(-1)

        if winnings is not None:
            if not bot_cards:
                print('Slumbot cards not found!')
            else:
                slumbot_hole_cards = [CARD2NUM[hc] for hc in bot_cards]
                
                # Calculate all of Slumbot's bucket ranks as before
                # [existing bucket calculation code]
                # Calculate Slumbot's bucket ranks for each street
                if len(board) >= 0:  # Preflop
                    slumbot_hand_ranks[0] = NUM2RANK[calcHoleCardsRankNum(slumbot_hole_cards)]

                if len(board) >= 3:  # Flop
                    flop_cards = [CARD2NUM[board[i]] for i in range(3)]
                    slumbot_hand_ranks[1] = bckt5[getDBKey(slumbot_hole_cards, flop_cards)]
                    
                if len(board) >= 4:  # Turn
                    turn_cards = [CARD2NUM[board[i]] for i in range(4)]
                    slumbot_hand_ranks[2] = bckt6[getDBKey(slumbot_hole_cards, turn_cards)]
                    
                if len(board) == 5:  # River
                    river_cards = [CARD2NUM[board[i]] for i in range(5)]
                    slumbot_hand_ranks[3] = bckt7[getDBKey(slumbot_hole_cards, river_cards)]
                # Get the simplified action string with no amounts
                action_no_digits = remove_digits(action)
                
                # Split by street
                street_actions = action_no_digits.split('/')
                
                # For each street, extract Slumbot's actions based on position
                for street_idx, street_action in enumerate(street_actions):
                    if slumbot_hand_ranks[street_idx] == 'Z':
                        continue  # Skip if we don't have a valid rank for this street
                        
                    # Determine indices of Slumbot's actions based on position
                    slumbot_indices = []
                    
                    # Preflop: SB (pos 1) acts first, then alternates
                    # Postflop: BB (pos 0) acts first, then alternates
                    if street_idx == 0:  # Preflop
                        if client_pos == 0:  # Client is BB, Slumbot is SB
                            slumbot_indices = list(range(0, len(street_action), 2))  # 0, 2, 4...
                        else:  # Client is SB, Slumbot is BB
                            slumbot_indices = list(range(1, len(street_action), 2))  # 1, 3, 5...
                    else:  # Postflop
                        if client_pos == 0:  # Client is BB, acts first postflop
                            slumbot_indices = list(range(1, len(street_action), 2))  # 1, 3, 5...
                        else:  # Client is SB, Slumbot acts first postflop
                            slumbot_indices = list(range(0, len(street_action), 2))  # 0, 2, 4...
                    
                    # Extract Slumbot's actions
                    slumbot_actions = [street_action[i] for i in slumbot_indices if i < len(street_action)]
                    
                    # Build infoset string 
                    infoset_str = slumbot_hand_ranks[street_idx]
                    recorded_street[street_idx] += 1
                    
                    # Update the strategy dictionary
                    for i, action_now in enumerate(slumbot_actions):
                        # Get history of actions before this action
                        prior_actions = ""
                        if i > 0:
                            # Include your own previous actions and opponent's actions
                            for j in range(min(2*i, len(street_action))):
                                prior_actions += street_action[j]
                        
                        full_infoset = infoset_str + prior_actions
                        
                        # Initialize if needed
                        if full_infoset not in slumbot_strategy[STREET_NAMES[street_idx]]:
                            slumbot_strategy[STREET_NAMES[street_idx]][full_infoset] = {'k': 0, 'c': 0, 'b': 0, 'f': 0, 'r': 0}
                        
                        # Update action count
                        slumbot_strategy[STREET_NAMES[street_idx]][full_infoset][action_now] += 1
            if abs(winnings) >= 12000:
                print('[{}].{}.'.format(action, client_hand_ranks),end='')
            print(f"Winning: {winnings}")
            print(f"Client Position: {'Big Blind' if client_pos == 0 else 'Small Blind'}")
            print(f"Total Action String: {action}")
            print(f"Slumbot Hole Cards: {bot_cards}")
            print(f"Agent Hole Cards: {hole_cards}")
            print(f"Community Cards: {board}")
            print(f"Client Hand Ranks: {client_hand_ranks}")
            print(f"Slumbot Hand Ranks: {slumbot_hand_ranks}")
            if abs(winnings) >= 2000:
                significant_hand = SignificantHand(
                    winnings=winnings,
                    action=action,
                    client_cards=hole_cards,
                    slumbot_cards=bot_cards,
                    client_pos=client_pos,
                    board=board
                )
                return winnings, significant_hand, sign_num, fold, betting_round, False, None
            return winnings, None, sign_num, fold, betting_round, False, None
        a = parse_action(action)
        if 'error' in a:
            print('Error parsing action %s: %s' % (action, a['error']))
            sys.exit(-1)
        # action_1 = '' if action == '' else action[-1]
        # if a['last_bettor'] == -1:
        #     legal_actions = ['c', 'f', 'b', 'r']

        #     if len(action) >= 2 and slumbot_pos == 1 and action_1 == '/' and action[-2] == 'c'\
        #         or len(action) >= 3 and slumbot_pos == 0 and action[-3] == 'c':
        #         player_n_bets[slumbot_pos] = player_n_bets[client_pos]  # slumbot chose to call
        # elif a['last_bettor'] == client_pos:
        #     legal_actions = ['c', 'b', 'k', 'r']
        # elif a['last_bettor'] == slumbot_pos:  # Update slumbot's bet if it is the last bettor
        #     legal_actions = ['c', 'f', 'b', 'r']
        #     player_n_bets[slumbot_pos] = min(player_n_bets[slumbot_pos] + a['street_last_bet_to'], STACK_SIZE)
        #     if player_n_bets[slumbot_pos] >= STACK_SIZE:
        #         legal_actions = ['c', 'f']
        # else:
        #     print('Error parsing last bettor %s' % a['last_bettor'])
        #     sys.exit(-1)
        # # additional adjustments for legal actions
        # if len(action) == 0:  # UtG's 1st action right after the blinds
        #     legal_actions = ['f', 'b', 'c', 'r']
        # else:
        #     if action_1 == '/' and 'f' in legal_actions:
        #         legal_actions.remove('f')  # illegal fold after a street-ending call or check
        #     if len(action) >= 2 and action_1 == '/':
        #         if action[-2] == 'c' or action[-2] == 'k':
        #             legal_actions = ['b', 'k', 'r']
        #     if action_1 == 'k':
        #         if 'c' in legal_actions:
        #             legal_actions.remove('c')  # illegal call after a check
        #         if 'f' in legal_actions:
        #             legal_actions.remove('f')  # illegal fold after a check
        #         if 'k' not in legal_actions:
        #             legal_actions.append('k')

        min_bet_amount = max(player_n_bets.values()) * 2 - player_n_bets[client_pos]  # Double for bet
        min_raise_amount = max(player_n_bets.values()) * 3 - player_n_bets[client_pos]  # Triple for raise
        
        min_bet_clipped = min(min_bet_amount, STACK_SIZE - player_n_bets[client_pos])
        min_raise_clipped = min(min_raise_amount, STACK_SIZE - player_n_bets[client_pos])


        # my_action = getCFRAction(legal_actions, sub(r'\d+', '', action), client_pos, client_hand_ranks, betting_round) # randomly sample an action from all legal ones
        
        if betting_round == 0:
            action_ = raise_action.split('/')[-1]
        if betting_round == 1:
            action_ = raise_action.split('/')[-1]
        if betting_round == 2:
            action_ = raise_action.split('/')[-1]
        if betting_round == 3:
            action_ = raise_action.split('/')[-1]
        if action_ not in INFOSET_LEGAL_ACTIONS:
            return 0, None, sign_num, fold, betting_round, True, None
        legal_actions = INFOSET_LEGAL_ACTIONS[action_]
        # if set(legal_actions_sec) != set(legal_actions):
        #     print('Error: legal_actions_sec[{}] != legal_actions[{}]'.format(legal_actions_sec, legal_actions))
        #     print('action_ = {}'.format(action_))
        #     print('action = {}'.format(action))
        #     mismatch = MismatchAction(
        #         expected_actions=legal_actions_sec,
        #         actual_actions=legal_actions,
        #         action_history=action,
        #         action_key=action_,
        #         street=betting_round,
        #         hand_rank=client_hand_ranks[betting_round]
        #     )
        #     return 0, None, sign_num, fold, betting_round, False, mismatch

        my_action = choices(legal_actions)
        infoSetStrIsInvalid = False
        if betting_round == 0:
            infoSetStr = client_hand_ranks[0]+action_
            if infoSetStr in infoSets0:
                valid_actions = [la for la in legal_actions if infoSets0[infoSetStr].actions[la].strategy > 0.05]
                if not valid_actions:
                    print('No valid actions for infoSetStr[{}], {}, {}, {}, {} street'.format(infoSetStr, action, raise_action, action_, STREET_NAMES[betting_round]), end=' | ')
                    print(infoSets0[infoSetStr].actions['k'].strategy)
                    print(infoSets0[infoSetStr].actions['c'].strategy)
                    print(infoSets0[infoSetStr].actions['b'].strategy)
                    print(infoSets0[infoSetStr].actions['f'].strategy)
                    print(infoSets0[infoSetStr].actions['r'].strategy)
                    print(legal_actions)
                    sys.exit(-1)
                my_action = choices(valid_actions, [infoSets0[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                
                # if valid_actions:
                #     my_action = choices(valid_actions, [infoSets0[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                # else:
                #     my_action = choices(legal_actions, [infoSets0[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                
                # my_action = choices(legal_actions, [infoSets0[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                act_num += 1
                if infoSets0[infoSetStr].actions[my_action].strategy <= 0.1:
                    sign_num += 1
            else:
                infoSetStrIsInvalid = True
                print('Invalid infoSetStr[{}], {}, {} street'.format(infoSetStr, action, STREET_NAMES[betting_round]), end=' | ')
        elif betting_round == 1:            
            infoSetStr = client_hand_ranks[1]+action_
            if infoSetStr in infoSets1:
                valid_actions = [la for la in legal_actions if infoSets1[infoSetStr].actions[la].strategy > 0.05]
                if not valid_actions:
                    print('No valid actions for infoSetStr[{}], {}, {}, {}, {} street'.format(infoSetStr, action, raise_action, action_, STREET_NAMES[betting_round]), end=' | ')
                    print(infoSets1[infoSetStr].actions['k'].strategy)
                    print(infoSets1[infoSetStr].actions['c'].strategy)
                    print(infoSets1[infoSetStr].actions['b'].strategy)
                    print(infoSets1[infoSetStr].actions['f'].strategy)
                    print(infoSets1[infoSetStr].actions['r'].strategy)
                    sys.exit(-1)
                my_action = choices(valid_actions, [infoSets1[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                
                # if valid_actions:
                #     my_action = choices(valid_actions, [infoSets1[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                # else:
                #     my_action = choices(legal_actions, [infoSets1[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                    
                # my_action = choices(legal_actions, [infoSets1[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                act_num += 1
                if infoSets1[infoSetStr].actions[my_action].strategy <= 0.1:
                    sign_num += 1
            else:
                infoSetStrIsInvalid = True
                print('Invalid infoSetStr[{}], {}, {} street'.format(infoSetStr, action, STREET_NAMES[betting_round]), end=' | ')
        elif betting_round == 2:
            infoSetStr = client_hand_ranks[2]+action_
            if infoSetStr in infoSets2:
                valid_actions = [la for la in legal_actions if infoSets2[infoSetStr].actions[la].strategy > 0.05]
                if not valid_actions:
                    print('No valid actions for infoSetStr[{}], {}, {}, {}, {} street'.format(infoSetStr, action, raise_action, action_, STREET_NAMES[betting_round]), end=' | ')
                    print(infoSets2[infoSetStr].actions['k'].strategy)
                    print(infoSets2[infoSetStr].actions['c'].strategy)
                    print(infoSets2[infoSetStr].actions['b'].strategy)
                    print(infoSets2[infoSetStr].actions['f'].strategy)
                    print(infoSets2[infoSetStr].actions['r'].strategy)
                    print(legal_actions)
                    sys.exit(-1)
                my_action = choices(valid_actions, [infoSets2[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                
                # if valid_actions:
                #     my_action = choices(valid_actions, [infoSets2[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                # else:
                #     my_action = choices(legal_actions, [infoSets2[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                
                # my_action = choices(legal_actions, [infoSets2[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                act_num += 1
                if infoSets2[infoSetStr].actions[my_action].strategy <= 0.1:
                    sign_num += 1
            else:
                infoSetStrIsInvalid = True
                print('Invalid infoSetStr[{}], {}, {} street'.format(infoSetStr, action, STREET_NAMES[betting_round]), end=' | ')
        elif betting_round == 3:
            infoSetStr = client_hand_ranks[3]+action_
            if infoSetStr in infoSets3:
                valid_actions = [la for la in legal_actions if infoSets3[infoSetStr].actions[la].strategy > 0.05]
                if not valid_actions:
                    print('No valid actions for infoSetStr[{}], {}, {}, {}, {} street'.format(infoSetStr, action, raise_action, action_, STREET_NAMES[betting_round]), end=' | ')
                    print(infoSets3[infoSetStr].actions['k'].strategy)
                    print(infoSets3[infoSetStr].actions['c'].strategy)
                    print(infoSets3[infoSetStr].actions['b'].strategy)
                    print(infoSets3[infoSetStr].actions['f'].strategy)
                    print(infoSets3[infoSetStr].actions['r'].strategy)
                    print(legal_actions)
                    sys.exit(-1)
                my_action = choices(valid_actions, [infoSets3[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                
                # if valid_actions:
                #     my_action = choices(valid_actions, [infoSets3[infoSetStr].actions[la].strategy for la in valid_actions])[0]
                # else:
                #     my_action = choices(legal_actions, [infoSets3[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                
                # my_action = choices(legal_actions, [infoSets3[infoSetStr].actions[la].strategy for la in legal_actions])[0]
                act_num += 1
                if infoSets3[infoSetStr].actions[my_action].strategy <= 0.1:
                    sign_num += 1
            else:
                infoSetStrIsInvalid = True
                print('Invalid infoSetStr[{}], {}, {} street'.format(infoSetStr, action, STREET_NAMES[betting_round]), end=' | ')
        if my_action == 'f':
            afold = 1
        if client_hand_ranks[betting_round] != 'Z':
            client_hand_rank = RANK2NUM[client_hand_ranks[betting_round]]
            max_rank = MAX_RANKS[betting_round]
            max_bet = max(max(player_n_bets.values()), min_bet_clipped)
            client_hand_is_weak = False
            if client_hand_rank < max_rank and max_bet > 6400\
                or client_hand_rank < max_rank - 1 and 3200 < max_bet <= 6400:
                # or client_hand_rank < max_rank - 2 and 1600 < max_bet <= 3200:
                client_hand_is_weak = True
            
            # if client_hand_is_weak: 
            #     if 'c' in legal_actions:
            #         my_action = 'c'
            #     if 'k' in legal_actions:
            #         my_action = 'k'
            #     if 'f' in legal_actions:
            #         my_action = 'f'

            #     if betting_round == 0 and client_pos == 1 and infoSetStr.count('b') >= 5: # do not three bet if your hand is too weak
            #         if 'f' in legal_actions:
            #             my_action = 'f'

            if infoSetStrIsInvalid:
                if client_hand_is_weak:
                    if 'f' in legal_actions:
                        my_action = 'f'
                elif len(legal_actions) >= 2 and my_action == 'f' and 'f' in legal_actions:
                    legal_actions_ = [la for la in legal_actions]
                    legal_actions_.remove('f')
                    my_action = choice(legal_actions_)

        if my_action[0] == 'b':
            my_action = 'b'  # ban all in -> bet size too big + causes wide spreads
            print(f"Player bets: {player_n_bets} | Street: {STREET_NAMES[betting_round]} | minbet: {min_bet_clipped}")
            player_n_bets[client_pos] += min_bet_clipped  # raiseX2 ('b')
            my_action += str(min_bet_clipped)
        if my_action[0] == 'r':
            my_action = 'b'  # ban all in -> bet size too big + causes wide spreads
            player_n_bets[client_pos] += min_raise_clipped  # raiseX2 ('b')
            my_action += str(min_raise_clipped)

        if len(my_action) and my_action[0] == 'c':  # call
            player_n_bets[client_pos] = player_n_bets[slumbot_pos]
            
        # if DISPLAY_INFO:
        #     print('Client\'s action: %s | legal action(s): %s | player&bets: %s | Client\' position: %i' % (
        #         my_action, str(legal_actions), str(player_n_bets), client_pos))
            
        r = act(token, my_action)
        delta_bet = 0
        prev = ''
        while 'error_msg' in r:
            print('Error message: %s' % r['error_msg'])
            if r['error_msg'] == 'Bet size too big':
                delta_bet -= 100
                if prev == 'small':
                    print('Invalid action: %s' % my_action)
                    print(int(my_action[1:]) if my_action[1:] else 0)
                    print(delta_bet)
                    return 0, None, sign_num, fold, betting_round, True, None
                prev = 'big'
            elif r['error_msg'] == 'Bet size too small':
                delta_bet += 100
                bet_amount = int(my_action[1:]) if my_action[1:] else 0
                print(f'bet_amount: {bet_amount} | delta_bet: {delta_bet}')
                if delta_bet + bet_amount > 20000:
                    print("exceed limit")
                    delta_bet -= 100
                
                if prev == 'big':
                    print('Invalid action: %s' % my_action)
                    print(int(my_action[1:]) if my_action[1:] else 0)
                    print(delta_bet)
                    return 0, None, sign_num, fold, betting_round, True, None
                prev = 'small'
            elif r['error_msg'] == 'Illegal check':
                print('Invalid action: %s' % my_action)
            elif r['error_msg'] == 'Illegal bet':
                print('Invalid bet: %s' % my_action)
                print('actionstr: %s' % action_)
                r = act(token, 'c')
                break
            bet_amount = int(my_action[1:]) if my_action[1:] else 0
            my_action = 'b' + str(min(20000, bet_amount + delta_bet))
            r = act(token, my_action)
# END


def login(username, password):
    data = {"username": username, "password": password}
    # If porting this code to another language, make sure that the Content-Type header is
    # set to application/json.
    response = requests.post(f'https://{host}/api/login', json=data)
    success = getattr(response, 'status_code') == 200
    if not success:
        print('Status code: %s' % repr(response.status_code))
        try:
            print('Error response: %s' % repr(response.json()))
        except ValueError:
            pass
        sys.exit(-1)

    try:
        r = response.json()
    except ValueError:
        print('Could not get JSON from response')
        sys.exit(-1)

    if 'error_msg' in r:
        print('Error: %s' % r['error_msg'])
        sys.exit(-1)
        
    token = r.get('token')
    if not token:
        print('Did not get token in response to /api/login')
        sys.exit(-1)
    return token


def plot_winnings_and_bb100(num_hands_list, winnings_list):
    """
    Plot winnings and BB/100 curves
    
    Args:
        num_hands_list: List of number of hands played
        winnings_list: List of cumulative winnings
    """
    # Calculate BB/100 for each point
    bb100_list = [w / BIG_BLIND / n * 100 for n, w in zip(num_hands_list, winnings_list)]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot winnings
    ax1.plot(num_hands_list, winnings_list, 'b-', marker='o')
    ax1.set_xlabel('Number of Hands')
    ax1.set_ylabel('Cumulative Winnings')
    ax1.set_title('Winnings vs Number of Hands')
    ax1.grid(True)
    
    # Plot BB/100
    ax2.plot(num_hands_list, bb100_list, 'r-', marker='o')
    ax2.set_xlabel('Number of Hands')
    ax2.set_ylabel('BB/100')
    ax2.set_title('BB/100 vs Number of Hands')
    ax2.grid(True)
    plt.savefig('winnings_bb100.png')
    plt.tight_layout()
    plt.show()


def tuple2str(hand_list): 
    """
    Args:
        hand_list (list): A list of card strings, e.g., ['Ah', 'Ks', '2d', '3c', '4h']
    
    Returns:
        str: A string representation of the cards with hole cards and board cards sorted separately.
             Additionally, if in the hole cards (first two) or board cards (last cards) two cards have the
             same rank and one is marked as suited, the offsuited card is placed before the suited one.
    """
    # Define rank ordering
    rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
                  '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    
    # Split into hole cards and board cards (assume hand_list length is 5, 6, or 7)
    hole_cards = hand_list[:2]
    board_cards = hand_list[2:] if len(hand_list) > 2 else []
    
    # Sort hole cards and board cards by rank (low-to-high here; you can reverse if desired)
    hole_cards = sorted(hole_cards, key=lambda x: rank_order[x[0]])
    board_cards = sorted(board_cards, key=lambda x: rank_order[x[0]])
    
    # Combine back – note: we keep them separated in our later processing.
    sorted_hand = hole_cards + board_cards
    
    # Extract ranks and suits from the combined list
    ranks = []
    suits = []
    for card in sorted_hand:
        rank, suit = card[0], card[1]
        ranks.append(rank)
        suits.append(suit)
    
    # Count suit frequencies over the entire hand
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    # Determine the dominant suit (if any)
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    max_suit = max(suit_counts.items(), key=lambda x: x[1])[0] if suit_counts else None
    
    # Use a threshold based on hand size: if at least threshold cards are in the dominant suit,
    # we mark cards: those whose suit equals max_suit get 's', others get 'o'
    threshold = 3 if len(hand_list) == 5 else (4 if len(hand_list) == 6 else 5)
    if max_suit_count >= threshold:
        result = []
        for i in range(len(ranks)):
            if suits[i] == max_suit:
                result.append(f"{ranks[i]}s")
            else:
                result.append(f"{ranks[i]}o")
        
        # Now, separately process the hole and board parts.
        hole_result = result[:len(hole_cards)]
        board_result = result[len(hole_cards):]
        
        # Re-sort each subgroup by: 
        # (1) rank (using the rank ordering), and if equal, then 
        # (2) card marker, placing offsuited ('o') before suited ('s').
        sort_key = lambda card: (rank_order[card[0]], 0 if card.endswith('o') else 1)
        hole_result = sorted(hole_result, key=sort_key)
        board_result = sorted(board_result, key=sort_key)
        
        # Combine the processed parts and return the final string.
        return ''.join(hole_result + board_result)
    else:
        # Otherwise, simply return the concatenated ranks (sorted by rank)
        simple_sorted = sorted(ranks, key=lambda r: rank_order[r])
        return ''.join(simple_sorted)
def hole2str(hole_cards):
    """
    Converts a pair of cards into a standardized string representation.
    
    Args:
        hole_cards (list): A list of 2 card strings, e.g., ['Ah', 'Ks']
    
    Returns:
        str: A string representation of the hole cards.
            - If the cards form a pair, returns a 2-character string of their ranks, e.g., 'AA'
            - If the cards are suited, returns the sorted ranks followed by 's', e.g., 'AKs'
            - If the cards are offsuit, returns the sorted ranks, e.g., 'AK'
    """
    # Extract ranks and suits
    rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
    rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
    

    
    if rank1 == rank2:
        # Pair: return rank twice
        return rank1 * 2
    else:
        # Sort ranks by poker ranking
        rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, 
                    '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        
        # Sort ranks high to low
        if rank_order[rank1] < rank_order[rank2]:
            sorted_ranks = rank1 + rank2
        else:
            sorted_ranks = rank2 + rank1
        
        # Add 's' suffix if suited
        if suit1 == suit2:
            return sorted_ranks + 's'
        else:
            return sorted_ranks

def remove_digits(betting_sequence):
    """
    Remove all digits from a betting sequence string.
        
    Example:
        >>> remove_digits("b200b300c/kb100c/b100b800c/kb1200c")
        "bbc/kbc/bbc/kbc"
    """
    result = ""
    for char in betting_sequence:
        if not char.isdigit():
            result += char
    
    return result

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

# def analyze_significant_hands_with_buckets(significanthands):
#     """
#     Analyze significant hands by extracting buckets and creating infoset strings.
#     """
#     print("\n=== Analyzing Significant Hands with Buckets ===")
    
#     # Count how many hands we process for each street
#     street_counts = {"turn": 0, "river": 0}
    
#     # Store infoset strings
#     turn_infosets = []
#     river_infosets = []
    
#     bucket_counts = {"turn": {}, "river": {}}
#     decided = 0
#     totwin = 0
#     totsignwin = 0
#     fold = 0
#     TURN_BUCKET = load_pickled_data('buckets6.pkl')
#     RIVER_BUCKET = load_pickled_data('buckets7.pkl')
#     infosets_turn = load_pickled_data('infoSets_TURN.pkl')
#     infosets_river = load_pickled_data('infoSets_RIVER.pkl')
        
#     for i, hand in enumerate(significanthands):
#         client_cards = hand.client_cards
#         board = hand.board
#         actions = hand.action       
#         action_streets = actions.split('/')
#         winnings = hand.winnings
#         totwin += winnings
#         pureaction = remove_digits(actions).split('/')[-1]
#         if len(pureaction) == 0:
#             continue    
#         if hand.client_pos == 0:
#             if len(pureaction) % 2 == 0:
#                 mainaction = pureaction[-2]
#                 base = pureaction[:-2]
#             else:
#                 mainaction = pureaction[-1]
#                 base = pureaction[:-1]
#         else:
#             if len(pureaction) % 2 == 0:
#                 mainaction = pureaction[-1]
#                 base = pureaction[:-1]
#             else:
#                 mainaction = pureaction[-2]
#                 base = pureaction[:-2]
#         if len(action_streets) == 4:  # Four streets means river (7-card)
#             river_hand = client_cards + board
#             hand_key = tuple2str(river_hand)  # You'll need to implement this function
            
#             if hand_key in RIVER_BUCKET:
#                 bucket = RIVER_BUCKET[hand_key]                
#                 if bucket in bucket_counts["river"]:
#                     bucket_counts["river"][bucket] += 1
#                 else:
#                     bucket_counts["river"][bucket] = 1
                
#                 river_actions = '/'.join(action_streets[:-1])  # All actions except river
#                 infoset_str = bucket + base
#                 infoset = infosets_river[infoset_str]
#                 if infoset.actions[mainaction].strategy < 0.09:
#                     decided += 1
#                     totsignwin += winnings
#                     if infoset.actions['f'].strategy >= 0.8:
#                         fold += 1
                
#             else:
#                 print(f"Warning: Hand {i} not found in RIVER_BUCKET")                
#         elif len(action_streets) == 3:  # Three streets means turn (6-card)
#             turn_hand = client_cards + board[:4]  # First 4 board cards            
#             hand_key = tuple2str(turn_hand)  # You'll need to implement this function
            
#             if hand_key in TURN_BUCKET:
#                 bucket = TURN_BUCKET[hand_key]                
#                 if bucket in bucket_counts["turn"]:
#                     bucket_counts["turn"][bucket] += 1
#                 else:
#                     bucket_counts["turn"][bucket] = 1                
#                 turn_actions = '/'.join(action_streets[:-1])  # All actions except turn
#                 infoset_str = bucket + base
#                 infoset = infosets_turn[infoset_str]
#                 if infoset.actions[mainaction].strategy < 0.09:
#                     decided += 1
#                     totsignwin += winnings

                
#             else:
#                 print(f"Warning: Hand {i} not found in TURN_BUCKET")
#         else:
#             print(f"Warning: Hand {i} has unusual number of action streets: {len(action_streets)}")
#     print(f"Decided: {decided} out of {len(significanthands)}")
#     print(f"Folded: {fold}")
#     print(f"Total winnings: {totwin}")
#     print(f"Total significant winnings: {totsignwin}")

def main():
    winnings = 0
    undecided = 0
    foldnum_a = 0
    foldnum_s = 0
    skipped = 0
    this_skip = False
    significant_hands: List[SignificantHand] = []
    unmatched_legal_actions: List[Tuple[List[str], List[str]]] = []
    unmatched_actions = []
    prev_res = []
    bettingr = 999
    round_stats = {
        'preflop': {'count': 0, 'winnings': 0, 'win': 0},
        'flop': {'count': 0, 'winnings': 0, 'win': 0},
        'turn': {'count': 0, 'winnings': 0, 'win': 0},
        'river': {'count': 0, 'winnings': 0, 'win': 0}
    }
    with open('./winnings.txt', 'a+') as file:
        for line in file:
            if ': ' in line:
                prev_res.append(int(line.split(': ')[-1]))
        file.close()
    if len(prev_res):
        print('#Hands: {} | bb/100: {:.0f} | total: {}'.format(len(prev_res), sum(prev_res)/len(prev_res), sum(prev_res)))
    
    parser = argparse.ArgumentParser(description='Texas Agents vs Slumbot API')
    parser.add_argument('--num_hands', type=int, default=100)

    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    username = ''
    password = ''
    if username and password:
        token = login(username, password)
    else:
        token = None

    # To avoid SSLError:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    num_hands = args.num_hands
    num_workers = args.num_workers
    num_hands_track = []
    winnings_track = []
    mis_legal_actions = None
    if CLIENT_TYPE == 2:
        # t0 = time()
        if num_workers == 0:
            for h in range(num_hands):
                print('current winnings: %i' % winnings)
                print('>>>Starting hand #%i/%i...' % ((h + 1), num_hands), end='')
                result = play_hand2(token, client_pos=(h % 2))
                if isinstance(result, tuple):
                    hand_winnings, sig_hand, sign_num, folds, bettingr, this_skip, mis_legal_actions = result
                    winnings += hand_winnings
                    undecided += sign_num
                    if folds == 1:
                        foldnum_a += 1
                    if folds == -1:
                        foldnum_s += 1
                    if sig_hand:
                        significant_hands.append(sig_hand)
                    if this_skip:
                        skipped += 1
                    if bettingr == 0:
                        round_stats['preflop']['count'] += 1
                        round_stats['preflop']['winnings'] += hand_winnings
                        if hand_winnings > 0:
                            round_stats['preflop']['win'] += 1
                    elif bettingr == 1:
                        round_stats['flop']['count'] += 1
                        round_stats['flop']['winnings'] += hand_winnings
                        if hand_winnings > 0:
                            round_stats['flop']['win'] += 1
                    elif bettingr == 2:
                        round_stats['turn']['count'] += 1
                        round_stats['turn']['winnings'] += hand_winnings
                        if hand_winnings > 0:
                            round_stats['turn']['win'] += 1
                    elif bettingr == 3:
                        round_stats['river']['count'] += 1
                        round_stats['river']['winnings'] += hand_winnings
                        if hand_winnings > 0:
                            round_stats['river']['win'] += 1
                    if mis_legal_actions:
                        unmatched_actions.append(mis_legal_actions)
                else:
                    winnings += result
                
                # Track progress every N hands
                if (h + 1) % 1 == 0:
                    num_hands_track.append(h + 1)
                    winnings_track.append(winnings)


    bb100 = round(winnings / BIG_BLIND / num_hands * 100)
    print('*{}* agent vs Slumbot | games: {} | BB/100: {} | total: {} '.format(
        CLIENT_NAME[CLIENT_TYPE], num_hands,
        ('\033[91m' if bb100 < 0 else '\033[92m') + str(bb100) + '\033[0m', winnings))
    # ^^ the standard measure is milli-big-blinds per hand (or per game), or mbb/g, 
    # where one milli-big-blind is 1/1000 of one big blind
    plot_winnings_and_bb100(num_hands_track, winnings_track)
    
    preflop_prob = defaultdict(float)
    flop_prob = defaultdict(float)
    turn_prob = defaultdict(float)
    river_prob = defaultdict(float)
    
    preflop_occur = defaultdict(int)
    flop_occur = defaultdict(int)
    turn_occur = defaultdict(int)
    river_occur = defaultdict(int)
    
    slum_preflop_prob = defaultdict(float)
    slum_flop_prob = defaultdict(float)
    slum_turn_prob = defaultdict(float)
    slum_river_prob = defaultdict(float)
    
    slum_preflop_occur = defaultdict(int)
    slum_flop_occur = defaultdict(int)
    slum_turn_occur = defaultdict(int)
    slum_river_occur = defaultdict(int)
        
    if significant_hands:
        print("\nSignificant Hands (|winnings| > 2000):")
        print("-" * 80)
        for hand in significant_hands:
            hole_player = hand.client_cards
            hole_slumbot = hand.slumbot_cards
            board = hand.board
            flop_cards = board[:3]
            turn_card = board[3] if len(board) >= 4 else None
            river_card = board[4] if len(board) == 5 else None
            player_preflop_buck = PREFLOP_BUCKETS[hole2str(hole_player)]
            player_flop_buck = bckt5[tuple2str(hole_player+flop_cards)]
            playerBuc = str(player_preflop_buck) + str(player_flop_buck)
            if turn_card is not None:  # Check if turn card exists
                player_turn_buck = bckt6[tuple2str(hole_player+flop_cards+[turn_card])]
                playerBuc += str(player_turn_buck)
            else:
                player_turn_buck = "N/A"
            if river_card is not None:  # Check if river card exists
                player_river_buck = bckt7[tuple2str(hole_player+flop_cards+[turn_card, river_card])]
                playerBuc += str(player_river_buck)
            else:
                player_river_buck = "N/A"  # Or some appropriate default value
            hand.playerBuc = playerBuc
            slumbot_preflop_buck = PREFLOP_BUCKETS[hole2str(hole_slumbot)]
            slumbot_flop_buck = bckt5[tuple2str(hole_slumbot+flop_cards)]
            slumbotBuc = str(slumbot_preflop_buck) + str(slumbot_flop_buck)
            if turn_card is not None:
                slumbot_turn_buck = bckt6[tuple2str(hole_slumbot+flop_cards+[turn_card])]
                slumbotBuc += str(slumbot_turn_buck)
            else:
                slumbot_turn_buck = "N/A"
            if river_card is not None:  # Check if river card exists
                slumbot_river_buck = bckt7[tuple2str(hole_slumbot+flop_cards+[turn_card, river_card])]
                slumbotBuc += str(slumbot_river_buck)
            else:
                slumbot_river_buck = "N/A"  # Or appropriate default
            hand.slumbotBuc = slumbotBuc
            preflop_occur[player_preflop_buck] += 1
            flop_occur[player_flop_buck] += 1
            turn_occur[player_turn_buck] += 1
            if river_card is not None:
                river_occur[player_river_buck] += 1
            slum_preflop_occur[slumbot_preflop_buck] += 1
            slum_flop_occur[slumbot_flop_buck] += 1
            slum_turn_occur[slumbot_turn_buck] += 1
            if river_card is not None:
                slum_river_occur[slumbot_river_buck] += 1
                # Calculate probabilities
        total_hands = len(significant_hands)
        
        if significant_hands:
            with open('significanthands.pkl', 'wb') as f:
                pickle.dump(significant_hands, f)
                print(f"\nSaved {len(significant_hands)} significant hands to significanthands.pkl")
        
        # Calculate player probabilities
        print("\nPlayer Bucket Probabilities:")
        print("-" * 40)
        
        print("Preflop Buckets:")
        for bucket, count in preflop_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        print("\nFlop Buckets:")
        for bucket, count in flop_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        print("\nTurn Buckets:")
        for bucket, count in turn_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        print("\nRiver Buckets:")
        for bucket, count in river_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        # Calculate Slumbot probabilities
        print("\nSlumbot Bucket Probabilities:")
        print("-" * 40)
        
        print("Preflop Buckets:")
        for bucket, count in slum_preflop_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        print("\nFlop Buckets:")
        for bucket, count in slum_flop_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        print("\nTurn Buckets:")
        for bucket, count in slum_turn_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
            
        print("\nRiver Buckets:")
        for bucket, count in slum_river_occur.items():
            prob = count / total_hands
            print(f"Bucket {bucket}: {count}/{total_hands} = {prob:.3f}")
    else:
        print("no significant hands")
    print('Undecided: %i' % undecided)
    # analyze_significant_hands_with_buckets(significant_hands)
    print('Folded agent: %i' % foldnum_a)
    print('Folded slumbot: %i' % foldnum_s)
    
    # Display round statistics
    print("\n=== Round Statistics ===")
    print("-" * 60)
    print(f"{'Round':<10} | {'Count':<10} | {'Win Rate':<10} | {'Total Winnings':<15} | {'Avg. Winnings':<15}")
    print("-" * 60)
    
    for round_name, stats in round_stats.items():
        count = stats['count']
        round_winnings = stats['winnings']
        win_rate = f"{(round_winnings / BIG_BLIND / count * 100):.2f}" if count > 0 else "N/A"
        avg_winnings = f"{(round_winnings / count):.2f}" if count > 0 else "N/A"
        
        print(f"{round_name:<10} | {count:<10} | {win_rate:<10} | {round_winnings:<15} | {avg_winnings:<15}")

        
    # After all hands are played, analyze and display Slumbot's strategy by street
    print("\n=== Slumbot Strategy Analysis ===")
    
    # Save the strategy to a file
    with open('slumbot_strategy_by_street.pkl', 'wb') as f:
        pickle.dump(slumbot_strategy, f)
    print("Saved Slumbot's observed strategy to slumbot_strategy_by_street.pkl")
    
    # Display analysis for each street
    for street in ['preflop', 'flop', 'turn', 'river']:
        street_data = slumbot_strategy[street]
        if street_data:
            print(f"\n{street.upper()} STRATEGY:")
            print(f"Total unique {street} infosets encountered: {len(street_data)}")
            
            # Count action frequencies for this street
            action_counts = {'k': 0, 'c': 0, 'b': 0, 'f': 0, 'r': 0}
            for infoset, actions in street_data.items():
                for action, count in actions.items():
                    action_counts[action] += count
            
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                print(f"Action distribution for {street}:")
                for action, count in action_counts.items():
                    percentage = (count / total_actions) * 100
                    print(f"  {action}: {count} ({percentage:.1f}%)")
            
            # Print counts for all infosets
            print(f"\nDetailed infoset actions for {street}:")
            print("-" * 80)
            
            # Sort infosets alphabetically for easier reading
            sorted_infosets = sorted(street_data.items())
            
            # Print all infosets in the requested format
            for infoset, actions in sorted_infosets:
                action_str = ""
                for action, count in actions.items():
                    if count > 0:
                        action_str += f"{action}:{count}, "
                action_str = action_str.rstrip(", ")
                print(f"infoset: {infoset}, {action_str}")
    print(recorded_street)
    print('Skipped: %i' % skipped)
    # Replace the existing code for printing mismatches around line 1562
    # if unmatched_actions:
    #     print("\n=== Unmatched Legal Actions ===")
    #     print(f"Found {len(unmatched_actions)} instances where legal actions didn't match expected actions")
    #     print("-" * 100)
        
    #     # Group mismatches by street
    #     by_street = defaultdict(list)
    #     for mismatch in unmatched_actions:
    #         street_name = STREET_NAMES[mismatch.street] if 0 <= mismatch.street < 4 else "unknown"
    #         by_street[street_name].append(mismatch)
        
    #     # Print summary by street
    #     for street, mismatches in by_street.items():
    #         print(f"\n{street.upper()} MISMATCHES: {len(mismatches)}")
    #         for mismatch in mismatches:
    #             print(f"  Key: '{mismatch.action_key}' | Expected: {sorted(mismatch.expected_actions)} | "
    #                 f"Actual: {sorted(mismatch.actual_actions)} | Hand rank: {mismatch.hand_rank} | "
    #                 f"History: '{mismatch.action_history}'")
    # else:
    #     print("No unmatched legal actions found.")

def parse_action_simplified(action):
    """
    Parse an action string into a simplified format.
    - Split action by '/' to separate streets
    - For bets ('b'):
      - If it's the first bet in its street and amount > 300, replace with 'r'
      - If it's not first bet and amount > 3x previous bet, replace with 'r'
    - Remove all numbers from the result

    Args:
        action (str): Original action string with bet amounts (e.g., "b200c/kb400")
        
    Returns:
        str: Simplified action string with 'b' or 'r' and no numbers
    """
    if not action:
        return ""
    
    # Split by '/' to process each street separately
    streets = action.split('/')
    simplified_streets = []
    
    for street in streets:
        # Extract bet actions and amounts
        i = 0
        simplified_street = ""
        prev_bet_amount = 100  # Start with big blind as reference
        
        while i < len(street):
            if street[i] == 'b':
                # Extract bet amount
                i += 1
                amount_start = i
                while i < len(street) and street[i].isdigit():
                    i += 1
                
                if amount_start < i:  # We found digits after 'b'
                    bet_amount = int(street[amount_start:i])
                    
                    # Check if this is the first bet on this street
                    is_first_bet = 'b' not in simplified_street
                    
                    # Determine if this is a raise ('r') or bet ('b')
                    if is_first_bet and bet_amount > 300:
                        simplified_street += 'r'
                    elif not is_first_bet and bet_amount > 3 * prev_bet_amount:
                        simplified_street += 'r'
                    else:
                        simplified_street += 'b'
                    
                    prev_bet_amount = bet_amount
                else:
                    simplified_street += 'b'  # Just a 'b' with no amount
            else:
                simplified_street += street[i]
                i += 1
        
        simplified_streets.append(simplified_street)
    
    # Join streets back with '/'
    return '/'.join(simplified_streets)

if __name__ == '__main__':
    main()