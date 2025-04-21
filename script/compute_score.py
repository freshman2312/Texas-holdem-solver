import numpy as np
# import sys
from copy import deepcopy
# from numba import jit, njit, typed, types


card2rank = {1: 14, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13,
             14: 14, 15: 2, 16: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
             27: 14, 28: 2, 29: 3, 30: 4, 31: 5, 32: 6, 33: 7, 34: 8, 35: 9, 36: 10, 37: 11, 38: 12, 39: 13,
             40: 14, 41: 2, 42: 3, 43: 4, 44: 5, 45: 6, 46: 7, 47: 8, 48: 9, 49: 10, 50: 11, 51: 12, 52: 13}

card2row = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
            14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1,
            27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2,
            40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3}

weights = np.array([537824, 38416, 2744, 196, 14, 1])  # hand_tier, main_rank, kicker1, ..., kicker4, 1+5=6 entries

hand_names = ['high card', 'pair', 'two pairs', 'three of a kind',
              'straight', 'flush', 'full house', 'four of a kind', 'straight flush']

row2symbol = {0: '\033[31m\u2665\033[0m', 1: '\033[31m\u2666\033[0m', 2: '\u2663', 3: '\u2660'}
rank2str = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
str2rank = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14}


# @njit
# def card2rank_f(c):
#     c13 = c%13
#     return c13+13 if c13 <= 1 else c13


# @njit
# def card2row_f(c):
#     return (c-1)//13


# # @jit(forceobj=True)
# @njit
# def compute_score0(all_cards):
#     num_cards = len(all_cards)

#     # Prep work
#     # rank_and_cards = {}
#     # suit_and_cards = {}

#     rank_and_cards = typed.Dict.empty(
#         key_type=types.int64,
#         value_type=types.int64[:],
#     )
#     suit_and_cards = typed.Dict.empty(
#         key_type=types.int64,
#         value_type=types.int64[:],
#     )

#     for c in all_cards:
#         # c_rank = card2rank[c]
#         c_rank = card2rank_f(c)
#         if c_rank in rank_and_cards:
#             rank_and_cards[c_rank] += [c]
#         else:
#             rank_and_cards[c_rank] = [c]

#         # c_suit = card2row[c]
#         c_suit = card2row_f(c)
#         if c_suit in suit_and_cards:
#             suit_and_cards[c_suit] += [c]
#         else:
#             suit_and_cards[c_suit] = [c]

#     # High card, t0
#     card_ranks = list(rank_and_cards.keys())
#     card_ranks.sort()
#     num_ranks = len(card_ranks)
#     # score = [0, 0, 0, 0, 0, 0]
#     # score = np.zeros(6,dtype=np.int8)
#     score = np.zeros(6)
#     score[1] = card_ranks[-1]
#     max_single1 = score[1]

#     if num_cards == 2:
#         # A pair, t1
#         if num_ranks == 1:
#             score[0] = 1
#         else:
#             score[2] = card_ranks[-2]
#     else:  # when `num_cards` is in [5, 6, 7]
#         pair_ranks = []
#         triple_ranks = []
#         quad_ranks = []
#         for _ in card_ranks:  # sorted
#             len_ = len(rank_and_cards[_])
#             if len_ == 2:
#                 pair_ranks.append(_)
#             elif len_ == 3:
#                 triple_ranks.append(_)
#             elif len_ == 4:
#                 quad_ranks.append(_)

#         # Generate `straights`
#         seq_records = list()
#         start = i = 0
#         longest = 1
#         while i < num_ranks:
#             if i + 1 < num_ranks and card_ranks[i + 1] - card_ranks[i] == 1:
#                 longest += 1
#                 i += 1
#             else:
#                 seq_records.append((start, longest))
#                 i += 1
#                 start = i
#                 longest = 1

#         straight_ranks = []
#         # treat aces as 1
#         if 14 in card_ranks:
#             if card_ranks[seq_records[0][0]] == 2 and seq_records[0][1] >= 4:
#                 straight_ranks.append(1)

#         for seq in seq_records:
#             if seq[1] >= 5:
#                 straight_ranks.append(card_ranks[seq[0]])

#         # Pair(s)
#         num_pairs = len(pair_ranks)
#         max_pair1 = 0
#         if num_pairs > 0:
#             max_pair1 = pair_ranks[-1]
#             score[1] = max_pair1

#             # One pair, t1
#             if num_pairs == 1:
#                 score[0] = 1

#             # >=Two pairs, t2
#             else:
#                 score[0] = 2
#                 max_pair2 = pair_ranks[-2]
#                 score[2] = max_pair2

#                 # card_ranks_copy = deepcopy(card_ranks)
#                 # card_ranks_copy.remove(max_pair1)
#                 # card_ranks_copy.remove(max_pair2)
#                 card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1 and cr != max_pair2]
#                 score[3] = card_ranks_copy[-1]
#                 # del card_ranks_copy

#         # Three of a kind, t3
#         max_triple1 = 0
#         num_triples = len(triple_ranks)
#         if num_triples >= 1:
#             score[0] = 3
#             max_triple1 = triple_ranks[-1]
#             score[1] = max_triple1
#             # card_ranks_copy = deepcopy(card_ranks)
#             # card_ranks_copy.remove(max_triple1)
#             card_ranks_copy = [cr for cr in card_ranks if cr != max_triple1]

#             score[2] = card_ranks_copy[-1]
#             score[3] = card_ranks_copy[-2] if len(card_ranks_copy) >= 2 else 0
#             score[4] = score[5] = 0
#             # del card_ranks_copy

#         # Straight, t4: five cards in sequential order, NOT of the same suit
#         if len(straight_ranks) >= 1:
#             score[0] = 4
#             score[1] = straight_ranks[-1]
#             for _ in range(2, 6):
#                 score[_] = 0

#         # Flush, t5: five cards, of the same suit, NOT of sequential order
#         straight_flush_ranks = []
#         for _ in suit_and_cards:
#             cards_ = suit_and_cards[_]
#             len_ = len(cards_)
#             if len_ >= 5:
#                 flushes = [card2rank_f(sc) for sc in cards_]
#                 flushes.sort()
#                 score[0] = 5
#                 for f_ in range(1, 6):
#                     score[f_] = flushes[-f_]

#                 if flushes[-1] == 14 and flushes[0] == 2 and flushes[3] == 5:
#                     straight_flush_ranks.append(1)

#                 for ic in range(len_ - 4):
#                     if flushes[ic] + 4 == flushes[ic + 4]:
#                         straight_flush_ranks.append(flushes[ic])

#         if len(straight_flush_ranks) == 0:
#             # Full house, t6: 3+2
#             if num_pairs > 0 and num_triples > 0:
#                 score[0] = 6
#                 score[1] = max_triple1
#                 score[2] = max_pair1 if len(triple_ranks) == 1 \
#                     else (triple_ranks[-2] if triple_ranks[-2] > max_pair1 else max_pair1)
#                 for _ in range(3, 6):
#                     score[_] = 0

#             # Four of a kind
#             if len(quad_ranks) == 1:
#                 score[0] = 7
#                 score[1] = quad_ranks[-1]
#                 score[2] = card_ranks[-2] if quad_ranks[-1] == max_single1 else max_single1
#                 for _ in range(3, 6):
#                     score[_] = 0

#         # Straight flush, t8
#         else:
#             score[0] = 8
#             score[1] = straight_flush_ranks[-1]
#             for _ in range(2, 6):
#                 score[_] = 0

#         # Additional consideration for tiebreaker
#         if num_cards >= 5:
#             if score[0] == 0:
#                 # High card, t0
#                 for _ in range(2, 6):
#                     score[_] = card_ranks[-_]  # no duplicates here
#             elif score[0] == 1:
#                 # One pair
#                 # card_ranks_copy = deepcopy(card_ranks)
#                 # card_ranks_copy.remove(max_pair1)
#                 card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1]

#                 for _ in range(1, 4):
#                     score[_ + 1] = card_ranks_copy[-_]

#     return np.sum(weights*score)


def compute_scoreI(all_cards: list[int]):
    num_cards = len(all_cards)

    # Prep work
    rank_and_cards: dict[int, list[int]] = {}
    suit_and_cards: dict[int, list[int]] = {}

    for c in all_cards:
        c_rank = card2rank[c]
        if c_rank in rank_and_cards:
            rank_and_cards[c_rank] += [c]
        else:
            rank_and_cards[c_rank] = [c]

        c_suit = card2row[c]
        if c_suit in suit_and_cards:
            suit_and_cards[c_suit] += [c]
        else:
            suit_and_cards[c_suit] = [c]

    # High card, t0
    card_ranks = list(rank_and_cards.keys())
    card_ranks.sort()
    num_ranks = len(card_ranks)
    score = np.zeros(6)
    score[1] = card_ranks[-1]
    max_single1 = score[1]

    if num_cards == 2:
        # A pair, t1
        if num_ranks == 1:
            score[0] = 1
        else:
            score[2] = card_ranks[-2]
    else:  # when `num_cards` is in [5, 6, 7]
        pair_ranks = []
        triple_ranks = []
        quad_ranks = []
        for _ in card_ranks:  # sorted
            len_ = len(rank_and_cards[_])
            if len_ == 2:
                pair_ranks.append(_)
            elif len_ == 3:
                triple_ranks.append(_)
            elif len_ == 4:
                quad_ranks.append(_)

        # Generate `straights`
        seq_records = list()
        start = i = 0
        longest = 1
        while i < num_ranks:
            if i + 1 < num_ranks and card_ranks[i + 1] - card_ranks[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        straight_ranks = []
        # treat aces as 1
        if 14 in card_ranks:
            if card_ranks[seq_records[0][0]] == 2 and seq_records[0][1] >= 4:
                straight_ranks.append(1)

        for seq in seq_records:
            if seq[1] >= 5:
                straight_ranks.append(card_ranks[seq[0]])

        # Pair(s)
        num_pairs = len(pair_ranks)
        max_pair1 = 0
        if num_pairs > 0:
            max_pair1 = pair_ranks[-1]
            score[1] = max_pair1

            # One pair, t1
            if num_pairs == 1:
                score[0] = 1

            # >=Two pairs, t2
            else:
                score[0] = 2
                max_pair2 = pair_ranks[-2]
                score[2] = max_pair2

                # card_ranks_copy = deepcopy(card_ranks)
                # card_ranks_copy.remove(max_pair1)
                # card_ranks_copy.remove(max_pair2)
                card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1 and cr != max_pair2]
                score[3] = card_ranks_copy[-1]
                # del card_ranks_copy

        # Three of a kind, t3
        max_triple1 = 0
        num_triples = len(triple_ranks)
        if num_triples >= 1:
            score[0] = 3
            max_triple1 = triple_ranks[-1]
            score[1] = max_triple1
            # card_ranks_copy = deepcopy(card_ranks)
            # card_ranks_copy.remove(max_triple1)
            card_ranks_copy = [cr for cr in card_ranks if cr != max_triple1]

            score[2] = card_ranks_copy[-1]
            score[3] = card_ranks_copy[-2] if len(card_ranks_copy) >= 2 else 0
            score[4] = score[5] = 0
            # del card_ranks_copy

        # Straight, t4: five cards in sequential order, NOT of the same suit
        if len(straight_ranks) >= 1:
            score[0] = 4
            score[1] = straight_ranks[-1]
            for _ in range(2, 6):
                score[_] = 0

        # Flush, t5: five cards, of the same suit, NOT of sequential order
        straight_flush_ranks = []
        for _ in suit_and_cards:
            cards_ = suit_and_cards[_]
            len_ = len(cards_)
            if len_ >= 5:
                # flushes = [card2rank_f(sc) for sc in cards_]
                flushes = [card2rank[sc] for sc in cards_]
                flushes.sort()
                score[0] = 5
                for f_ in range(1, 6):
                    score[f_] = flushes[-f_]

                if flushes[-1] == 14 and flushes[0] == 2 and flushes[3] == 5:
                    straight_flush_ranks.append(1)

                for ic in range(len_ - 4):
                    if flushes[ic] + 4 == flushes[ic + 4]:
                        straight_flush_ranks.append(flushes[ic])

        if len(straight_flush_ranks) == 0:
            # Full house, t6: 3+2
            if num_pairs > 0 and num_triples > 0:
                score[0] = 6
                score[1] = max_triple1
                score[2] = max_pair1 if len(triple_ranks) == 1 \
                    else (triple_ranks[-2] if triple_ranks[-2] > max_pair1 else max_pair1)
                for _ in range(3, 6):
                    score[_] = 0

            # Four of a kind
            if len(quad_ranks) == 1:
                score[0] = 7
                score[1] = quad_ranks[-1]
                score[2] = card_ranks[-2] if quad_ranks[-1] == max_single1 else max_single1
                for _ in range(3, 6):
                    score[_] = 0

        # Straight flush, t8
        else:
            score[0] = 8
            score[1] = straight_flush_ranks[-1]
            for _ in range(2, 6):
                score[_] = 0

        # Additional consideration for tiebreaker
        if num_cards >= 5:
            if score[0] == 0:
                # High card, t0
                for _ in range(2, 6):
                    score[_] = card_ranks[-_]  # no duplicates here
            elif score[0] == 1:
                # One pair
                # card_ranks_copy = deepcopy(card_ranks)
                # card_ranks_copy.remove(max_pair1)
                card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1]

                for _ in range(1, 4):
                    score[_ + 1] = card_ranks_copy[-_]

    return np.sum(weights*score)


# def compute_scoreII(all_cards: list[int], suited: bool)->int:
#     score = np.zeros(6, dtype=np.int32)
#     if suited:
#         # Prep work
#         rank_and_cards: dict[int, list[int]] = {}
#         suit_and_cards: dict[int, list[int]] = {}

#         for c in all_cards:
#             c_rank = card2rank[c]
#             if c_rank in rank_and_cards:
#                 rank_and_cards[c_rank] += [c]
#             else:
#                 rank_and_cards[c_rank] = [c]

#             c_suit = card2row[c]
#             if c_suit in suit_and_cards:
#                 suit_and_cards[c_suit] += [c]
#             else:
#                 suit_and_cards[c_suit] = [c]

#         # Flush, t5: five cards, of the same suit, NOT of sequential order
#         straight_flush_ranks: list[int] = []
#         for _ in suit_and_cards:
#             cards_ = suit_and_cards[_]
#             len_ = len(cards_)
#             if len_ >= 5:
#                 flushes = [card2rank_f(sc) for sc in cards_]
#                 flushes.sort()
#                 score[0] = 5
#                 for f_ in range(1, 6):
#                     score[f_] = flushes[-f_]

#                 if flushes[-1] == 14 and flushes[0] == 2 and flushes[3] == 5:
#                     straight_flush_ranks.append(1)

#                 for ic in range(len_ - 4):
#                     if flushes[ic] + 4 == flushes[ic + 4]:
#                         straight_flush_ranks.append(flushes[ic])

#         if len(straight_flush_ranks) == 0:
#             if score[0] != 5: # we don't have a flush
#                 card_ranks = list(rank_and_cards.keys())
#                 card_ranks.sort()
#                 max_single1 = card_ranks[-1]

#                 pair_ranks: list[int] = []
#                 triple_ranks: list[int] = []
#                 quad_ranks: list[int] = []
#                 for _ in card_ranks:  # sorted
#                     len_ = len(rank_and_cards[_])
#                     if len_ == 2:
#                         pair_ranks.append(_)
#                     elif len_ == 3:
#                         triple_ranks.append(_)
#                     elif len_ == 4:
#                         quad_ranks.append(_)

#                 # Four of a kind
#                 if len(quad_ranks) == 1:
#                     score[0] = 7
#                     score[1] = quad_ranks[-1]
#                     score[2] = card_ranks[-2] if quad_ranks[-1] == max_single1 else max_single1
#                 # Full house, t6: 3+2
#                 elif len(pair_ranks) > 0 and len(triple_ranks) > 0: 
#                     max_pair1 = pair_ranks[-1]
#                     max_triple1 = triple_ranks[-1]

#                     score[0] = 6
#                     score[1] = max_triple1
#                     score[2] = max_pair1 if len(triple_ranks) == 1 \
#                         else (triple_ranks[-2] if triple_ranks[-2] > max_pair1 else max_pair1)
#             # no need to reset score[3:6] to zeros: cannot have flush + full house or flush + four of a kind simultaneously

#         else: # Straight flush, t8
#             score[0] = 8
#             score[1] = straight_flush_ranks[-1]
#             for _ in range(2, 6):
#                 score[_] = 0

#         return np.sum(weights*score)
#     else:  # non-suited
#         # Prep work
#         rank_and_cards: dict[int, list[int]] = {}
#         for c in all_cards:
#             c_rank = card2rank[c]
#             if c_rank in rank_and_cards:
#                 rank_and_cards[c_rank] += [c]
#             else:
#                 rank_and_cards[c_rank] = [c]

#         # High card, t0
#         card_ranks = list(rank_and_cards.keys())
#         card_ranks.sort()
#         num_ranks = len(card_ranks)
#         score[1] = card_ranks[-1]
#         max_single1 = card_ranks[-1]

#         pair_ranks: list[int] = []
#         triple_ranks: list[int] = []
#         quad_ranks: list[int] = []
#         for _ in card_ranks:  # sorted
#             len_ = len(rank_and_cards[_])
#             if len_ == 2:
#                 pair_ranks.append(_)
#             elif len_ == 3:
#                 triple_ranks.append(_)
#             elif len_ == 4:
#                 quad_ranks.append(_)

#         # Generate `straights`
#         seq_records = list()
#         start = i = 0
#         longest = 1
#         while i < num_ranks:
#             if i + 1 < num_ranks and card_ranks[i + 1] - card_ranks[i] == 1:
#                 longest += 1
#                 i += 1
#             else:
#                 seq_records.append((start, longest))
#                 i += 1
#                 start = i
#                 longest = 1

#         straight_ranks: list[int] = []
#         # treat aces as 1
#         if 14 in card_ranks:
#             if card_ranks[seq_records[0][0]] == 2 and seq_records[0][1] >= 4:
#                 straight_ranks.append(1)

#         for seq in seq_records:
#             if seq[1] >= 5:
#                 straight_ranks.append(card_ranks[seq[0]])

#         # Pair(s)
#         num_pairs = len(pair_ranks)
#         max_pair1 = 0
#         if num_pairs > 0:
#             max_pair1 = pair_ranks[-1]
#             score[1] = max_pair1

#             # One pair, t1
#             if num_pairs == 1:
#                 score[0] = 1

#             # >=Two pairs, t2
#             else:
#                 score[0] = 2
#                 max_pair2 = pair_ranks[-2]
#                 score[2] = max_pair2

#                 card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1 and cr != max_pair2]
#                 score[3] = card_ranks_copy[-1]

#         # Three of a kind, t3
#         max_triple1 = 0
#         num_triples = len(triple_ranks)
#         if num_triples >= 1:
#             score[0] = 3
#             max_triple1 = triple_ranks[-1]
#             score[1] = max_triple1
#             card_ranks_copy = [cr for cr in card_ranks if cr != max_triple1]

#             score[2] = card_ranks_copy[-1]
#             score[3] = card_ranks_copy[-2] if len(card_ranks_copy) >= 2 else 0
#             score[4] = score[5] = 0

#         # Straight, t4: five cards in sequential order, NOT of the same suit
#         if len(straight_ranks) >= 1:
#             score[0] = 4
#             score[1] = straight_ranks[-1]
#             for _ in range(2, 6):
#                 score[_] = 0

#         # Full house, t6: 3+2
#         if num_pairs > 0 and num_triples > 0:
#             score[0] = 6
#             score[1] = max_triple1
#             score[2] = max_pair1 if len(triple_ranks) == 1 \
#                 else (triple_ranks[-2] if triple_ranks[-2] > max_pair1 else max_pair1)
#             for _ in range(3, 6):
#                 score[_] = 0

#         # Four of a kind
#         if len(quad_ranks) == 1:
#             score[0] = 7
#             score[1] = quad_ranks[-1]
#             score[2] = card_ranks[-2] if quad_ranks[-1] == max_single1 else max_single1
#             for _ in range(3, 6):
#                 score[_] = 0

#         # Additional consideration for tiebreaker
#         if score[0] == 0:
#             # High card, t0
#             for _ in range(2, 6):
#                 score[_] = card_ranks[-_]  # no duplicates here
#         elif score[0] == 1:
#             # One pair
#             card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1]

#             for _ in range(1, 4):
#                 score[_ + 1] = card_ranks_copy[-_]

#         return np.sum(weights*score)


def compute_score(all_cards):
    num_cards = len(all_cards)
    # if num_cards not in [2, 5, 6, 7]:
    #     print('Error processing all_cards -> compute_score()')
    #     sys.exit(-1)

    # Prep work
    rank_and_cards = {}
    suit_and_cards = {}

    for c in all_cards:
        c_rank = card2rank[c]
        if c_rank in rank_and_cards:
            rank_and_cards[c_rank].append(c)
        else:
            rank_and_cards[c_rank] = [c]

        c_suit = card2row[c]
        if c_suit in suit_and_cards:
            suit_and_cards[c_suit].append(c)
        else:
            suit_and_cards[c_suit] = [c]

    # High card, t0
    card_ranks = list(rank_and_cards.keys())
    card_ranks.sort()
    num_ranks = len(card_ranks)
    # score = [0, 0, 0, 0, 0, 0]
    score = np.zeros(6)
    score[1] = card_ranks[-1]
    max_single1 = score[1]

    if num_cards == 2:
        # A pair, t1
        if num_ranks == 1:
            score[0] = 1
        else:
            score[2] = card_ranks[-2]
    else:  # when `num_cards` is in [5, 6, 7]
        pair_ranks = []
        triple_ranks = []
        quad_ranks = []
        for _ in card_ranks:  # sorted
            len_ = len(rank_and_cards[_])
            if len_ == 2:
                pair_ranks.append(_)
            elif len_ == 3:
                triple_ranks.append(_)
            elif len_ == 4:
                quad_ranks.append(_)

        # Generate `straights`
        seq_records = list()
        start = i = 0
        longest = 1
        while i < num_ranks:
            if i + 1 < num_ranks and card_ranks[i + 1] - card_ranks[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1

        straight_ranks = []
        # treat aces as 1
        if 14 in card_ranks:
            if card_ranks[seq_records[0][0]] == 2 and seq_records[0][1] >= 4:
                straight_ranks.append(1)

        for seq in seq_records:
            if seq[1] >= 5:
                straight_ranks.append(card_ranks[seq[0]])

        # Pair(s)
        num_pairs = len(pair_ranks)
        max_pair1 = 0
        if num_pairs > 0:
            max_pair1 = pair_ranks[-1]
            score[1] = max_pair1

            # One pair, t1
            if num_pairs == 1:
                score[0] = 1

            # >=Two pairs, t2
            else:
                score[0] = 2
                max_pair2 = pair_ranks[-2]
                score[2] = max_pair2

                # card_ranks_copy = deepcopy(card_ranks)
                # card_ranks_copy.remove(max_pair1)
                # card_ranks_copy.remove(max_pair2)
                card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1 and cr != max_pair2]
                
                score[3] = card_ranks_copy[-1]
                # del card_ranks_copy

        # Three of a kind, t3
        max_triple1 = 0
        num_triples = len(triple_ranks)
        if num_triples >= 1:
            score[0] = 3
            max_triple1 = triple_ranks[-1]
            score[1] = max_triple1
            # card_ranks_copy = deepcopy(card_ranks)
            # card_ranks_copy.remove(max_triple1)
            card_ranks_copy = [cr for cr in card_ranks if cr != max_triple1]

            score[2] = card_ranks_copy[-1]
            score[3] = card_ranks_copy[-2] if len(card_ranks_copy) >= 2 else 0
            score[4] = score[5] = 0
            # del card_ranks_copy

        # Straight, t4: five cards in sequential order, NOT of the same suit
        if len(straight_ranks) >= 1:
            score[0] = 4
            score[1] = straight_ranks[-1]
            for _ in range(2, 6):
                score[_] = 0

        # Flush, t5: five cards, of the same suit, NOT of sequential order
        straight_flush_ranks = []
        for _ in suit_and_cards:
            cards_ = suit_and_cards[_]
            len_ = len(cards_)
            if len_ >= 5:
                flushes = [card2rank[sc] for sc in cards_]
                flushes.sort()
                score[0] = 5
                for f_ in range(1, 6):
                    score[f_] = flushes[-f_]

                if flushes[-1] == 14 and flushes[0] == 2 and flushes[3] == 5:
                    straight_flush_ranks.append(1)

                for ic in range(len_ - 4):
                    if flushes[ic] + 4 == flushes[ic + 4]:
                        straight_flush_ranks.append(flushes[ic])

        if len(straight_flush_ranks) == 0:
            # Full house, t6: 3+2
            if num_pairs > 0 and num_triples > 0:
                score[0] = 6
                score[1] = max_triple1
                score[2] = max_pair1 if len(triple_ranks) == 1 \
                    else (triple_ranks[-2] if triple_ranks[-2] > max_pair1 else max_pair1)
                for _ in range(3, 6):
                    score[_] = 0

            # Four of a kind
            if len(quad_ranks) == 1:
                score[0] = 7
                score[1] = quad_ranks[-1]
                score[2] = card_ranks[-2] if quad_ranks[-1] == max_single1 else max_single1
                for _ in range(3, 6):
                    score[_] = 0

        # Straight flush, t8
        else:
            score[0] = 8
            score[1] = straight_flush_ranks[-1]
            for _ in range(2, 6):
                score[_] = 0

        # Additional consideration for tiebreaker
        if num_cards >= 5:
            if score[0] == 0:
                # High card, t0
                for _ in range(2, 6):
                    score[_] = card_ranks[-_]  # no duplicates here
            elif score[0] == 1:
                # One pair
                # card_ranks_copy = deepcopy(card_ranks)
                # card_ranks_copy.remove(max_pair1)
                card_ranks_copy = [cr for cr in card_ranks if cr != max_pair1]

                for _ in range(1, 4):
                    score[_ + 1] = card_ranks_copy[-_]

    return [np.sum(weights*score), score]


def show_cards(all_cards):
    if len(all_cards) == 0:
        return
    
    for c in all_cards:
        print('%s%s|' % (rank2str[card2rank[c]], row2symbol[card2row[c]]), end='')
    print()


def main():
    deck = list(range(1, 53))
    scene = 2

    if scene == 0:  # random hole cards and/or community cards
        num_sims = 10
        for _ in range(num_sims):
            np.random.shuffle(deck)
            cards = deck[:7]
            print('Cards: ', end='')
            show_cards(cards)

            score_ = compute_score(cards)[1]
            print('\nHand: [\33[94m%s\033[0m] | main rank: %i | kicker1: %i | kicker2: %i'
                  % (hand_names[score_[0]], score_[1], score_[2], score_[3]))
    elif scene == 1:  # prescribed hole cards and/or community cards
        hole_cards = [40, 5, 14, 37]
        community_cards = [20, 9, 44, 13, 45]
        num_players = int(len(hole_cards) / 2)

        for _ in range(num_players):
            cards = hole_cards[_*2:_*2+2] + community_cards
            print('p%i all cards: ' % _, end='')
            show_cards(cards)

            score_ = compute_score(cards)
            print('\nHand: [\33[94m%s\033[0m] | main rank: %i | kicker1: %i | kicker2: %i | score: %i'
                  % (hand_names[score_[1][0]], score_[1][1], score_[1][2], score_[1][3], score_[0]))
    elif scene == 11:  # real test cases
        num_sims = 10000
        i_data = 0

        test_data = {}
        if i_data == 0:
            test_data = {
                0: {'hc': [40, 5, 14, 37], 'cc': [], 'ewr': [25, 67]},
                1: {'hc': [40, 5, 14, 37], 'cc': [20, 9, 44], 'ewr': [82, 15]},
                2: {'hc': [40, 5, 14, 37], 'cc': [20, 9, 44, 13], 'ewr': [93, 7]},

                3: {'hc': [9, 35, 36, 30], 'cc': [], 'ewr': [68, 32]},
                4: {'hc': [9, 35, 36, 30], 'cc': [29, 12, 38], 'ewr': [59, 41]},
                5: {'hc': [9, 35, 36, 30], 'cc': [29, 12, 38, 3], 'ewr': [70, 25]},

                6: {'hc': [11, 10, 9, 46], 'cc': [], 'ewr': [67, 31]},
                7: {'hc': [11, 10, 9, 46], 'cc': [23, 12, 20], 'ewr': [81, 19]},

                8: {'hc': [26, 39, 49, 10], 'cc': [], 'ewr': [81, 19]},
                9: {'hc': [26, 39, 49, 10], 'cc': [36, 12, 17], 'ewr': [12, 88]},

                10: {'hc': [27, 34, 43, 17], 'cc': [], 'ewr': [48, 51]},
                11: {'hc': [27, 34, 43, 17], 'cc': [22, 30, 16], 'ewr': [4, 96]},

                12: {'hc': [49, 33, 42, 29, 1, 13], 'cc': [], 'ewr': [28, 28, 44]},
                13: {'hc': [42, 29, 1, 13], 'cc': [], 'ewr': [50, 49]},
                14: {'hc': [42, 29, 1, 13], 'cc': [40, 36, 9], 'ewr': [8, 92]},

                15: {'hc': [12, 25, 1, 50, 7, 20], 'cc': [], 'ewr': [57, 25, 18]},
                16: {'hc': [12, 25, 1, 50], 'cc': [], 'ewr': [71, 28]},
                17: {'hc': [12, 25, 1, 50], 'cc': [43, 9, 23], 'ewr': [84, 16]},
                18: {'hc': [12, 25, 1, 50], 'cc': [43, 9, 23, 37], 'ewr': [89, 11]},

                19: {'hc': [39, 32, 49, 23, 40, 8], 'cc': [], 'ewr': [28, 47, 25]},
                20: {'hc': [39, 32, 49, 23], 'cc': [], 'ewr': [32, 68]},
                21: {'hc': [39, 32, 49, 23], 'cc': [10, 38, 47], 'ewr': [7, 93]},
                22: {'hc': [39, 32, 49, 23], 'cc': [10, 38, 47, 31], 'ewr': [16, 84]},

                23: {'hc': [1, 22, 40, 13], 'cc': [], 'ewr': [24, 71]},
                24: {'hc': [1, 22, 40, 13], 'cc': [8, 41, 31], 'ewr': [14, 82]},
                25: {'hc': [1, 22, 40, 13], 'cc': [8, 41, 31, 7], 'ewr': [16, 84]},

                26: {'hc': [13, 25, 49, 23], 'cc': [], 'ewr': [44, 56]},
                27: {'hc': [13, 25, 49, 23], 'cc': [36, 24, 27], 'ewr': [65, 34]},
                28: {'hc': [13, 25, 49, 23], 'cc': [36, 24, 27, 41], 'ewr': [77, 23]},

                29: {'hc': [39, 6, 45, 19], 'cc': [], 'ewr': [29, 68]},
                30: {'hc': [39, 6, 45, 19], 'cc': [5, 30, 52], 'ewr': [91, 4]},
                31: {'hc': [39, 6, 45, 19], 'cc': [5, 30, 52, 26], 'ewr': [100, 0]},
                #: {'hc': [], 'cc': [], 'ewr': []},
            }
        elif i_data == 1:
            test_data = {
                2: {'hc': [40, 5, 14, 37], 'cc': [20, 9, 44, 13], 'ewr': [93, 7]},
            }

        for test_ in test_data.values():
            deck_ = deepcopy(deck)
            hole_cards = test_['hc']
            community_cards = test_['cc']
            sz_cc = len(community_cards)
            num_players = int(len(hole_cards)/2)
            player_and_wins = {}
            for _ in range(num_players):
                player_and_wins[_] = 0
            player_and_wins[-1] = 0  # draws

            for _ in hole_cards:
                deck_.remove(_)
            for _ in community_cards:
                deck_.remove(_)

            for _ in range(num_sims):
                np.random.shuffle(deck_)
                community_cards_all = []

                if sz_cc == 0:
                    community_cards_all.append(deck_[1])
                    community_cards_all.append(deck_[2])
                    community_cards_all.append(deck_[3])
                    community_cards_all.append(deck_[5])
                    community_cards_all.append(deck_[7])
                elif sz_cc == 3 or sz_cc == 4:
                    for cc in community_cards:
                        community_cards_all.append(cc)

                    for rcc in range(5 - sz_cc):
                        community_cards_all.append(deck_[rcc*2+1])

                winning_player = 0
                winning_score = compute_score(hole_cards[:2] + community_cards_all)[0]
                for p in range(1, num_players):
                    p_score = compute_score(hole_cards[p*2:p*2+2] + community_cards_all)[0]
                    if p_score > winning_score:
                        winning_score = p_score
                        winning_player = p
                        # for cca in community_cards_all:
                        #     print('%s%s|' % (rank2str[card2rank[cca]], row2symbol[card2row[cca]]), end='')
                        # print()
                    elif p_score == winning_score and num_players == 2:
                        winning_player = -1
                        # for cca in community_cards_all:
                        #     print('%s%s|' % (rank2str[card2rank[cca]], row2symbol[card2row[cca]]), end='')
                        # print()

                player_and_wins[winning_player] += 1

            print('Community cards: ' + ('' if sz_cc else '[]'), end='')
            show_cards(community_cards)
            print('\u2186')

            for _ in range(num_players):
                hole_card1 = hole_cards[_ * 2]
                hole_card2 = hole_cards[_ * 2 + 1]
                print('p%i: %s%s|%s%s --> '
                      '%i%% vs expected: %i%%' % (_,
                                                  rank2str[card2rank[hole_card1]], row2symbol[card2row[hole_card1]],
                                                  rank2str[card2rank[hole_card2]], row2symbol[card2row[hole_card2]],
                                                  np.round(player_and_wins[_]/num_sims*100), test_['ewr'][_]))

            if sum([abs(player_and_wins[p_]/num_sims*100 - test_['ewr'][p_]) for p_ in range(num_players)]) \
                    <= num_players * 2:
                print('::::::::::\033[32mtest succeeded!\033[0m::::::::::')
            else:
                print('::::::::::\033[31mtest failed!\033[0m::::::::::')
    elif scene == 2:
        print(compute_score([20,22,25,46,16,18,19])[0])


if __name__ == '__main__':
    main()
