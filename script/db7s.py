import collections
import itertools
import time
import random
import pickle
import math

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
ROYAL_FLUSH = 9

def comb(n, k):
    """
    Compute the combination C(n, k) = n! / (k! * (n - k)!).
    """
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)  # Requires Python 3.10+
    # For earlier Python versions, uncomment the following:
    # return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def card_to_int(card):
    """
    Convert a (rank, suit) tuple to a unique integer.
    Example Mapping:
      - (1, 0) -> 1
      - (1, 1) -> 14
      - (2, 0) -> 2
      - (2, 1) -> 15
      - ...
      - (13, 0) -> 13
      - (13, 1) -> 26
    """
    rank, suit = card

    return rank - 1 + suit * 13

def int_to_card(integer):
    """
    Convert a unique integer back to a (rank, suit) tuple.
    """
    if not (1 <= integer <= 26):
        raise ValueError("Integer out of valid range.")
    suit = 0 if integer <= 13 else 1
    rank = integer if suit == 0 else integer - 13
    return (rank, suit)

def vector_to_int(vector):
    """
    Convert a sorted 7-element combination vector to its unique integer rank.
    Here, vector contains (rank, suit) tuples.
    """
    # First, convert each card to its unique integer representation
    int_vector = sorted([card_to_int(card) for card in vector])

    rank_value = 0
    for i in range(7):
        remaining = 7 - i - 1
        start = int_vector[i -1] + 1 if i > 0 else 1
        for x in range(start, int_vector[i]):
            rank_value += comb(26 - x, remaining)
    return rank_value

def int_to_vector(rank):
    """
    Convert an integer rank back to its corresponding sorted 7-element combination vector.
    Returns a list of (rank, suit) tuples.
    """
    if not (0 <= rank < comb(26,7)):
        raise ValueError(f"Rank must be between 0 and {comb(26,7) -1}.")

    vector = []
    previous = 0
    for i in range(7, 0, -1):
        x = previous + 1
        while x <= 26:
            c = comb(26 - x, i -1)
            if c <= rank:
                rank -= c
                x += 1
            else:
                break
        card_int = x
        vector.append(int_to_card(card_int))
        previous = x
    return vector

class HandResult:
    """
    Stores the evaluated 5-card best-result in:
      handType, handRank, kicker1, kicker2, kicker3, kicker4
    This mimics the struct logic from C++ but in Python.
    """
    def __init__(self, handType, handRank, kickers):
        self.handType = handType
        self.handRank = handRank
        # Fill up to 4 kickers
        self.kickers = list(kickers)[:4] + [0]*(4 - len(kickers))

    def to_list(self):
        return [self.handType, self.handRank] + self.kickers

def compareHands(hr1, hr2):
    """
    Compare two HandResult objects. Return positive if hr1>hr2,
    negative if hr1<hr2, zero if tie.
    """
    arr1 = hr1.to_list()
    arr2 = hr2.to_list()
    for x, y in zip(arr1, arr2):
        if x != y:
            return x - y
    return 0

def evaluate7CardHand(cards):
    """
    Evaluate a 7-card hand using a direct approach:
      1) Count ranks/suits
      2) Check flush / straight
      3) Then check four-of-a-kind, full house, flush, straight, etc.
      4) Return the best found result immediately (as in encode.hpp)

    Special flush rule: only suit=0 is considered a flush.
    """
    # Extract ranks and suits
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    # Sort ranks descending
    ranks.sort(reverse=True)

    rankCount = collections.Counter(ranks)
    suitCount = collections.Counter(suits)

    # Check for "flush" (only suit=0 is flush)
    flushSuit = 0
    if suitCount.get(0, 0) >= 5:
        flushSuit = 1  # We have at least 5 cards with suit=0

    # Check for straight
    is_straight, straight_top = checkStraight(ranks)

    # ---- Check for straight flush ----
    if flushSuit == 1:
        # Extract only the suited cards (suit=0)
        suited_cards = [card for card in cards if card[1] == 0]
        suited_ranks = [card[0] for card in suited_cards]
        suited_ranks.sort(reverse=True)
        
        # Check if the suited cards form a straight
        is_flush_straight, flush_straight_top = checkStraight(suited_ranks)
        
        # If the suited cards form a straight, then it's a straight flush
        if is_flush_straight:
            # Check for royal flush (A-K-Q-J-10 with suit=0)
            if flush_straight_top == 14 and hasAKQJT(suited_cards, 0):
                return HandResult(ROYAL_FLUSH, 14, [])
            else:
                return HandResult(STRAIGHT_FLUSH, flush_straight_top, [])

    # ---- Four of a kind ----
    for rk, cnt in rankCount.items():
        if cnt == 4:
            # Kicker is first rank not equal to 'rk'
            kicker = max(r for r in ranks if r != rk)
            return HandResult(FOUR_OF_A_KIND, rk, [kicker])

    # ---- Full house (3 + 2) ----
    three_rank = None
    pair_rank = None
    for rk, cnt in rankCount.items():
        if cnt == 3:
            if three_rank is None or rk > three_rank:
                three_rank = rk
    for rk, cnt in rankCount.items():
        if cnt >= 2 and rk != three_rank:
            if pair_rank is None or rk > pair_rank:
                pair_rank = rk
    if three_rank and pair_rank:
        return HandResult(FULL_HOUSE, three_rank, [pair_rank])

    # ---- Flush (just suit=0) ----
    if flushSuit == 1:
        # Gather flush cards, sorted descending
        flush_cards = sorted((r for (r, s) in cards if s == 0), reverse=True)
        # We only need the top 5 from these flush cards
        top5 = flush_cards[:5]
        return HandResult(FLUSH, top5[0], top5[1:])

    # ---- Straight ----
    if is_straight:
        return HandResult(STRAIGHT, straight_top, [])

    # ---- Three of a kind ----
    for rk, cnt in rankCount.items():
        if cnt == 3:
            # pick top 2 kickers from the other ranks
            kickers = [r for r in ranks if r != rk]
            return HandResult(THREE_OF_A_KIND, rk, kickers[:2])

    # ---- Two pair ----
    pairs = [rk for rk, cnt in rankCount.items() if cnt == 2]
    if len(pairs) >= 2:
        pairs.sort(reverse=True)
        highPair, secondPair = pairs[0], pairs[1]
        # find a kicker
        other_ranks = [r for r in ranks if r not in (highPair, secondPair)]
        kicker = other_ranks[0]
        return HandResult(TWO_PAIR, highPair, [secondPair, kicker])

    # ---- One pair ----
    if len(pairs) == 1:
        pairRank = pairs[0]
        # gather up to 3 kickers
        other_ranks = [r for r in ranks if r != pairRank]
        return HandResult(ONE_PAIR, pairRank, other_ranks[:3])

    # ---- High card ----
    top = ranks[0]
    rest = ranks[1:5]  # only up to 4 kickers
    return HandResult(HIGH_CARD, top, rest)

def checkStraight(sorted_desc_ranks):
    """
    Check if there's a 5-card straight in a descending list of ranks (e.g., [14,13,12,11,10,8,3] for 7 cards).
    If found, return (True, topOfStraight). If multiple possible straights, pick the highest top.
    Also handle Ace-low check.
    """
    unique = []
    for r in sorted_desc_ranks:
        if r not in unique:
            unique.append(r)
    # If fewer than 5 unique ranks, no straight
    if len(unique) < 5:
        return False, 0

    # Slide to find consecutive
    for start in range(len(unique) - 4):
        window = unique[start:start+5]
        if window[0] - window[4] == 4:
            return True, window[0]

    # Ace-low check: if we have ranks like A,5,4,3,2 => 14,5,4,3,2
    # We'll check if 14 is in unique and 5,4,3,2 are in unique
    if 14 in unique and all(x in unique for x in [2,3,4,5]):
        return True, 5  # 5-high straight

    return False, 0

def hasAKQJT(cards, suit=0):
    """
    Check if the given 7 cards contain the sequence A,K,Q,J,10 (ranks=14,13,12,11,10)
    all in the specified suit. This is a helper for Royal Flush check.
    """
    needed = {14,13,12,11,10}
    actual = {r for (r,s) in cards if s == suit and r in needed}
    return len(actual) == 5

def generate_26_card_deck():
    """
    Generate a deck of 26 cards:
    Ranks: 2..14 (inclusive)
    Suits: 0 (suited) and 1 (nonsuited)
    """
    return [(rank, suit) for rank in range(2, 15) for suit in [0, 1]]

def generate_7_card_combinations(deck):
    """
    Generate all combinations of 7 cards from the 26-card deck.
    
    Note: This will generate C(26,7) = 888,030 combinations.
    """
    return itertools.combinations(deck, 7)



def reduce_combination7(original_combination):
    """
    Reduces a 7-card combination from 52 cards to a 7-card combination in 26 cards.
    
    Rules:
    - If more than 5 cards share the same suit, mark those as suited (0) and others as nonsuited (1).
    - Otherwise, mark all cards as nonsuited (1).
    
    Args:
        original_combination (list of tuples): 
            List of 7 tuples, each representing a card as (rank, suit),
            where rank is int [1-13] and suit is int [0-3].
    
    Returns:
        tuple of tuples: Reduced 7-card combination in 26-card deck, sorted for consistency.
    """

    # Count suits
    suit_counts = collections.Counter(card[1] for card in original_combination)

    # Find the most common suit
    most_common_suit, count = suit_counts.most_common(1)[0]

    if count >= 5:
        # Mark cards with the most_common_suit as suited (0) and others as nonsuited (1)
        reduced_combination = tuple(
            sorted(
                (card[0], 0 if card[1] == most_common_suit else 1)
                for card in original_combination
            )
        )
    else:
        # Mark all cards as nonsuited (1)
        reduced_combination = tuple(
            sorted(
                (card[0], 1)
                for card in original_combination
            )
        )

    return reduced_combination

def generate_full_deck():
    ranks = range(1, 14)    # 1 to 13
    suits = range(0, 4)     # 0 to 3
    deck = [(rank, suit) for suit in suits for rank in ranks]
    return deck

def reduce_combination7_preserve_order(combination):
    """
    Reduce a 6-card combination based on the following rule:
        - If more than four cards share the same suit (i.e., four or five), mark the rest as nonsuited (1).
        - Else, mark all as nonsuited (1).

    Additionally:
        - Sort the first two elements by rank.
        - Sort the last four elements by rank.
        - Preserve the separate ordering of the first two and last four elements.

    Args:
        combination (tuple): A tuple of 6 cards, each represented as (rank, suit).
    """
    # Count the occurrences of each suit in the combination
    suit_counts = collections.Counter(card[1] for card in combination)

    # Identify the most common suit and its count
    most_common_suit, count = suit_counts.most_common(1)[0]
    # Split the reduced combination into first two and last three cards
    last_four_com = combination[2:]
    suit_counts_com = collections.Counter(card[1] for card in last_four_com)
    most_common_suit_com, count_com = suit_counts_com.most_common(1)[0]
    if count >= 5:
        # More than three cards share the same suit
        # Mark cards with the most_common_suit as suited (0) and the rest as nonsuited (1)
        reduced = [(card[0], 0 if card[1] == most_common_suit else 1) for card in combination]
    elif count_com == 3 or count_com == 4:
        reduced = [(card[0], 0 if card[1] == most_common_suit_com else 1) for card in combination]
    else:
        # No suit has more than three cards; mark all as nonsuited (1)
        reduced = [(card[0], 1) for card in combination]

    first_two = reduced[:2]
    last_four = reduced[2:]
    first_two_sorted = sorted(first_two, key=lambda x: (x[0], -x[1]))
    last_four_sorted = sorted(last_four, key=lambda x: (x[0], -x[1]))

    # Combine the sorted first two and last three cards
    reduced_sorted = first_two_sorted + last_four_sorted

    return tuple(reduced_sorted)


def generate_all_hold_community_reduced_combinations():
    """
    Generate all unique reduced 5-card combinations by combining 2-card hold and 4-card community.
    
    Returns:
        set: A set of tuples, each representing a unique reduced 6-card combination.
            Each combination is a tuple of 6 (rank, suit_flag) tuples.
    """
    startgen = time.time()
    deck = generate_full_deck()
    unique_reduced_combinations = set()
    total_combinations = 0
    hold_combos = itertools.combinations(deck, 2)
    
    for hold in hold_combos:
        remaining_deck = set(deck) - set(hold)
        community_combos = itertools.combinations(remaining_deck, 5)
        for community in community_combos:
            six_card = tuple(hold + community)
            reduced = reduce_combination7_preserve_order(six_card)
            unique_reduced_combinations.add(reduced)
            total_combinations += 1
            if total_combinations % 2000000 == 0:
                print(f"Processed {total_combinations} 6-card combinations...")
                print(f"Unique reduced 6-card combinations: {len(unique_reduced_combinations)}")
                print(f"Current Combination: {reduced}")
                print(f"Total Combinations: {six_card}")
                now = time.time()
                print(f"Time taken: {now - startgen:.2f} seconds.")
    
    print(f"Total original 6-card combinations processed: {total_combinations}")
    print(f"Total unique reduced 6-card combinations: {len(unique_reduced_combinations)}")
    endgen = time.time()
    print(f"Total time taken generation: {endgen - startgen:.2f} seconds.")
    return unique_reduced_combinations

def generate_all_reduced_hold_opponent_combinations(hand):
    """
    Given a full deck and a 6-card hand, remove these 6 cards from the deck to form a 46-card deck.
    Then generate all 2-card combinations from this 46-card deck as community cards,
    and all 1-card combinations from the remaining 44-card deck as opponent hole cards.
    
    Returns:
        tuple: A tuple containing two sets:
                - community_combinations: Set of tuples representing community card combinations.
                - opponent_hole_combinations: Set of tuples representing opponent hole card combinations.
    """
    
    full_deck = generate_full_deck()
    # Remove the 5-card hand from the deck
    remaining_deck = set(full_deck) - set(hand)
    
    # Generate all 2-card community combinations from the remaining 47 cards
    opponent_hole_combos = set(itertools.combinations(remaining_deck, 2))    
    return opponent_hole_combos

def generate_winrate_given_holeplayer_and_community(community1, holeplayer, iteration = 100000):
    """"
    Given a community and hole player, generate the winrate.
    
    Args:
        community1 (list): The community cards.
        holeplayer (list): The hole player cards.
    
    Returns:
        winrate (float): The winrate of the hole player given the community cards.
    """
    hand1 = tuple(community1 + holeplayer)
    win = 0 
    lose = 0
    tie = 0
    winrate = 0
    cnt = 0
    # print(f"Community: {community1}, Hole Player: {holeplayer}")
    opponent_hole_combos = generate_all_reduced_hold_opponent_combinations(hand1)
    for holeplayer2 in opponent_hole_combos:
            hand2tem = tuple(tuple(holeplayer2) + tuple(community1))
            hand1tem = tuple(hand1)
            reduced1 = reduce_combination7(hand1tem)
            reduced2 = reduce_combination7(hand2tem)
            evaluation1 = evaluate7CardHand(reduced1)
            evaluation2 = evaluate7CardHand(reduced2)
            comparison = compareHands(evaluation1, evaluation2)
            if comparison >= 0:
                win += 1
            elif comparison <= 0:
                lose += 1
            else:
                tie += 1
            cnt += 1

            
    winrate = float(win) / (win + lose + tie)
    return win, lose, tie

def generate_winrate_all():
    """
    Generate winrate for all unique reduced 5-card combinations and store in a dictionary.
    The index is a pair calculated by vector2pair function.

    The resulting dictionary is serialized and saved using pickle.
    """
    start_time = time.time()
    unique_combos = generate_all_hold_community_reduced_combinations()
    with open('uniquecombo_7.pkl', 'wb') as f:
        pickle.dump(unique_combos, f)
    print("Winrate data has been saved to 'uniquecombo_7.pkl'.")    
    print(f"Total unique reduced 5-card combinations to evaluate: {len(unique_combos)}")
    
    winrate_dict = {}
    evaluated = 0
    start_eval = time.time()

    for hand in unique_combos:
        holeplayer = hand[:2]
        community = hand[2:]
        win, lose, tie = generate_winrate_given_holeplayer_and_community(community, holeplayer)
        encoded_key = tuple2str(hand)
        winrate_dict[encoded_key] = tuple(win, lose, tie)
        evaluated += 1

        # Optional: Print progress every 100,000 hands
        if evaluated % 100 == 0:
            elapsed = time.time() - start_eval
            print(f"Evaluated {evaluated} hands in {elapsed:.2f} seconds...")
    
    end_eval = time.time()
    print(f"Completed evaluating {evaluated} hands in {end_eval - start_eval:.2f} seconds.")
    
    # Serialize the winrate dictionary using pickle
    with open('db7s.pkl', 'wb') as f:
        pickle.dump(winrate_dict, f)
    print("Winrate data has been saved to 'db7s.pkl'.")
    
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds.")   

def tuple2str(hand_tuple): 
    """"
    Args:
        hand_tuple (tuple): A tuple of 6 (rank, suit) tuples.
                        - rank (int): Rank of the card.
                        - suit (int): 0 for suited, 1 for nonsuited.
    Returns:
        str: 
            - If more than 4 cards have suit=0, returns a 12-character string with ranks and suit indicators.
            - Otherwise, returns a 6-character string with just ranks.
    """
    RANK_REVERSE_MAP = {
        14: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}

    # Validate input
    if not isinstance(hand_tuple, tuple) or len(hand_tuple) != 7:
        raise ValueError("Input must be a tuple of 7 (rank, suit) tuples.")

    for card in hand_tuple:
        if not isinstance(card, tuple) or len(card) != 2:
            raise ValueError("Each card must be a tuple of (rank, suit).")
        rank, suit = card
        if rank not in RANK_REVERSE_MAP:
            raise ValueError(f"Invalid rank value: {rank}")
        if suit not in (0, 1):
            raise ValueError(f"Invalid suit value: {suit}")

    # Count the number of suited cards
    suit_zero_count = sum(1 for card in hand_tuple if card[1] == 0)
    commun_zero_count = sum(1 for card in hand_tuple[2:] if card[1] == 0)

    if suit_zero_count >= 5 or commun_zero_count >= 4:
        # Create a 12-character string with ranks and suit indicators
        hand_str = ''.join([
            f"{RANK_REVERSE_MAP[card[0]]}{'s' if card[1] == 0 else 'o'}"
            for card in hand_tuple
        ])
    else:
        # Create a 6-character string with just ranks
        hand_str = ''.join([
            RANK_REVERSE_MAP[card[0]]
            for card in hand_tuple
        ])

    return hand_str


def main():
    """
    Main function to generate and analyze all unique reduced 5-card combinations.
    Also demonstrates generating community and opponent hole card combinations.
    """
    start = time.time()

    generate_winrate_all()
    
    end = time.time()
    print(f"\nScript completed in {end - start:.2f} seconds.")
    
if __name__ == "__main__":
    main()