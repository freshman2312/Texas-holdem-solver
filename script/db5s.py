import itertools
import collections
import pickle
import time
from db7s import *
from concurrent.futures import *

# Define Hand Types (if needed for further evaluation)
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6

def generate_full_deck():
    """
    Generate a standard deck of 52 cards.
    Each card is represented as a tuple (rank, suit), where:
        rank: 1-13 (Ace to King)
        suit: 0-3 (e.g., 0: Hearts, 1: Diamonds, 2: Clubs, 3: Spades)
    """
    ranks = range(2, 15)  # 1 to 13
    suits = range(0, 4)   # 0 to 3
    deck = [(rank, suit) for suit in suits for rank in ranks]
    return deck


def reduce_combination5_preserve_order(combination):
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
    if count >= 3:
        # More than three cards share the same suit
        # Mark cards with the most_common_suit as suited (0) and the rest as nonsuited (1)
        reduced = [(card[0], 0 if card[1] == most_common_suit else 1) for card in combination]
    elif count_com == 2:
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
    Generate all unique reduced 5-card combinations by combining 2-card hold and 3-card community.
    Reduction Rule:
        - If more than three cards share the same suit in the 5-card combination,
          mark the rest as nonsuited.
        - Else, mark all as nonsuited.
    
    Returns:
        set: A set of tuples, each representing a unique reduced 5-card combination.
             Each combination is a tuple of 5 (rank, suit_flag) tuples.
    """
    deck = generate_full_deck()
    unique_reduced_combinations = set()
    total_combinations = 0
    hold_combos = itertools.combinations(deck, 2)
    
    for hold in hold_combos:
        remaining_deck = set(deck) - set(hold)
        community_combos = itertools.combinations(remaining_deck, 3)
        for community in community_combos:
            five_card = tuple(hold + community)
            reduced = reduce_combination5_preserve_order(five_card)
            unique_reduced_combinations.add(reduced)
            total_combinations += 1
            if total_combinations % 1000000 == 0:
                print(f"Processed {total_combinations} 5-card combinations...")
                print(f"Unique reduced 5-card combinations: {len(unique_reduced_combinations)}")
                print(f"Current Combination: {reduced}")
                print(f"Total Combinations: {five_card}")
    
    print(f"Total original 5-card combinations processed: {total_combinations}")
    print(f"Total unique reduced 5-card combinations: {len(unique_reduced_combinations)}")
    return unique_reduced_combinations


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

    Raises:
        ValueError: If the input is not properly formatted or contains invalid values.
    """
    RANK_REVERSE_MAP = {
        14: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'
    }

    # Validate input
    if not isinstance(hand_tuple, tuple) or len(hand_tuple) != 5:
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
    comm_zero_count = sum(1 for card in hand_tuple[2:] if card[1] == 0)

    if suit_zero_count >= 3 or comm_zero_count >= 2:
        # Create a 10-character string with ranks and suit indicators
        hand_str = ''.join([
            f"{RANK_REVERSE_MAP[card[0]]}{'s' if card[1] == 0 else 'o'}"
            for card in hand_tuple
        ])
    else:
        # Create a 5-character string with just ranks
        hand_str = ''.join([
            RANK_REVERSE_MAP[card[0]]
            for card in hand_tuple
        ])

    return hand_str


def generate_all_hold_opponent_combinations(hand):
    """
    Given a full deck and a 5-card hand, remove these 5 cards from the deck to form a 47-card deck.
    Then generate all 2-card combinations from this 47-card deck as community cards,
    and all 2-card combinations from the remaining 45-card deck as opponent hole cards.
    
    Returns:
        tuple: A tuple containing two sets:
               - community_combinations: Set of tuples representing community card combinations.
               - opponent_hole_combinations: Set of tuples representing opponent hole card combinations.
    """
    
    full_deck = generate_full_deck()
    remaining_deck = set(full_deck) - set(hand)

    community_combos = set(itertools.combinations(remaining_deck, 2))
    
    opponent_hole_combos = set()
    for community in community_combos:
        available_for_opponent = remaining_deck - set(community)
        opponent_combos = itertools.combinations(available_for_opponent, 2)
        opponent_hole_combos.update(opponent_combos)
    return community_combos, opponent_hole_combos


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
    community_combos, opponent_hole_combos = generate_all_hold_opponent_combinations(hand1)
    for community2 in community_combos:
        for holeplayer2 in opponent_hole_combos:
            hand2tem = tuple(community2 + tuple(holeplayer2) + tuple(community1))
            hand1tem = tuple(hand1 + community2)
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
            if cnt % iteration == 0:
                winrate = float(win) / (win + lose + tie)
                return winrate
            
    winrate = float(win) / (win + lose + tie)
    return winrate
            
def generate_winrate_all():
    """
    Generate winrate for all unique reduced 5-card combinations and store in a dictionary.
    The index is a pair calculated by vector2pair function.

    The resulting dictionary is serialized and saved using pickle.
    """
    start_time = time.time()
    unique_combos = generate_all_hold_community_reduced_combinations()
    print(f"Total unique reduced 5-card combinations to evaluate: {len(unique_combos)}")
    
    winrate_dict = {}
    evaluated = 0
    start_eval = time.time()

    for hand in unique_combos:
        holeplayer = hand[:2]
        community = hand[2:]
        winrate = generate_winrate_given_holeplayer_and_community(community, holeplayer)
        encoded_key = tuple2str(hand)
        winrate_dict[encoded_key] = winrate
        evaluated += 1

        # Optional: Print progress every 100,000 hands
        if evaluated % 50 == 0:
            elapsed = time.time() - start_eval
            print(f"Evaluated {evaluated} hands in {elapsed:.2f} seconds...")
    
    end_eval = time.time()
    print(f"Completed evaluating {evaluated} hands in {end_eval - start_eval:.2f} seconds.")
    
    # Serialize the winrate dictionary using pickle
    with open('db5s.pkl', 'wb') as f:
        pickle.dump(winrate_dict, f)
    print("Winrate data has been saved to 'db5s.pkl'.")
    
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds.")    
    


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