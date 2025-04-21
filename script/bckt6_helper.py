# calculate the possible best and worst hands given a set of community cards.
# will be used in relative buycketing process: bucket6-relative.py
# the rule of key str is same as that in db file.

import pickle
import itertools
import time
from collections import defaultdict
import collections
import random

def tuple2str(hand_tuple): 
    """"
    Args:
        hand_tuple (tuple): A tuple of 6 (rank, suit) tuples.
                        - rank (int): Rank of the card.
                        - suit (int): 0 for suited, 1 for nonsuited.

    Returns:
        str: 
            - If more than 3 cards share the same suit, returns a string with ranks and suit indicators.
            - Otherwise, returns a string with just ranks.

    Raises:
        ValueError: If the input is not properly formatted or contains invalid values.
    """
    RANK_REVERSE_MAP = {
        14: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'
    }

    # Validate input
    if not isinstance(hand_tuple, tuple) or len(hand_tuple) != 6:
        raise ValueError("Input must be a tuple of 6 (rank, suit) tuples.")

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

    if suit_zero_count >= 4 or comm_zero_count >= 2:
        # Create a string with ranks and suit indicators
        hand_str = ''.join([
            f"{RANK_REVERSE_MAP[card[0]]}{'s' if card[1] == 0 else 'o'}"
            for card in hand_tuple
        ])
    else:
        # Create a string with just ranks
        hand_str = ''.join([
            RANK_REVERSE_MAP[card[0]]
            for card in hand_tuple
        ])

    return hand_str

def generate_full_deck():
    """
    Generate a standard deck of 52 cards.
    Each card is represented as a tuple (rank, suit), where:
        rank: 2-14 (2 to Ace)
        suit: 0-3 (e.g., 0: Hearts, 1: Diamonds, 2: Clubs, 3: Spades)
    """
    ranks = range(2, 15)  # 2 to 14 (Ace)
    suits = range(0, 4)   # 0 to 3
    deck = [(rank, suit) for suit in suits for rank in ranks]
    return deck

def reduce_combination6_preserve_order(combination):
    """
    Reduce a 6-card combination based on the following rule:
        - If more than three cards share the same suit, mark them as suited (0),
          and mark the rest as offsuited (1).
        - Else, mark all as offsuited (1).

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
    # Split the reduced combination into first two and last four cards
    last_four_com = combination[2:]
    suit_counts_com = collections.Counter(card[1] for card in last_four_com)
    most_common_suit_com, count_com = suit_counts_com.most_common(1)[0]
    
    if count >= 4:
        # More than three cards share the same suit
        # Mark cards with the most_common_suit as suited (0) and the rest as nonsuited (1)
        reduced = [(card[0], 0 if card[1] == most_common_suit else 1) for card in combination]
    elif count_com >= 3:
        reduced = [(card[0], 0 if card[1] == most_common_suit_com else 1) for card in combination]
    else:
        # No suit has more than three cards; mark all as nonsuited (1)
        reduced = [(card[0], 1) for card in combination]

    first_two = reduced[:2]
    last_four = reduced[2:]
    first_two_sorted = sorted(first_two, key=lambda x: (x[0], -x[1]))
    last_four_sorted = sorted(last_four, key=lambda x: (x[0], -x[1]))

    # Combine the sorted first two and last four cards
    reduced_sorted = first_two_sorted + last_four_sorted

    return tuple(reduced_sorted)

def load_db6_data(file_path='db6-new.pkl'):
    """Load the database of 6-card hands."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully loaded data from '{file_path}'.")
        print(f"Data type: {type(data)}")
        print(f"Number of entries: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_score(win, tie):
    """Calculate score as 1*win + 0.8*tie."""
    return win + 0.8 * tie

def community_card_str(community):
    """
    Convert a 4-card community tuple to a readable string.
    First mark each card as 's' if it belongs to the majority suit (>=2 cards),
    otherwise 'o'. Then sort by rank, and for ties sort 'o' before 's'.
    """
    from collections import Counter
    
    RANK_MAP = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
        7: '7', 8: '8', 9: '9', 10: 'T',
        11: 'J', 12: 'Q', 13: 'K', 14: 'A'
    }

    # Count occurrences of each suit
    suit_counts = Counter(s for _, s in community)
    most_common_suit, count = suit_counts.most_common(1)[0]

    # Mark each card with 's' or 'o'
    marked = []
    for rank, suit in community:
        flag = 's' if suit == most_common_suit and count >= 2 else 'o'
        marked.append((rank, flag))

    # Sort by rank, then ensure 'o' comes before 's' for same rank
    sorted_cards = sorted(marked, key=lambda x: (x[0], x[1]))

    # Build result string
    return ''.join(f"{RANK_MAP[r]}{f}" for r, f in sorted_cards)

def analyze_community_combinations(db_data):
    """
    For each 4-card community combination, find the highest and lowest winrates
    from all possible hole card combinations.
    
    Returns a dictionary with community card strings as keys and (min_score, max_score) as values.
    """
    start_time = time.time()
    result = {}
    deck = generate_full_deck()
    community_combos = itertools.combinations(deck, 4)  # 4 cards for community in 6-card Texas Hold'em
    total_communities = 0
    
    print("Starting community card analysis...")
    
    for community in community_combos:
        community_str = community_card_str(community)
        total_communities += 1
        
        remaining_deck = [card for card in deck if card not in community]
        hole_combos = itertools.combinations(remaining_deck, 2)
        
        min_score = float('inf')
        max_score = float('-inf')
        min_hand = None
        max_hand = None
        
        if community_str not in result:
            for hole in hole_combos:
                six_card = hole + community  # 2 hole cards + 4 community cards
                reduced_hand = reduce_combination6_preserve_order(six_card)
                hand_str = tuple2str(reduced_hand)
                
                if hand_str in db_data:
                    value = db_data[hand_str]
                    win = value[0]
                    loss = value[1] 
                    tie = 45540 - win - loss  # Assuming total games is 990
                    score = calculate_score(win, tie)
                    
                    if score < min_score:
                        min_score = score
                        min_hand = hole
                    if score > max_score:
                        max_score = score
                        max_hand = hole
                else:
                    print(f"Warning: Hand {hand_str} not found in database")
            
            if min_score != float('inf') and max_score != float('-inf'):
                result[community_str] = (min_score, max_score)
        else:
            min_score, max_score = result[community_str]
        
        if total_communities % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {total_communities} communities in {elapsed:.2f} seconds.")
            print(f"Current community: {community_str}, Min: {min_score:.2f}, Max: {max_score:.2f}")
            if min_hand and max_hand:
                print(f"Min hand: {hole_to_str(min_hand)}, Max hand: {hole_to_str(max_hand)}")
    
    print(f"Analysis complete. Processed {total_communities} community card combinations.")
    return result

def hole_to_str(hole):
    """Convert a hole card tuple to a readable string."""
    RANK_MAP = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
        10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
    }
    SUIT_MAP = {0: 'h', 1: 'd', 2: 'c', 3: 's'}
    
    return ''.join([f"{RANK_MAP[rank]}{SUIT_MAP[suit]}" for rank, suit in hole])

def save_results(results, filename='community6_analysis.pkl'):
    """Save the analysis results to a pickle file."""
    try:
        with open(filename, 'wb') as file:
            pickle.dump(results, file)
        print(f"Successfully saved results to '{filename}'.")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    print("Loading database...")
    db_data = load_db6_data()
    if not db_data:
        return
    
    print("Analyzing community combinations...")
    results = analyze_community_combinations(db_data)
    
    # Print some statistics
    print("\nAnalysis summary:")
    print(f"Total community combinations analyzed: {len(results)}")
    
    # Find the community with the largest spread (max - min)
    largest_spread = 0
    largest_spread_community = None
    
    for community, (min_score, max_score) in results.items():
        spread = max_score - min_score
        if spread > largest_spread:
            largest_spread = spread
            largest_spread_community = community
    
    print(f"\nCommunity with largest spread: {largest_spread_community}")
    print(f"Min score: {results[largest_spread_community][0]:.2f}")
    print(f"Max score: {results[largest_spread_community][1]:.2f}")
    print(f"Spread: {largest_spread:.2f}")
    
    # Save results
    save_results(results)

if __name__ == "__main__":
    main()