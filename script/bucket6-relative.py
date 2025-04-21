# bucketing using the relative position of the score between min and max scores for each community combination
# for a 6-card hand, we first extract the community cards from the hand string, then we look up the min and max scores for that community combination
# min and max scores stand for the possible best and worst hands formed using this community cards.
# this approach can avoid the case where community cards are already strong enough while cfr still recognizes itself as leading position.
# for more details about the community card analysis, please refer to bckt6_helper.py

import pickle
import itertools
import time
from collections import defaultdict
import collections
import sys

# Constants
NUM_BUCKETS = 15  # Number of buckets to create
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, 
            "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 
            11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T"}

def load_community_analysis(file_path='community6_analysis.pkl'):
    """Load the community card analysis results."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully loaded community analysis from '{file_path}'.")
        print(f"Data type: {type(data)}")
        print(f"Number of community combinations: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading community analysis: {e}")
        return None

def load_db6_data(file_path='db6-new.pkl'):
    """Load the database of 6-card hands."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully loaded hand data from '{file_path}'.")
        print(f"Data type: {type(data)}")
        print(f"Number of hands: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading hand data: {e}")
        return None

def calculate_score(win, tie):
    """Calculate score as 1*win + 0.8*tie."""
    return win + 0.8 * tie

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
    Reduce a 6-card combination based on suit patterns.
    Args:
        combination (tuple): A tuple of 6 cards, each represented as (rank, suit).
    """
    # Count the occurrences of each suit in the combination
    suit_counts = collections.Counter(card[1] for card in combination)

    # Identify the most common suit and its count
    most_common_suit, count = suit_counts.most_common(1)[0]
    # Split the combination into hole cards and community cards
    last_four_com = combination[2:]
    suit_counts_com = collections.Counter(card[1] for card in last_four_com)
    most_common_suit_com, count_com = suit_counts_com.most_common(1)[0]
    
    if count >= 4:
        # More than three cards share the same suit
        reduced = [(card[0], 0 if card[1] == most_common_suit else 1) for card in combination]
    elif count_com >= 3:
        reduced = [(card[0], 0 if card[1] == most_common_suit_com else 1) for card in combination]
    else:
        # No suit has significant count; mark all as nonsuited (1)
        reduced = [(card[0], 1) for card in combination]

    first_two = reduced[:2]
    last_four = reduced[2:]
    first_two_sorted = sorted(first_two, key=lambda x: (x[0], -x[1]))
    last_four_sorted = sorted(last_four, key=lambda x: (x[0], -x[1]))

    # Combine the sorted first two and last four cards
    reduced_sorted = first_two_sorted + last_four_sorted

    return tuple(reduced_sorted)

def tuple2str(hand_tuple): 
    """Convert a 6-card hand tuple to a string representation."""
    RANK_REVERSE_MAP = {
        14: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'
    }

    # Validate input
    if not isinstance(hand_tuple, tuple) or len(hand_tuple) != 6:
        raise ValueError("Input must be a tuple of 6 (rank, suit) tuples.")

    for card in hand_tuple:
        if not isinstance(card, tuple) or len(card) != 2:
            raise ValueError("Each card must be a tuple of (rank, suit).")
        
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

def extract_community_from_key(hand_str):
    """
    Extract the community cards string from a db6 key string.
    
    If length is 6: Return the last 4 characters (community cards)
    If length is 12: First extract the last 8 chars, then:
        - If more than 2 characters at indexes 1,3,5,7 are 's', return all 8 chars
        - Otherwise, return just the ranks (indexes 0,2,4,6)
    
    Args:
        hand_str (str): The hand string from db6 key
        
    Returns:
        str: Community cards string
    """
    # Check the length of the string
    if len(hand_str) == 6:
        # Simply return the last 4 characters (community cards)
        return hand_str[2:]
    
    elif len(hand_str) == 12:
        # Extract the last 8 characters (community cards with suit indicators)
        comm_chars = hand_str[4:]
        
        # Count how many 's' characters appear at positions 1,3,5,7
        s_count = 0
        for i in [1, 3, 5, 7]:
            if comm_chars[i] == 's':
                s_count += 1
        
        # If more than 2 's', return the full 8 characters
        if s_count >= 2:
            return comm_chars
        # Otherwise return just the ranks
        else:
            return comm_chars[0] + comm_chars[2] + comm_chars[4] + comm_chars[6]
    
    else:
        raise ValueError(f"Invalid hand string length: {len(hand_str)}. Expected 6 or 12.")


def assign_relative_bucket(score, min_score, max_score, num_buckets=NUM_BUCKETS):
    """
    Assign a bucket based on where the score falls between min and max.
    
    Args:
        score: The score to bucket
        min_score: The minimum score for this community combination
        max_score: The maximum score for this community combination
        num_buckets: Number of buckets to divide range into
        
    Returns:
        int: Bucket number from 1 to num_buckets
    """
    if max_score == min_score:
        print(f"Warning: min_score and max_score are the same ({min_score}). Assigning to middle bucket.")
        return num_buckets // 2  # Middle bucket if all scores are the same
    
    # Calculate relative position (0.0 to 1.0)
    relative_position = (score - min_score) / (max_score - min_score)
    
    # Convert to bucket number (1 to num_buckets)
    bucket_num = int(relative_position * num_buckets) + 1
    
    # Handle edge cases
    if bucket_num > num_buckets:
        bucket_num = num_buckets
    if bucket_num < 1:
        bucket_num = 1
        
    return bucket_num

def create_relative_buckets(db_data, community_analysis):
    """
    Create buckets for hands based on their relative position between 
    min and max scores for their community combination.
    
    Args:
        db_data (dict): The hand database
        community_analysis (dict): Dictionary of community card min/max scores
        
    Returns:
        dict: New dictionary with hands mapped to their bucket numbers
    """
    start_time = time.time()
    result_dict = {}
    bucket_counts = {i: 0 for i in range(1, NUM_BUCKETS + 1)}
    
    deck = generate_full_deck()
    total_hands = 0
    processed_hands = 0
    
    print("Starting relative bucketing...")
    
    # Precompute all valid 6-card hands and their community strings
    for hole in itertools.combinations(deck, 2):
        remaining_deck = [card for card in deck if card not in hole]
        
        for community in itertools.combinations(remaining_deck, 4):
            total_hands += 1
            
            # Get the community card string
            community_str = community_card_str(community)
            
            if community_str not in community_analysis:
                print(f"Community string {community_str} not found in community analysis.")
                sys.exit(-1)
            
            min_score, max_score = community_analysis[community_str]
            
            
            # Form the 6-card hand
            six_card = hole + community
            reduced_hand = reduce_combination6_preserve_order(six_card)
            hand_str = tuple2str(reduced_hand)
            processed_hands += 1
            # Look up hand in database
            if hand_str in db_data:
                value = db_data[hand_str]
                win = value[0]
                loss = value[1]
                tie = 45540 - win - loss  # Assuming total games is 45540
                score = calculate_score(win, tie)
                
                # Assign relative bucket
                bucket = assign_relative_bucket(score, min_score, max_score)
                bucket_counts[bucket] += 1
            else:
                print(f"Hand {hand_str} not found in database.")
                sys.exit(-1)
            
            # Print progress
            if processed_hands % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {processed_hands} hands in {elapsed:.2f} seconds.")
    
    # Calculate bucket probabilities
    total_bucketed = sum(bucket_counts.values())
    bucket_probabilities = {i: count/total_bucketed for i, count in bucket_counts.items()}
    
    print(f"\nBucketing summary:")
    print(f"Total hands examined: {total_hands}")
    print(f"Total hands bucketed: {processed_hands}")
    
    print("\nBucket distribution:")
    for bucket, count in bucket_counts.items():
        probability = bucket_probabilities[bucket]
        print(f"Bucket {bucket}: {count} hands ({probability:.4f})")
    
    return result_dict, bucket_counts, bucket_probabilities

def assign_data_to_buckets(data_dict, bucket_probs, num_buckets=NUM_BUCKETS):
    """
    Iterate through all elements in data_dict, calculate a score based on win, lose, and tie,
    assign the element to a bucket, and store the result in a new dictionary.

    The score is computed as: score = win - lose + 0.5 * tie.
        
    Returns:
        dict: A new dictionary mapping each index to a dictionary with fields "score", "index",
              and "bucket".
    """
    result_dict = {}
    for index, value in data_dict.items():
        # Extract values. Value may be a dict or tuple.
        win = value[0]
        lose = value[1]
        
        # Compute score. The formula can be adjusted as needed.
        tie = 45540 - win - lose
        score = win + 0.8 * tie
        # Determine bucket.
        
        comstr = extract_community_from_key(index)
        min_score, max_score = bucket_probs[comstr]
        
        bucket = assign_relative_bucket(score, min_score, max_score, num_buckets)

        result_dict[index] = NUM2RANK[bucket]
    
    return result_dict

def create_simplified_buckets(extended_data):
    """
    Convert extended bucket data to simplified format with just bucket letters.
    
    Args:
        extended_data (dict): The extended bucket data
        
    Returns:
        dict: Simplified dictionary mapping hands to bucket letters
    """
    simplified_data = {}
    for key, value in extended_data.items():
        bucket_num = value["bucket"]
        simplified_data[key] = NUM2RANK[bucket_num]
    
    return simplified_data

def save_bucket_data(data, filename):
    """Save bucket data to a pickle file."""
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Successfully saved data to '{filename}'.")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    # Load necessary data
    print("Loading community analysis data...")
    community_analysis = load_community_analysis()
    if not community_analysis:
        return
    
    print("Loading hand database...")
    db_data = load_db6_data()
    if not db_data:
        return
    
    # Create relative buckets
    print("Creating relative buckets...")
    extended_buckets, bucket_counts, bucket_probs = create_relative_buckets(db_data, community_analysis)
    
    simplified_buckets = assign_data_to_buckets(db_data, community_analysis)
    
    # Save results
    print("Saving results...")
    save_bucket_data(simplified_buckets, 'relative_buckets6_simplified.pkl')
    
    print("\nAll done! Bucket data has been saved.")


if __name__ == "__main__":
    main()