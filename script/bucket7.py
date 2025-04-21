# same as bucket6.py, but 7 card case

import time
from collections import defaultdict
import pickle
import itertools
import collections
import sys
RANK2NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20}
NUM2RANK = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E",  6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T"}
NUM_BUCKETS = 20
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
        
def max_difference_score(data_dict, total_games=990):
    """
    Calculate the maximum difference of the scores based on wins and ties in the dictionary.
    
    Each win counts as 1 and each tie counts as 0.5 for the final score.
    
    Args:
        data_dict (dict): A dictionary where the key is a string and the value is a pair (tuple) of (wins, losses).
        total_games (int): Total number of games per entry. Default is 96,525.
        
    Returns:
        tuple: (min_score, max_score, max_difference)
    """
    if not data_dict:
        return (0, 0, 0)  # Return zeros if the dictionary is empty

    scores = []
    for key, value in data_dict.items():
        if (
            isinstance(value, (list, tuple)) 
            and len(value) >= 2 
            and isinstance(value[0], (int, float)) 
            and isinstance(value[1], (int, float))
        ):
            wins, losses = value[:2]
            ties = total_games - wins - losses
            if ties < 0:
                print(f"Warning: Ties calculated as negative for key '{key}'. Setting ties to 0.")
                ties = 0
            score = wins * 1 + ties * 0.8
            weighted_score = non_linear_bucket_score(score)
            scores.append(weighted_score)
        else:
            print(f"Invalid data format for key '{key}'. Expected a pair of (wins, losses).")
    
    if not scores:
        return (0, 0, 0)  # Return zeros if no valid scores are found

    max_score = max(scores)
    min_score = min(scores)
    max_diff = max_score - min_score

    print(f"Minimum score: {min_score}")
    print(f"Maximum score: {max_score}")
    print(f"Maximum difference: {max_diff}")

    return (min_score, max_score, max_diff)

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
    comm_zero_count = sum(1 for card in hand_tuple[2:] if card[1] == 0)

    if suit_zero_count >= 5 or comm_zero_count >= 3:
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

def non_linear_bucket_score(score, midpoint=495):
    """
    Transform a score using a non-linear function to emphasize extremes.
    
    For scores >= midpoint: f(x) = (((1/20) * (x-midpoint))^(1.75) + midpoint)
    For scores < midpoint: g(x) = 990 - f(990-x)
    """
    total_range = 990  # Total range of scores
    
    def f(x):
        return ((1/10) * (x - midpoint)) ** 1.75 + midpoint + 0.5 * (x - midpoint)
    
    if score >= midpoint:
        return f(score)
    else:
        return total_range - f(total_range - score)
    # return score * score / 990  


def assign_bucket_ties(wins, ties, min_val, bucket_size, num_buckets=NUM_BUCKETS):
    """
    Assign a bucket number to a given value based on the bucket size.
    
    Args:
        value (float): The first element value of a pair.
        min_val (float): The minimum value among all first elements.
        bucket_size (float): The size of each bucket.
        
    Returns:
        int: The bucket number (1 to 9).
    """
    if bucket_size == 0:
        return 1  # If all values are the same, assign to bucket 1
    
    score = wins + 0.8 * ties
    weighted_score = non_linear_bucket_score(score)

    bucket_num = int((weighted_score - min_val) / bucket_size) + 1
    if bucket_num == num_buckets + 1:
        bucket_num = num_buckets
    if bucket_num > num_buckets:
        print(f"Warning: Bucket number {bucket_num} exceeds the maximum of {num_buckets}.")
        sys.exit(-1)

    return bucket_num


def create_buckets(data_dict, min_val, max_val, num_buckets=NUM_BUCKETS):
    """
    Divide the first element values into buckets and calculate bucket probabilities.
    
    Args:
        data_dict (dict): The original data dictionary.
        min_val (float): The minimum first element value.
        max_val (float): The maximum first element value.
        num_buckets (int): Number of buckets to create.
        
    Returns:
        tuple: (new_data_dict, bucket_counts, bucket_probabilities)
    """
    start = time.time()
    max_diff = max_val - min_val
    bucket_size = max_diff / num_buckets if max_diff > 0 else 1  # Avoid division by zero
    print(f"Bucket size: {bucket_size}")
    bucket_counts = {i: 0 for i in range(1, num_buckets + 1)}
    bucket_lens = {i: 0 for i in range(1, num_buckets + 1)}
    bucket_scores_sum = {i: 0.0 for i in range(1, num_buckets + 1)}  # Initialize score sums
    new_data_dict = {}
    saved_data = {}
    ite = 0
    seven_card = ()
    deck = generate_full_deck()
    hole_combos = itertools.combinations(deck, 2)
    for hole in hole_combos:
        remaining_deck = set(deck) - set(hole)
        community_combos = itertools.combinations(remaining_deck, 5)
        for community in community_combos:
            seven_card = tuple(hole + community)
            ite += 1
            if ite % 1000000 == 0:
                print(f"Processing {ite}th combination...")
                end_tem = time.time()
                print(f"Time taken: {end_tem - start:.2f} seconds")
            tem = reduce_combination7_preserve_order(seven_card)
            key = tuple2str(tem)
            value = data_dict[key]

            win = value[0]  # Assuming the first element is 'win'
            lose = value[1] if len(value) > 1 else 0  # Assuming the second element is 'lose'
            tie = 990 - win - lose

            bucket_num = assign_bucket_ties(win, 990 - win - lose, min_val, bucket_size)
            bucket_counts[bucket_num] += 1
        
            winrate = win / 990
            tierate = tie / 990
            score = 1 * winrate + 0.8 * tierate
            weighted_score = non_linear_bucket_score(score)
            bucket_scores_sum[bucket_num] += score
            bucket_lens[bucket_num] += 1

            # Placeholder for probability; will calculate after counting
            # new_data_dict[six_card] = {
            #     'win': win,
            #     'lose': lose,
            #     'bucket_number': bucket_num,
            #     'probability': 0,  # To be updated later
            #     'score': score
            # }
        
            # if key not in saved_data:
            #     saved_data[key] = {
            #         'win': win,
            #         'lose': lose,
            #         'bucket_number': bucket_num,
            #         'probability': 0,  # To be updated later
            #         'score': score
            #     }

    total_elements = sum(bucket_counts.values())
    print(f"Total elements assigned to buckets: {ite}")
    print(f"Total elements in saved_data: {len(saved_data)}")

    # Calculate probabilities
    bucket_probabilities = {}
    for bucket_num, count in bucket_counts.items():
        probability = count / total_elements if total_elements > 0 else 0
        bucket_probabilities[bucket_num] = probability

    # Update probabilities in the new_data_dict
    for key, details in new_data_dict.items():
        bucket_num = details['bucket_number']
        details['probability'] = bucket_probabilities.get(bucket_num, 0)

    bucket_avg_scores = {i: (bucket_scores_sum[i] / bucket_lens[i]) if bucket_counts[i] > 0 else 0 for i in bucket_counts}
    end = time.time()
    print(f"bucketing Time taken: {end - start:.2f} seconds")

    return (new_data_dict, saved_data, bucket_counts, bucket_probabilities, bucket_avg_scores)

# ...existing code...

def assign_data_to_buckets(data_dict, min_val, max_val, bucket_probs, num_buckets=NUM_BUCKETS):
    """
    Iterate through all elements in data_dict, calculate a score based on win, lose, and tie,
    assign the element to a bucket, and store the result in a new dictionary.

    The score is computed as: score = win - lose + 0.5 * tie.
        
    Returns:
        dict: A new dictionary mapping each index to a dictionary with fields "score", "index",
              and "bucket".
    """
    # Calculate bucket size. Avoid division by zero.
    score_range = max_val - min_val
    bucket_size = score_range / num_buckets if score_range > 0 else 1

    result_dict = {}
    for index, value in data_dict.items():
        # Extract values. Value may be a dict or tuple.
        win = value[0]
        lose = value[1]
        
        # Compute score. The formula can be adjusted as needed.
        tie = 990 - win - lose
        score = win + 0.8 * tie
        # Determine bucket.
        bucket = assign_bucket_ties(win, tie, min_val, bucket_size, num_buckets)

        result_dict[index] = {"score": score, "index": index, "bucket": bucket, "probability": bucket_probs[bucket]}
    
    return result_dict

def save_bucketing_data(file_path, data):
    """
    Save the bucketing data into a pickle file.
    
    Args:
        file_path (str): Path to the output pickle file.
        data (dict): The data dictionary to be pickled.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Successfully saved bucketing data to '{file_path}'.")
    except Exception as e:
        print(f"Error saving bucketing data: {e}")

def convert_to_simplified_bucket(input_path, output_path):
    """
    Convert the extended bucket dictionary to a simplified version with only bucket numbers.
    
    Args:
        input_path (str): Path to the input extended bucket pickle file.
        output_path (str): Path to output the simplified bucket pickle file.
    """
    # Load the extended bucket data
    try:
        with open(input_path, 'rb') as file:
            extended_data = pickle.load(file)
        print(f"Successfully loaded data from '{input_path}'.")
        print(f"Number of entries: {len(extended_data)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create simplified dictionary with only bucket numbers
    simplified_data = {}
    for key, value in extended_data.items():
        simplified_data[key] = NUM2RANK[value["bucket"]]
    
    # Save the simplified data
    try:
        with open(output_path, 'wb') as file:
            pickle.dump(simplified_data, file)
        print(f"Successfully saved simplified bucket data to '{output_path}'.")
        print(f"Number of entries in simplified data: {len(simplified_data)}")
    except Exception as e:
        print(f"Error saving simplified data: {e}")

def main():
    data = load_pickled_data('db7-new.pkl')
    min_val, max_val, max_diff = max_difference_score(data)
    new_data_dict, saved_data, bucket_counts, bucket_probabilities, bucket_aver_score = create_buckets(data, min_val, max_val)
    print("\nBucket Counts and Probabilities:")
    for bucket_num in bucket_counts.keys():
        count = bucket_counts[bucket_num]
        probability = bucket_probabilities[bucket_num]
        aver_score = bucket_aver_score[bucket_num]
        print(f"Bucket {bucket_num}: Count = {count}, Probability = {probability:.4f}, average score = {aver_score:.4f}")

    saved_data = assign_data_to_buckets(data, min_val, max_val, bucket_probabilities)
    output_pickle_path = 'bucket7_extended-ten.pkl'  # Path to the output pickle file
    save_bucketing_data(output_pickle_path, saved_data)
    convert_to_simplified_bucket('bucket7_extended-ten.pkl', 'bckt7s-ten.pkl')


if __name__ == '__main__':
    main()