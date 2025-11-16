import struct
import copy
import hashlib
import numpy as np
import math
from collections import Counter
import string
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# --- SHA-256 Constants ---
# Round constants K (first 32 bits of the fractional parts of the cube roots of the first 64 primes)
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]


# --- Bitwise Helper Functions (operating on 32-bit integers) ---
def ROTR(x, n):
    """Perform a right rotate operation."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def SHR(x, n):
    """Perform a right shift operation."""
    return (x >> n) & 0xFFFFFFFF

def Ch(x, y, z):
    """SHA-256 Ch function."""
    return ((x & y) ^ (~x & z)) & 0xFFFFFFFF

def Maj(x, y, z):
    """SHA-256 Maj function."""
    return ((x & y) ^ (x & z) ^ (y & z)) & 0xFFFFFFFF

def Sigma0(x):
    """SHA-256 Sigma0 function."""
    return (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22)) & 0xFFFFFFFF

def Sigma1(x):
    """SHA-256 Sigma1 function."""
    return (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25)) & 0xFFFFFFFF

def sigma0(x):
    """SHA-256 sigma0 function (lowercase)."""
    return (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3)) & 0xFFFFFFFF

def sigma1(x):
    """SHA-256 sigma1 function (lowercase)."""
    return (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10)) & 0xFFFFFFFF

def sha256_compress_with_states(message_block_bytes, h_prev):
    """
    Performs one block (64 rounds) of SHA-256 compression and records state evolution.

    Args:
        message_block_bytes (bytes): A 64-byte (512-bit) message block.
        h_prev (list[int]): A list of 8 32-bit integers representing the
                             previous hash state (H0-H7).

    Returns:
        list[list[int]]: A list of 64 lists. Each inner list contains 9 integers:
                         the state of the working variables [a, b, c, d, e, f, g, h]
                         *after* the round calculation, plus the message schedule word W[t]
                         used in that round. Returns None if input is invalid.
        list[int]: The final updated hash state H' (8 32-bit integers).
                   Returns None if input is invalid.
    """
    if len(message_block_bytes) != 64:
        # This should not happen if padding is correct, but good practice to check
        print(f"Error: Input block size is {len(message_block_bytes)} bytes, expected 64.")
        return None, None
    if len(h_prev) != 8:
        print("Error: Previous hash state must contain 8 32-bit integers.")
        return None, None

    # 1. Prepare the message schedule W[0...63]
    W = [0] * 64
    # Copy the first 16 words (32-bit integers) from the message block
    for t in range(16):
        # '>I' means big-endian unsigned integer (32-bit)
        W[t] = struct.unpack('>I', message_block_bytes[t*4:t*4+4])[0]

    # Extend the first 16 words into the remaining 48 words
    for t in range(16, 64):
        s0 = sigma0(W[t-15])
        s1 = sigma1(W[t-2])
        W[t] = (W[t-16] + s0 + W[t-7] + s1) & 0xFFFFFFFF

    # 2. Initialize the working variables a, b, c, d, e, f, g, h
    a, b, c, d, e, f, g, h = h_prev[:] # Make a copy

    # List to store the state evolution [a,b,c,d,e,f,g,h,Wt] for each round
    state_evolution = []

    # 3. Compression loop (64 rounds)
    for t in range(64):
        # Calculate round functions
        S1 = (Sigma1(e) + Ch(e, f, g) + h + K[t] + W[t]) & 0xFFFFFFFF
        S0 = (Sigma0(a) + Maj(a, b, c)) & 0xFFFFFFFF

        # Update working variables
        h = g
        g = f
        f = e
        e = (d + S1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (S1 + S0) & 0xFFFFFFFF

        # Store the state *after* this round's calculation + W[t]
        current_state = [a, b, c, d, e, f, g, h, W[t]]
        state_evolution.append(current_state)

    # 4. Compute the intermediate hash value H'
    h_new = [0] * 8
    h_new[0] = (h_prev[0] + a) & 0xFFFFFFFF
    h_new[1] = (h_prev[1] + b) & 0xFFFFFFFF
    h_new[2] = (h_prev[2] + c) & 0xFFFFFFFF
    h_new[3] = (h_prev[3] + d) & 0xFFFFFFFF
    h_new[4] = (h_prev[4] + e) & 0xFFFFFFFF
    h_new[5] = (h_prev[5] + f) & 0xFFFFFFFF
    h_new[6] = (h_prev[6] + g) & 0xFFFFFFFF
    h_new[7] = (h_prev[7] + h) & 0xFFFFFFFF

    return state_evolution, h_new

def GetEvolution(message_string):
    """
    Computes the SHA-256 hash and returns the full state evolution and final hash.
    Also returns the list of padded message blocks used.
    """
    try:
        message_bytes = message_string.encode('utf-8')
        original_length_bits = len(message_bytes) * 8
        message_bytes += b'\x80'
        padding_len = (56 - len(message_bytes) % 64 + 64) % 64
        message_bytes += b'\x00' * padding_len
        message_bytes += struct.pack('>Q', original_length_bits)

        if len(message_bytes) % 64 != 0:
            print("Error: Padded message length is not a multiple of 64 bytes.")
            return None, None, None

        current_h = H0[:]
        full_state_evolution = []
        padded_blocks = []

        for i in range(0, len(message_bytes), 64):
            current_block = message_bytes[i:i+64]
            padded_blocks.append(current_block) # Store the block
            block_state_evolution, next_h = sha256_compress_with_states(current_block, current_h)
            if block_state_evolution is None or next_h is None:
                print(f"Error processing block {i//64 + 1}.")
                return None, None, None
            full_state_evolution.append(block_state_evolution)
            current_h = next_h

        final_hash = current_h
        return full_state_evolution, final_hash, padded_blocks # Return blocks too

    except Exception as e:
        print(f"An error occurred in GetEvolution: {e}")
        return None, None, None
    
def calculate_shannon_entropy(data_sequence):
    """
    Calculates the Shannon entropy (in bits) for a sequence of discrete symbols.

    Args:
        data_sequence: An iterable (list, string, etc.) containing the data points (symbols).

    Returns:
        float: The calculated Shannon entropy in bits. Returns 0.0 if sequence is empty.
    """
    n = len(data_sequence)
    if n <= 0:
        return 0.0 # Entropy is 0 for empty sequence

    counts = Counter(data_sequence) # Count frequency of each unique symbol
    entropy = 0.0

    for symbol_count in counts.values():
        probability = symbol_count / n
        if probability > 0: # Avoid log2(0)
            entropy -= probability * math.log2(probability)

    return entropy

def EvolutionToBinary(complete_evolution):

  binary_evolution = []
  if complete_evolution: # Check if it's not None
      for block_evolution_int in complete_evolution:
          binary_block = []
          for round_state_int in block_evolution_int:
              # Convert each integer in the round_state to a 32-bit binary string
              binary_round_state = [f'{val:032b}' for val in round_state_int]
              binary_block.append(binary_round_state)
          binary_evolution.append(binary_block)

  return binary_evolution

def BinaryToLowRes(binary_evolution, rounds = 4):

  low_res_evolution = []

  for _ in range(rounds):
    if _ > 0:
      binary_evolution = low_res_evolution
      low_res_evolution = []
    if binary_evolution:
        for binary_block in binary_evolution:
            low_res_block = []
            for binary_round_state in binary_block:
                low_res_round_state = []
                for bin_str in binary_round_state:
                    bits = [float(b) for b in bin_str]
                    averages = []
                    if len(bits) > 1:
                        for i in range(len(bits) - 1):
                            avg = (bits[i] + bits[i+1]) / 2.0
                            averages.append(avg)
                    low_res_round_state.append(averages)
                low_res_block.append(low_res_round_state)
            low_res_evolution.append(low_res_block)

  return low_res_evolution

def average_adjacent(values):
    """Helper function to average adjacent numbers in a list."""
    if not isinstance(values, list) or len(values) < 2:
        return [] # Return empty list if input is not suitable or too short
    averages = []
    for i in range(len(values) - 1):
        try:
             # Ensure values are numeric for averaging
            avg = (float(values[i]) + float(values[i+1])) / 2.0
            averages.append(int(avg))
        except (ValueError, TypeError):
            print(f"Warning: Could not average non-numeric values: {values[i]}, {values[i+1]}")
            return [] # Return empty on error
    return averages

def GetLowResEvolution(int_evolution_data, max_rounds):
    """
    Takes SHA-256 state evolution data (integers) and returns its
    evolution over multiple rounds of adjacent value averaging.

    Args:
        int_evolution_data (list[list[list[int]]]): The state evolution data
            from GetEvolution (Blocks -> Rounds -> Vars[int]).
        max_rounds (int): The maximum number of averaging rounds to perform.

    Returns:
        list: A list where index `r` contains the state evolution data
              after `r` rounds of averaging.
              - Index 0: Binary representation (list[list[list[str]]]).
              - Index 1+: Averaged floats (list[list[list[list[float]]]]).
              Returns None if input is invalid.
    """
    if not int_evolution_data:
        print("Error: Input integer evolution data is empty or None.")
        return None

    all_resolutions = []

    # 1. Convert initial integer data to binary strings (Resolution 0)
    current_res_data_binary = []
    try:
        for block_int in int_evolution_data:
            block_bin = []
            for round_int in block_int:
                # Ensure round_int is a list/tuple of 9 integers
                if not isinstance(round_int, (list, tuple)) or len(round_int) != 9:
                     raise ValueError("Invalid structure: round_state is not a list of 9 ints.")
                round_bin = [f'{val:032b}' for val in round_int]
                block_bin.append(round_bin)
            current_res_data_binary.append(block_bin)
        all_resolutions.append(current_res_data_binary) # Add binary as level 0
    except Exception as e:
        print(f"Error converting integer evolution to binary: {e}")
        return None

    # 2. Apply averaging iteratively
    # Start with the binary data for the first averaging round (r=0)
    previous_res_data = current_res_data_binary

    for r in range(max_rounds):
        next_res_data_block_level = []
        possible_this_round = True

        for block_data in previous_res_data: # Iterate through blocks
            next_res_data_round_level = []
            for round_data in block_data: # Iterate through rounds
                next_res_data_var_level = []
                for var_data in round_data: # Iterate through variables (binary string or list of floats)
                    if r == 0: # First averaging round: input is binary string
                        values_to_avg = [int(b) for b in var_data]
                    else: # Subsequent rounds: input is list of floats
                        values_to_avg = var_data

                    if len(values_to_avg) < 2:
                        possible_this_round = False
                        # Append empty list or marker to maintain structure if needed,
                        # or break if strict structure is required. Let's append empty.
                        next_res_data_var_level.append([])
                        continue # Skip averaging for this variable

                    averaged_list = average_adjacent(values_to_avg)
                    next_res_data_var_level.append(averaged_list)

                next_res_data_round_level.append(next_res_data_var_level)
            next_res_data_block_level.append(next_res_data_round_level)

        if not possible_this_round:
            # print(f"Stopping averaging at round {r+1} - cannot lower resolution further for some variables.")
            break # Stop if any variable list became too short in this round

        all_resolutions.append(next_res_data_block_level)
        previous_res_data = next_res_data_block_level # Input for next iteration

    return all_resolutions

def sha256_compress(message_block_bytes, h_prev):
    """Internal function: Processes one block, returns next hash state."""
    if len(message_block_bytes) != 64: return None
    if len(h_prev) != 8: return None
    W = [0] * 64
    for t in range(16): W[t] = struct.unpack('>I', message_block_bytes[t*4:t*4+4])[0]
    for t in range(16, 64):
        s0 = sigma0(W[t-15]); s1 = sigma1(W[t-2])
        W[t] = (W[t-16] + s0 + W[t-7] + s1) & 0xFFFFFFFF
    a, b, c, d, e, f, g, h = h_prev[:]
    for t in range(64):
        S1 = (Sigma1(e) + Ch(e, f, g) + h + K[t] + W[t]) & 0xFFFFFFFF
        S0 = (Sigma0(a) + Maj(a, b, c)) & 0xFFFFFFFF
        h = g; g = f; f = e; e = (d + S1) & 0xFFFFFFFF
        d = c; c = b; b = a; a = (S1 + S0) & 0xFFFFFFFF
    h_new = [(h_prev[i] + val) & 0xFFFFFFFF for i, val in enumerate([a, b, c, d, e, f, g, h])]
    return h_new

def create_low_res_hash_evolution(final_hash_ints, max_rounds):
    """
    Creates a list representing the evolution of a hash digest
    through multiple rounds of adjacent-value averaging.

    Args:
        final_hash_ints (list[int]): The 8 x 32-bit integer final hash state.
        max_rounds (int): The number of averaging rounds to perform.

    Returns:
        list: A list where index `r` contains the representation after `r` rounds.
              Index 0 contains the original hash as 8 lists of 32 bits (0/1 ints).
              Subsequent indices contain 8 lists of 32-r floats.
              Returns None if input is invalid.
    """
    if not final_hash_ints or len(final_hash_ints) != 8:
        print("Error: Invalid final_hash_ints input.")
        return None

    all_resolutions = []
    # Initial state: Convert hash integers to list of lists of bits
    current_res_data = []
    for h_int in final_hash_ints:
        bin_str = f'{h_int:032b}'
        current_res_data.append([int(b) for b in bin_str])
    all_resolutions.append(current_res_data) # Add resolution 0 (binary bits)

    # Apply averaging iteratively
    for r in range(max_rounds):
        previous_res_data = all_resolutions[-1] # Get data from previous round
        next_res_data = []
        possible_this_round = True
        for var_list in previous_res_data: # Iterate through each of the 8 variables
            if len(var_list) < 2: # Cannot average if less than 2 values
                 possible_this_round = False
                 next_res_data.append([]) # Append empty list to maintain structure
                 continue # Skip averaging for this variable

            averaged_list = average_adjacent(var_list)
            next_res_data.append(averaged_list)

        if not possible_this_round:
            # print(f"Stopping averaging at round {r+1} - cannot lower resolution further for some variables.")
            break # Stop if any variable list became too short in this round
        all_resolutions.append(next_res_data)

    return all_resolutions

def GetFinalHashAndBlocks(message_string):
    """
    Computes the final SHA-256 hash digest and returns the list of padded blocks.

    Args:
        message_string (str): The input message string.

    Returns:
        list[int]: Final hash digest (8 32-bit integers). None if error.
        list[bytes]: List of 64-byte padded message blocks. None if error.
    """
    try:
        message_bytes = message_string.encode('utf-8')
        original_length_bits = len(message_bytes) * 8
        message_bytes += b'\x80'
        padding_len = (56 - len(message_bytes) % 64 + 64) % 64
        message_bytes += b'\x00' * padding_len
        message_bytes += struct.pack('>Q', original_length_bits)

        if len(message_bytes) % 64 != 0:
            print("Error: Padded message length is not a multiple of 64 bytes.")
            return None, None

        current_h = H0[:]
        padded_blocks = []
        for i in range(0, len(message_bytes), 64):
            current_block = message_bytes[i:i+64]
            padded_blocks.append(current_block) # Store the block
            next_h = sha256_compress(current_block, current_h)
            if next_h is None:
                print(f"Error processing block {i//64 + 1}.")
                return None, None
            current_h = next_h
        return current_h, padded_blocks
    except Exception as e:
        print(f"An error occurred in GetFinalHashAndBlocks: {e}")
        return None, None

def ReduceMessageBlocksToSingleValues(padded_blocks):
    """
    Takes padded message blocks and iteratively averages the bits
    of each 32-bit word within each block until a single value remains per word.

    Args:
        padded_blocks (list[bytes]): List of 64-byte padded message blocks.

    Returns:
        list[list[float]]: A list where each inner list contains 16 floats,
                           representing the final averaged value for each word
                           of the corresponding input block. Returns None on error.
    """
    if not padded_blocks:
        print("Error: No padded blocks provided.")
        return None

    final_block_averages = []
    num_words_per_block = 16

    for block_bytes in padded_blocks:
        if len(block_bytes) != 64:
            print(f"Error: Block size is {len(block_bytes)}, expected 64.")
            return None

        block_word_averages = []
        # Process each 32-bit word (16 words per block)
        for i in range(num_words_per_block):
            # Unpack word as big-endian unsigned integer
            word_int = struct.unpack('>I', block_bytes[i*4 : i*4+4])[0]
            # Convert word to list of bits
            bin_str = f'{word_int:032b}'
            current_values = [int(b) for b in bin_str]

            # Iteratively average until only one value remains
            while len(current_values) > 1:
                current_values = average_adjacent(current_values)
                if not current_values and len(current_values) != 1:
                    print(f"Error during averaging for word {i} in block.")
                    block_word_averages.append(None)
                    break # Stop averaging for this word

            # Check and store the final single value
            if len(current_values) == 1:
                block_word_averages.append(current_values[0])
            elif not block_word_averages or block_word_averages[-1] is not None:
                 print(f"Warning: Did not reduce word {i} to single value.")
                 block_word_averages.append(None)

        # Check if all words in the block were processed successfully
        if len(block_word_averages) == num_words_per_block and all(v is not None for v in block_word_averages):
             final_block_averages.append(block_word_averages)
        else:
             print("Error: Failed to reduce all words in a block to single values.")
             # Decide how to handle partial failure - return None or partial results?
             # Returning None for the whole process if any part fails is safer.
             return None

    return final_block_averages

def prepare_ml_data_v3(message_string, hash_avg_rounds=16):
    """
    Generates one data point (X, Y) for the revised ML task.
    X = Flattened final low-resolution representation of the hash after k rounds.
    Y = Single averaged values for the first input message block.

    Args:
        message_string (str): The original input message.
        hash_avg_rounds (int): How many rounds of averaging to apply to the hash.

    Returns:
        tuple: (X, Y) where X is the flattened feature vector (numpy array, fixed length 256)
               and Y is the target vector (16 averaged floats for the first
               message block as numpy array). Returns (None, None) on error.
    """
    # 1. Get final hash and padded blocks
    final_h_ints, padded_blocks = GetFinalHashAndBlocks(message_string)
    if final_h_ints is None or padded_blocks is None or not padded_blocks:
        print("Failed to get hash or padded blocks.")
        return None, None

    # 2. Calculate Low-Res Evolution of Final Hash
    hash_low_res_levels = create_low_res_hash_evolution(final_h_ints, hash_avg_rounds)
    if hash_low_res_levels is None:
        print("Failed to create low-res hash evolution.")
        return None, None

    # --- Prepare Input Features X ---
    # Use ONLY the final resolution level available
    if not hash_low_res_levels: # Should not happen if previous check passed, but safety
        print("Error: No resolution levels generated for hash.")
        return None, None

    final_res_data = hash_low_res_levels[-1] # Get data after the last successful averaging round
    final_res_level_index = len(hash_low_res_levels) - 1

    if len(final_res_data) != 8:
         print(f"Error: Final resolution level {final_res_level_index} data has incorrect structure (expected 8 variables).")
         return None, None

    feature_vector = []
    max_len_per_var = 32 # Pad each variable's final representation back to this length
    pad_value = -1.0   # Value used for padding

    for var_list in final_res_data: # Iterate through the 8 variables at the final resolution
        processed_var_data = []
        if isinstance(var_list, list):
            current_len = len(var_list)
            # Handle empty list case explicitly if averaging stopped early
            if current_len == 0 and final_res_level_index > 0:
                 processed_var_data = [pad_value] * max_len_per_var
            # Truncate if somehow longer (shouldn't happen if max_len=32)
            elif current_len >= max_len_per_var:
                processed_var_data = var_list[:max_len_per_var]
            # Pad if shorter
            else:
                processed_var_data = var_list + [pad_value] * (max_len_per_var - current_len)
        else:
             # This case indicates an issue in create_low_res_hash_evolution
             print(f"Warning: Unexpected data type in final resolution level {final_res_level_index}. Padding.")
             processed_var_data = [pad_value] * max_len_per_var

        # Final check on length before extending
        if len(processed_var_data) != max_len_per_var:
             print(f"Error: Padding/truncation failed for final res var. Length is {len(processed_var_data)}. Forcing length.")
             processed_var_data = (processed_var_data + [pad_value] * max_len_per_var)[:max_len_per_var] # Force length

        feature_vector.extend(processed_var_data)

    # Expected length is now 8 * max_len_per_var = 256
    expected_total_features = 8 * max_len_per_var
    if len(feature_vector) != expected_total_features:
         print(f"Error: Final feature vector X has length {len(feature_vector)}, expected {expected_total_features}.")
         return None, None # Fail if final length is wrong

    X = np.array(feature_vector, dtype=np.float32)

    # 3. Calculate Single Averaged Values for First Input Block (Target Y)
    # Pass only the first block to the reduction function
    first_block_avg_values = ReduceMessageBlocksToSingleValues([padded_blocks[0]])
    if first_block_avg_values is None or not first_block_avg_values:
        print("Failed to reduce first message block to single values.")
        return None, None

    # Result is [[16 floats]], so take the inner list
    Y = np.array(first_block_avg_values[0], dtype=np.float32)

    if Y.shape != (16,): # Ensure Y has the expected shape
         print(f"Error: Target vector Y has unexpected shape {Y.shape}. Expected (16,).")
         return None, None

    return X, Y

def generate_random_string(length, char_set=string.ascii_letters + string.digits + string.punctuation + ' '):
    """Generates a random string of specified length from a given character set."""
    if length <= 0:
        return ""
    # Consider using a more restricted charset if certain chars cause issues
    # char_set = string.ascii_letters + string.digits
    return ''.join(random.choice(char_set) for _ in range(length))

# =============================
# New utilities for state tasks
# =============================

def _u32_to_unit(x: int) -> float:
    """
    Map a 32-bit unsigned integer to approximately [-1, 1].
    Uses symmetric mapping around 0 via signed offset division.
    """
    x_u32 = x & 0xFFFFFFFF
    return float((x_u32 - 2147483648) / 2147483648.0)

def normalize_state_block(states_rounds_9):
    """
    Normalize a single block's per-round states from sha256_compress_with_states output.

    Input: list of length 64; each item is [a,b,c,d,e,f,g,h,Wt] (ints)
    Output: numpy array of shape (64, 8) with a..h normalized to [-1,1]
    """
    if not states_rounds_9 or len(states_rounds_9) != 64:
        return None
    out = []
    try:
        for r in range(64):
            row9 = states_rounds_9[r]
            if not isinstance(row9, (list, tuple)) or len(row9) != 9:
                return None
            a_to_h = row9[:8]
            out.append([_u32_to_unit(int(v)) for v in a_to_h])
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return None

def normalize_state_block_with_word(states_rounds_9):
    """
    Normalize per-round states along with the message schedule word.

    Returns:
      states_norm: np.ndarray shape (64, 8)
      word_norm:   np.ndarray shape (64,) containing normalized W[t]
    """
    if not states_rounds_9 or len(states_rounds_9) != 64:
        return None, None
    states = []
    words = []
    try:
        for row9 in states_rounds_9:
            if not isinstance(row9, (list, tuple)) or len(row9) != 9:
                return None, None
            states.append([_u32_to_unit(int(v)) for v in row9[:8]])
            words.append(_u32_to_unit(int(row9[8])))
        return np.asarray(states, dtype=np.float32), np.asarray(words, dtype=np.float32)
    except Exception:
        return None, None

def message_to_first_block_states(message: str):
    """
    Compute normalized per-round working states (a..h) for the first block and final hash ints.

    Returns:
      states_norm: np.ndarray (64, 8) with values in ~[-1,1]
      final_hash_ints: list[int] length 8 (uint32 words)
    """
    full_evo, final_h, blocks = GetEvolution(message)
    if full_evo is None or final_h is None or not blocks:
        return None, None
    first_block_states_9 = full_evo[0]  # 64 x 9
    states_norm = normalize_state_block(first_block_states_9)
    if states_norm is None:
        return None, None
    return states_norm, final_h

def message_to_evolution_maps(message: str):
    """
    Returns normalized per-round states (a..h) and message schedule words W[t]
    for the first padded block of the provided message.

    Output:
      states_norm: np.ndarray shape (64, 8)
      words_norm:  np.ndarray shape (64,)
    """
    full_evo, _, blocks = GetEvolution(message)
    if full_evo is None or not blocks:
        return None, None
    first_block_states_9 = full_evo[0]
    states_norm, words_norm = normalize_state_block_with_word(first_block_states_9)
    if states_norm is None or words_norm is None:
        return None, None
    return states_norm, words_norm

def hash_to_features(hash_words, mode: str = "basic"):
    """
    Convert final hash (8 uint32 words) into a model-ready feature vector.

    mode="basic": returns 8-dim normalized floats in [-1,1].
    """
    if hash_words is None or len(hash_words) != 8:
        return None
    if mode == "basic":
        feats = [_u32_to_unit(int(w)) for w in hash_words]
        return np.asarray(feats, dtype=np.float32)
    else:
        # Placeholder for future modes (e.g., bit-level or Fourier features)
        feats = [_u32_to_unit(int(w)) for w in hash_words]
        return np.asarray(feats, dtype=np.float32)
