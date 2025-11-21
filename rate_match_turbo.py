"""
LTE Rate Matching for Turbo Coded Data - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteRateMatchTurbo

MATLAB Compatibility:
- Sub-block interleaving (32 columns, Table 5.1.4-1 permutation)
- Circular buffer creation with interlacing
- Bit selection and pruning based on redundancy version (RV)
- NULL filler bits (-1) skipped during rate matching
- Supports RV values: 0, 1, 2, 3 for HARQ retransmissions

Based on 3GPP TS 36.212 Section 5.1.4.1

Note: For turbo encoding, use turbo_encode.py
      For CRC and code block segmentation, use crc_encode.py and code_block_segment.py
"""

import numpy as np
from typing import Tuple

# ============================================================================
# RATE MATCHING
# ============================================================================

class LTE_RateMatching:
    """
    LTE Rate Matching for Turbo Coded Data
    MATLAB-COMPATIBLE - Matches lteRateMatchTurbo

    Based on 3GPP TS 36.212 Section 5.1.4.1
    """

    def __init__(self):
        # Sub-block interleaver parameters
        self.C_subblock = 32  # Fixed number of columns

        # Column permutation pattern from Table 5.1.4-1
        self.P = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                          1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])

    def sub_block_interleaver(self, d: np.ndarray, stream_idx: int) -> Tuple[np.ndarray, int]:
        """
        Sub-block interleaver (3GPP TS 36.212 Section 5.1.4.1.1)

        The sub-block interleaver reshapes the encoded bit sequence into a matrix
        with CTCSubblock=32 columns and RTCSubblock rows (row-by-row filling).

        For d^(0) and d^(1): Inter-column permutation applied
        For d^(2): Special permutation formula applied

        Parameters:
            d: Input bit stream (may contain -1 for NULL)
            stream_idx: Stream index (0, 1, or 2)

        Returns:
            (v, K_PI): Interleaved output and size
        """
        D = len(d)

        # Step (1): Number of columns is fixed at 32
        C_subblock = self.C_subblock

        # Step (2): Determine number of rows
        R_subblock = int(np.ceil(D / C_subblock))

        # Calculate total matrix size
        K_PI = R_subblock * C_subblock

        # Step (3): Pad with NULL bits (-1) if needed
        N_D = K_PI - D

        # Create padded sequence y (prepend NULL bits)
        if N_D > 0:
            y = np.concatenate([np.full(N_D, -1, dtype=int), d])
        else:
            y = d.copy()

        # Write y into (R_subblock × C_subblock) matrix ROW BY ROW
        matrix = y.reshape(R_subblock, C_subblock)

        # For d^(0) and d^(1): Standard inter-column permutation
        if stream_idx in [0, 1]:
            # Step (4): Perform inter-column permutation
            matrix_permuted = matrix[:, self.P]

            # Step (5): Read column by column (column-major order / Fortran order)
            v = matrix_permuted.flatten('F')

        # For d^(2): Special permutation formula
        else:
            # Apply special permutation formula
            # π(k) = (P[⌊k/R⌋] + C_subblock × (k mod R) + 1) mod K_PI
            v = np.zeros(K_PI, dtype=int)

            for k in range(K_PI):
                col_index = k // R_subblock
                row_index = k % R_subblock
                pi_k = (self.P[col_index] + C_subblock * row_index + 1) % K_PI
                v[k] = y[pi_k]

        return v, K_PI

    def create_circular_buffer(self, d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create circular buffer from three interleaved streams
        3GPP TS 36.212 Section 5.1.4.1.2

        The circular buffer is created by:
        1. Sub-block interleaving each of the three streams (d^(0), d^(1), d^(2))
        2. Interlacing v^(1) and v^(2)
        3. Appending the interlaced result to v^(0)

        Interlacing pattern:
        w_k = v_k^(0)              for k = 0, ..., K_Π - 1
        w_{K_Π + 2k} = v_k^(1)     for k = 0, ..., K_Π - 1
        w_{K_Π + 2k+1} = v_k^(2)   for k = 0, ..., K_Π - 1

        This allows equal protection for each parity sequence.

        Parameters:
            d0: Systematic stream (d^(0))
            d1: Parity stream 1 (d^(1))
            d2: Parity stream 2 (d^(2))

        Returns:
            (w, K_w): Circular buffer and its size
        """
        # Apply sub-block interleaving to each stream
        v0, K_pi = self.sub_block_interleaver(d0, stream_idx=0)
        v1, _ = self.sub_block_interleaver(d1, stream_idx=1)
        v2, _ = self.sub_block_interleaver(d2, stream_idx=2)

        # Create circular buffer with interlacing pattern
        K_w = 3 * K_pi
        w = np.full(K_w, -1, dtype=int)

        # w_k = v_k^(0) for k = 0, ..., K_Π - 1
        for k in range(K_pi):
            w[k] = v0[k]

        # w_{K_Π + 2k} = v_k^(1) for k = 0, ..., K_Π - 1
        for k in range(K_pi):
            w[K_pi + 2*k] = v1[k]

        # w_{K_Π + 2k+1} = v_k^(2) for k = 0, ..., K_Π - 1
        for k in range(K_pi):
            w[K_pi + 2*k + 1] = v2[k]

        return w, K_w

    def bit_selection_and_pruning(self, w: np.ndarray, K_w: int, E: int, rv: int, R_subblock: int) -> np.ndarray:
        """
        Bit selection and pruning from circular buffer
        3GPP TS 36.212 Section 5.1.4.1.2

        Selects E bits from the circular buffer starting at position k_0
        (determined by redundancy version RV). NULL bits (-1) are skipped.

        MATLAB-COMPATIBLE: Skips NULL bits (-1) during selection

        For HARQ:
        - Different RV values provide different starting points
        - Chase combining: Same data in retransmissions
        - Incremental redundancy: Different data in retransmissions

        Parameters:
            w: Circular buffer
            K_w: Circular buffer size
            E: Number of bits to select
            rv: Redundancy version (0, 1, 2, or 3)
            R_subblock: Number of rows in sub-block interleaver

        Returns:
            Selected and pruned output bits
        """
        # For UL-SCH: N_cb = K_w (no soft buffer limitation)
        N_cb = K_w

        # Calculate starting position based on redundancy version
        # k_0 = R_subblock × (2 × ⌈N_cb/(8×R_subblock)⌉ × rv + 2)
        k_0 = int(R_subblock * (2 * np.ceil(N_cb / (8 * R_subblock)) * rv + 2))

        # Bit selection (circular buffer readout, skip NULL bits)
        out = []
        j = 0  # Buffer position counter
        k = 0  # Output bit counter

        while k < E:
            # Wrap around circular buffer
            w_idx = (k_0 + j) % N_cb
            bit = w[w_idx]

            # Skip NULL bits (represented as -1)
            if bit != -1:
                out.append(int(bit))
                k += 1

            j += 1

        return np.array(out, dtype=int)


# ============================================================================
# MATLAB-COMPATIBLE WRAPPER FUNCTION
# ============================================================================

def lteRateMatchTurbo(in_data, outlen, rv, chs=None):
    """
    MATLAB lteRateMatchTurbo equivalent - Turbo rate matching

    Syntax:
        out = lteRateMatchTurbo(in, outlen, rv)
        out = lteRateMatchTurbo(in, outlen, rv, chs)

    Parameters:
        in_data: Input data - vector or cell array (Python list) of vectors
                 Assumed to be code blocks from turbo encoder
                 Each vector length must be integer multiple of 3
                 Negative values (-1) treated as NULL filler bits (skipped)
        outlen: Output vector length (nonnegative integer)
        rv: Redundancy version (0, 1, 2, or 3)
        chs: Optional channel configuration structure (dict) for downlink
             Not implemented yet - assumes UL-SCH (no soft buffer limit)

    Returns:
        out: Rate matched output as column vector

    MATLAB Documentation:
        "This function includes the stages of sub-block interleaving, bit
        collection and bit selection and pruning defined for turbo encoded
        data (TS 36.212 Section 5.1.4.1). The function considers negative
        values in the input data as <NULL> filler bits inserted during code
        block segmentation and skips them during rate matching."

    Examples:
        >>> # Single code block
        >>> from turbo_encode import lteTurboEncode
        >>> encoded = lteTurboEncode(np.ones(40, dtype=int))  # 132 bits
        >>> rm_out = lteRateMatchTurbo(encoded, 100, 0)
        >>> len(rm_out)
        100

        >>> # Cell array (multiple code blocks)
        >>> cbs = [np.ones(132, dtype=int), np.ones(132, dtype=int)]
        >>> rm_out = lteRateMatchTurbo(cbs, 200, 0)
        >>> len(rm_out)
        200

    Note:
        Currently implements UL-SCH behavior (no soft buffer limitation).
        Downlink configuration (chs parameter) not yet implemented.
    """
    # Validate RV
    if rv not in [0, 1, 2, 3]:
        raise ValueError(f"RV must be 0, 1, 2, or 3, got {rv}")

    # Create rate matcher instance
    rate_matcher = LTE_RateMatching()

    # Check if input is cell array (list)
    if isinstance(in_data, list):
        # Process each code block separately
        result = []

        for code_block in in_data:
            if len(code_block) == 0:
                continue

            # Verify length is multiple of 3
            if len(code_block) % 3 != 0:
                raise ValueError(f"Code block length ({len(code_block)}) must be multiple of 3")

            # Calculate D (length of each stream)
            D = len(code_block) // 3

            # Extract streams (already in [S P1 P2] format from lteTurboEncode)
            d0 = code_block[0::3]
            d1 = code_block[1::3]
            d2 = code_block[2::3]

            # Create circular buffer
            w, K_w = rate_matcher.create_circular_buffer(d0, d1, d2)

            # Calculate R_subblock for k_0 calculation
            R_subblock = int(np.ceil(D / rate_matcher.C_subblock))

            # Bit selection and pruning per code block
            # For cell array, outlen is total - divide equally among blocks
            E_per_block = outlen // len([cb for cb in in_data if len(cb) > 0])

            out_bits = rate_matcher.bit_selection_and_pruning(w, K_w, E_per_block, rv, R_subblock)
            result.append(out_bits)

        # Concatenate all rate-matched blocks
        if len(result) == 0:
            return np.array([], dtype=int)

        output = np.concatenate(result)

        # Trim or pad to exact outlen if needed
        if len(output) > outlen:
            output = output[:outlen]
        elif len(output) < outlen:
            # Pad with zeros if needed
            output = np.concatenate([output, np.zeros(outlen - len(output), dtype=int)])

        return output.astype(int)

    else:
        # Single vector input
        in_array = np.array(in_data, dtype=int)

        if len(in_array) == 0:
            return np.array([], dtype=int)

        # Verify length is multiple of 3
        if len(in_array) % 3 != 0:
            raise ValueError(f"Input length ({len(in_array)}) must be multiple of 3")

        # Calculate D (length of each stream)
        D = len(in_array) // 3

        # Extract streams (already in [S P1 P2] format from lteTurboEncode)
        d0 = in_array[0::3]
        d1 = in_array[1::3]
        d2 = in_array[2::3]

        # Create circular buffer
        w, K_w = rate_matcher.create_circular_buffer(d0, d1, d2)

        # Calculate R_subblock for k_0 calculation
        R_subblock = int(np.ceil(D / rate_matcher.C_subblock))

        # Bit selection and pruning
        output = rate_matcher.bit_selection_and_pruning(w, K_w, outlen, rv, R_subblock)

        return output.astype(int)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE Rate Match Turbo - MATLAB lteRateMatchTurbo Equivalent")
    print("="*70)
    print()

    # Example 1: Single code block (simulating turbo encoded output)
    print("Example 1: Rate match 132 bits to 100 bits (RV=0)")
    print("-" * 70)
    # Simulate turbo encoded output (normally from lteTurboEncode)
    invec = np.ones(132, dtype=int)
    rm_out = lteRateMatchTurbo(invec, 100, 0)
    print(f"Input: {len(invec)} bits")
    print(f"Output: {len(rm_out)} bits")
    print(f"RV: 0")
    print()

    # Example 2: Different redundancy versions
    print("Example 2: Different redundancy versions (HARQ)")
    print("-" * 70)
    for rv in [0, 1, 2, 3]:
        rm = lteRateMatchTurbo(invec, 100, rv)
        print(f"RV={rv}: {len(rm)} bits, first 5: {rm[:5]}")
    print()

    # Example 3: With filler bits
    print("Example 3: With filler bits (-1)")
    print("-" * 70)
    # Simulate encoded with filler bits
    input_filler = np.array([-1, -1, -1] + [1]*129, dtype=int)
    rm_filler = lteRateMatchTurbo(input_filler, 100, 0)
    print(f"Input: {len(input_filler)} bits (3 filler bits)")
    print(f"Output: {len(rm_filler)} bits")
    print(f"Filler bits (-1) skipped during rate matching")
    print()

    print("="*70)
    print("For complete LTE encoding chain:")
    print("  1. CRC encoding → crc_encode.py")
    print("  2. Code block segmentation → code_block_segment.py")
    print("  3. Turbo encoding → turbo_encode.py")
    print("  4. Rate matching → rate_match_turbo.py (this module)")
    print("="*70)
