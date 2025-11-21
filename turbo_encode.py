"""
LTE Turbo Encoding - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteTurboEncode

MATLAB Compatibility:
- PCCC (Parallel Concatenated Convolutional Code) with two 8-state RSC encoders
- QPP (Quadratic Permutation Polynomial) interleaver
- Coding rate: 1/3
- Output format: [S P1 P2] concatenated block-wise
- Filler bits (-1) treated as logical 0 for encoding
- Filler bits passed through to S and P1 output positions

Based on 3GPP TS 36.212 Section 5.1.3

Note: For CRC encoding, use crc_encode.py
      For code block segmentation, use code_block_segment.py
"""

import numpy as np
from typing import List, Tuple

# ============================================================================
# TURBO ENCODER
# ============================================================================

class LTE_TurboEncoder:
    """
    LTE Turbo Encoder with QPP Interleaver
    MATLAB-COMPATIBLE - Matches lteTurboEncode

    Based on 3GPP TS 36.212 Section 5.1.3
    """

    def __init__(self):
        # QPP Interleaver parameters from TS 36.212 Table 5.1.3-3
        # Format: K: (f1, f2)
        self.interleaver_params = {
            40: (3, 10), 48: (7, 12), 56: (19, 42), 64: (7, 16), 72: (7, 18), 80: (11, 20),
            88: (5, 22), 96: (11, 24), 104: (7, 26), 112: (41, 84), 120: (103, 90), 128: (15, 32),
            136: (9, 34), 144: (17, 108), 152: (9, 38), 160: (21, 120), 168: (101, 84), 176: (21, 44),
            184: (57, 46), 192: (23, 48), 200: (13, 50), 208: (27, 52), 216: (11, 36), 224: (27, 56),
            232: (85, 58), 240: (29, 60), 248: (33, 62), 256: (15, 32), 264: (17, 198), 272: (33, 68),
            280: (103, 210), 288: (19, 36), 296: (19, 74), 304: (37, 76), 312: (19, 78), 320: (21, 120),
            328: (21, 82), 336: (115, 84), 344: (193, 86), 352: (21, 44), 360: (133, 90), 368: (81, 46),
            376: (45, 94), 384: (23, 48), 392: (243, 98), 400: (151, 40), 408: (155, 102), 416: (25, 52),
            424: (51, 106), 432: (47, 72), 440: (91, 110), 448: (29, 168), 456: (29, 114), 464: (247, 58),
            472: (29, 118), 480: (89, 180), 488: (91, 122), 496: (157, 62), 504: (55, 84), 512: (31, 64),
            528: (17, 66), 544: (35, 68), 560: (227, 420), 576: (65, 96), 592: (19, 74), 608: (37, 76),
            624: (41, 234), 640: (39, 80), 656: (185, 82), 672: (43, 252), 688: (21, 86), 704: (155, 44),
            720: (79, 120), 736: (139, 92), 752: (23, 94), 768: (217, 48), 784: (25, 98), 800: (17, 80),
            816: (127, 102), 832: (25, 52), 848: (239, 106), 864: (17, 48), 880: (137, 110), 896: (215, 112),
            912: (29, 114), 928: (15, 58), 944: (147, 118), 960: (29, 60), 976: (59, 122), 992: (65, 124),
            1008: (55, 84), 1024: (31, 64), 1056: (17, 66), 1088: (171, 204), 1120: (67, 140), 1152: (35, 72),
            1184: (19, 74), 1216: (39, 76), 1248: (19, 78), 1280: (199, 240), 1312: (21, 82), 1344: (211, 252),
            1376: (21, 86), 1408: (43, 88), 1440: (149, 60), 1472: (45, 92), 1504: (49, 846), 1536: (71, 48),
            1568: (13, 28), 1600: (17, 80), 1632: (25, 102), 1664: (183, 104), 1696: (55, 954), 1728: (127, 96),
            1760: (27, 110), 1792: (29, 112), 1824: (29, 114), 1856: (57, 116), 1888: (45, 354), 1920: (31, 120),
            1952: (59, 610), 1984: (185, 124), 2016: (113, 420), 2048: (31, 64), 2112: (17, 66), 2176: (171, 136),
            2240: (209, 420), 2304: (253, 216), 2368: (367, 444), 2432: (265, 456), 2496: (181, 468), 2560: (39, 80),
            2624: (27, 164), 2688: (127, 504), 2752: (143, 172), 2816: (43, 88), 2880: (29, 300), 2944: (45, 92),
            3008: (157, 188), 3072: (47, 96), 3136: (13, 28), 3200: (111, 240), 3264: (443, 204), 3328: (51, 104),
            3392: (51, 212), 3456: (451, 192), 3520: (257, 220), 3584: (57, 336), 3648: (313, 228), 3712: (271, 232),
            3776: (179, 236), 3840: (331, 120), 3904: (363, 244), 3968: (375, 248), 4032: (127, 168), 4096: (31, 64),
            4160: (33, 130), 4224: (43, 264), 4288: (33, 134), 4352: (477, 408), 4416: (35, 138), 4480: (233, 280),
            4544: (357, 142), 4608: (337, 480), 4672: (37, 146), 4736: (71, 444), 4800: (71, 120), 4864: (37, 152),
            4928: (39, 462), 4992: (127, 234), 5056: (39, 158), 5120: (39, 80), 5184: (31, 96), 5248: (113, 902),
            5312: (41, 166), 5376: (251, 336), 5440: (43, 170), 5504: (21, 86), 5568: (43, 174), 5632: (45, 176),
            5696: (45, 178), 5760: (161, 120), 5824: (89, 182), 5888: (323, 184), 5952: (47, 186), 6016: (23, 94),
            6080: (47, 190), 6144: (263, 480)
        }

        # Transfer function: G(D) = [1, g1(D)/g0(D)]
        # g0(D) = 1 + D^2 + D^3
        # g1(D) = 1 + D + D^3

    def qpp_interleaver(self, c: List[int], K: int) -> List[int]:
        """
        QPP (Quadratic Permutation Polynomial) interleaver

        MATLAB-COMPATIBLE: Π(i) = (f1*i + f2*i^2) mod K

        Handles filler bits (-1) correctly by permuting them

        Parameters:
            c: Input sequence (may contain -1 for filler bits)
            K: Block size (must be in Table 5.1.3-3)

        Returns:
            Interleaved sequence
        """
        if K not in self.interleaver_params:
            raise ValueError(f"Interleaver params for K={K} not found in Table 5.1.3-3")

        f1, f2 = self.interleaver_params[K]
        i = np.arange(K, dtype=np.int64)
        p = (f1 * i + f2 * (i * i)) % K

        # Use int16 to handle -1 (NULL) values
        c_arr = np.array(c, dtype=np.int16)
        return c_arr[p].tolist()

    def rsc_encode(self, u: List[int], filler_count: int = 0) -> Tuple[List[int], List[int], int]:
        """
        RSC (Recursive Systematic Convolutional) encoder

        MATLAB-COMPATIBLE filler bit handling:
        - Encoder initialized with all zeros (state = 0)
        - For filler positions (k=0,...,filler_count-1):
          * Input: Treat -1 as 0 for encoding (state calculation)
          * Output: Set to NULL (-1)

        Transfer function: G(D) = [1, g1(D)/g0(D)]
        - g0(D) = 1 + D^2 + D^3
        - g1(D) = 1 + D + D^3

        Parameters:
            u: Input sequence
            filler_count: Number of filler bits at start

        Returns:
            (systematic output, parity output, final state)
        """
        sys_out, par_out = [], []
        state = 0  # Encoder initialized with all zeros

        for idx, bit in enumerate(u):
            # For encoding: treat -1 as 0, otherwise use the bit value
            encoding_bit = 0 if bit == -1 else int(bit)

            # RSC encoder logic (8-state, rate 1/2)
            # State bits: s3, s2, s1
            s1 = state & 1
            s2 = (state >> 1) & 1
            s3 = (state >> 2) & 1

            # Feedback: r = input XOR s2 XOR s3
            r = encoding_bit ^ s2 ^ s3

            # Parity: p = r XOR s1 XOR s3
            p = r ^ s1 ^ s3

            # For filler positions: output NULL
            if idx < filler_count:
                sys_out.append(-1)  # x_k = NULL
                par_out.append(-1)  # z_k = NULL
            else:
                sys_out.append(encoding_bit)
                par_out.append(int(p))

            # State always updates (using encoding_bit = 0 for fillers)
            # Next state: (s2, s1, r)
            state = (s2 << 2) | (s1 << 1) | (r & 1)

        return sys_out, par_out, state

    def generate_tail(self, state: int) -> Tuple[List[int], List[int]]:
        """
        Generate trellis termination tail bits

        3 tail bits are generated to return encoder to all-zeros state
        """
        tail_u, tail_p = [], []
        cur = state

        for _ in range(3):
            s1 = cur & 1
            s2 = (cur >> 1) & 1
            s3 = (cur >> 2) & 1

            # Feedback bit to force return to zero state
            u = s2 ^ s3
            r = u ^ s2 ^ s3
            p = r ^ s1 ^ s3

            tail_u.append(int(u))
            tail_p.append(int(p))

            cur = (s2 << 2) | (s1 << 1) | (r & 1)

        return tail_u, tail_p

    def turbo_encode(self, message: List[int], F: int = 0) -> Tuple[List[int], List[int], List[int]]:
        """
        Main turbo encoder (PCCC)

        MATLAB-COMPATIBLE: Matches lteTurboEncode behavior

        Encoder architecture:
        - Two 8-state RSC constituent encoders
        - QPP interleaver between encoders
        - Trellis termination (3 tail bits per encoder)
        - Output: [d0, d1, d2] streams (systematic, parity1, parity2)

        Filler bit handling (MATLAB documentation):
        "To support the correct processing of filler bits, negative input bit
        values are specially processed. They are treated as logical 0 at the
        input to both encoders but their negative values are passed directly
        through to the associated output positions in sub-blocks S and P1."

        For filler bits (k=0,...,F-1):
        - Input: c_k = -1 (NULL marker)
        - Processing: Encoder treats -1 as 0 for state calculation
        - Output: x_k = -1, z^(1)_k = -1, z^(2)_k = -1 (depending on interleaver)
        - State: Updates normally (as if input was 0)

        Parameters:
            message: Input bit sequence (may contain -1 for filler bits)
            F: Number of filler bits at start

        Returns:
            (d0, d1, d2) - systematic, parity1, parity2 streams
                Each stream has length K+4 (including trellis termination)
        """
        K = len(message)

        # First RSC encoder (initialized with state=0)
        # Processes -1 as 0, outputs -1 for first F positions
        sys1, par1, state1 = self.rsc_encode(message, filler_count=F)
        tail_u1, tail_p1 = self.generate_tail(state1)

        # Interleave the input (interleaves -1 values normally)
        c_prime = self.qpp_interleaver(message, K)

        # Second RSC encoder
        # Process c_prime and manually set outputs to -1 where input is -1
        sys2_temp, par2_temp, state2 = self.rsc_encode(c_prime, filler_count=0)

        # Set outputs to -1 where c_prime had -1
        par2 = []
        for idx in range(K):
            if c_prime[idx] == -1:
                par2.append(-1)  # z^(2)_k = NULL for filler positions
            else:
                par2.append(par2_temp[idx])

        tail_u2, tail_p2 = self.generate_tail(state2)

        # Build transmission sequence in interleaved format
        # Format: x0, z0, z0', x1, z1, z1', ..., xK-1, zK-1, zK-1'
        # Then append tail bits
        tx_seq = []
        for k in range(K):
            tx_seq.extend([sys1[k], par1[k], par2[k]])

        # Append tails (never NULL)
        for t in range(3):
            tx_seq.append(tail_u1[t])
            tx_seq.append(tail_p1[t])
        for t in range(3):
            tx_seq.append(tail_u2[t])
            tx_seq.append(tail_p2[t])

        # Split into d0, d1, d2 streams for MATLAB compatibility
        d0 = tx_seq[0::3]  # Systematic: S
        d1 = tx_seq[1::3]  # Parity 1: P1
        d2 = tx_seq[2::3]  # Parity 2: P2

        return d0, d1, d2


# ============================================================================
# MATLAB-COMPATIBLE WRAPPER FUNCTION
# ============================================================================

def lteTurboEncode(blk):
    """
    MATLAB lteTurboEncode equivalent - Turbo encoding

    Syntax:
        out = lteTurboEncode(in)

    Parameters:
        blk: Input data vector or cell array (Python list) of vectors
             Only legal turbo interleaver block sizes are supported (40-6144)
             Filler bits supported through negative input values (-1)

    Returns:
        out: Turbo encoded bits as int8 vector or cell array of int8 vectors
             Output format: [S P1 P2] where:
               S = systematic bits (K+4)
               P1 = encoder 1 parity bits (K+4)
               P2 = encoder 2 parity bits (K+4)
             Total output length: 3*(K+4) bits

    MATLAB Documentation:
        "The encoder is a Parallel Concatenated Convolutional Code (PCCC) with
        two 8-state constituent encoders and a contention free interleaver. The
        coding rate of turbo encoder is 1/3 and the 3 encoded parity streams
        are concatenated block-wise to form the encoded output i.e. [S P1 P2]
        where S is the systematic bits, P1 is the encoder 1 bits and P2 is the
        encoder 2 bits. To support the correct processing of filler bits,
        negative input bit values are specially processed. They are treated as
        logical 0 at the input to both encoders but their negative values are
        passed directly through to the associated output positions in
        sub-blocks S and P1."

    Examples:
        >>> # Single vector input
        >>> out1 = lteTurboEncode(np.ones(40, dtype=int))
        >>> out1.shape
        (132,)
        >>> out1.dtype
        dtype('int8')

        >>> # Cell array input
        >>> out2 = lteTurboEncode([np.ones(40, dtype=int), np.ones(6144, dtype=int)])
        >>> len(out2)
        2
        >>> [len(cb) for cb in out2]
        [132, 18444]

        >>> # With filler bits (-1)
        >>> input_with_filler = np.array([-1, -1, 1, 0, 1, 1, 0, 1] + [1]*32, dtype=int)
        >>> out3 = lteTurboEncode(input_with_filler)
        >>> out3[0]  # Filler bit passed through in S
        -1

    Note:
        For CRC encoding, use crc_encode.py
        For code block segmentation, use code_block_segment.py
    """
    encoder = LTE_TurboEncoder()

    # Check if input is cell array (Python list)
    if isinstance(blk, list):
        # Process each vector in the cell array
        result = []
        for vec in blk:
            vec_array = np.array(vec, dtype=int)
            K = len(vec_array)

            # Count filler bits (negative values)
            F = np.sum(vec_array < 0)

            # Encode
            d0, d1, d2 = encoder.turbo_encode(vec_array.tolist(), F=F)

            # Concatenate [S P1 P2] and convert to int8
            output = np.concatenate([d0, d1, d2]).astype(np.int8)
            result.append(output)

        return result

    else:
        # Single vector input
        blk_array = np.array(blk, dtype=int)
        K = len(blk_array)

        # Count filler bits (negative values)
        F = np.sum(blk_array < 0)

        # Encode
        d0, d1, d2 = encoder.turbo_encode(blk_array.tolist(), F=F)

        # Concatenate [S P1 P2] and convert to int8
        output = np.concatenate([d0, d1, d2]).astype(np.int8)

        return output


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE Turbo Encode - MATLAB lteTurboEncode Equivalent")
    print("="*70)
    print()

    # Example 1: Single vector
    print("Example 1: Single vector - ones(40,1)")
    print("-" * 70)
    out1 = lteTurboEncode(np.ones(40, dtype=int))
    print(f"Input: 40 bits")
    print(f"Output: {len(out1)} bits (3*44 = 132)")
    print(f"Type: {out1.dtype}")
    print()

    # Example 2: Cell array
    print("Example 2: Cell array - {{ones(40,1), ones(6144,1)}}")
    print("-" * 70)
    out2 = lteTurboEncode([np.ones(40, dtype=int), np.ones(6144, dtype=int)])
    print(f"Input: 2 vectors [40, 6144]")
    print(f"Output: 2 vectors [{len(out2[0])}, {len(out2[1])}]")
    print(f"Types: {[o.dtype for o in out2]}")
    print()

    # Example 3: With filler bits
    print("Example 3: With filler bits (-1)")
    print("-" * 70)
    input_with_filler = np.array([-1, -1, -1] + [1]*37, dtype=int)
    out3 = lteTurboEncode(input_with_filler)
    print(f"Input: 40 bits with 3 filler bits (-1) at start")
    print(f"Output: {len(out3)} bits")
    print(f"First 3 bits of S: {out3[:3]} (should be [-1, -1, -1])")
    print()

    print("="*70)
    print("For complete LTE encoding chain:")
    print("  1. CRC encoding → crc_encode.py")
    print("  2. Code block segmentation → code_block_segment.py")
    print("  3. Turbo encoding → turbo_encode.py (this module)")
    print("="*70)
