"""
LTE Code Block Segmentation - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteCodeBlockSegment

MATLAB Compatibility:
- Splits input into code blocks (max size 6144)
- Prepends -1 (NULL) filler bits to first block as needed
- Appends CRC24B when B > 6144
- Returns cell array (Python list) of int8 arrays

Based on 3GPP TS 36.212 Section 5.1.2
"""

import numpy as np
from typing import List, Dict

class LTE_CodeBlockSegmentation:
    """
    LTE Code Block Segmentation
    MATLAB-COMPATIBLE - Matches lteCodeBlockSegment
    """

    def __init__(self):
        # Maximum code block size
        self.Z = 6144

        # CRC24B Generator Polynomial (for code blocks when B > 6144)
        # gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
        self.gCRC24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        # Valid turbo interleaver block sizes from 3GPP TS 36.212 Table 5.1.3-3
        self.K_table = [
            40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
            200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336,
            344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480,
            488, 496, 504, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736,
            752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1024,
            1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504,
            1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984,
            2016, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880,
            2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776, 3840,
            3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544, 4608, 4672, 4736, 4800,
            4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504, 5568, 5632, 5696, 5760,
            5824, 5888, 5952, 6016, 6080, 6144
        ]

    def crc_calculate(self, input_bits, generator_poly, L):
        """
        Calculate CRC parity bits using polynomial division in GF(2)

        MATLAB-COMPATIBLE: Negative input bit values (-1) are interpreted as logical 0
        """
        input_bits = np.array(input_bits)

        # Convert -1 (NULL/filler) to 0 for CRC calculation
        input_for_crc = np.where(input_bits < 0, 0, input_bits).astype(int)

        # Polynomial division in GF(2)
        poly = np.concatenate([input_for_crc, np.zeros(L, dtype=int)])

        for i in range(len(input_for_crc)):
            if poly[i] == 1:
                for j in range(len(generator_poly)):
                    poly[i + j] = (poly[i + j] + generator_poly[j]) % 2

        return poly[-L:].astype(int)

    def segment(self, input_bits):
        """
        Code block segmentation following 3GPP TS 36.212 Section 5.1.2

        MATLAB lteCodeBlockSegment behavior:
        - If B <= 6144: Single block, no CRC24B, may have -1 filler bits
        - If B > 6144: Multiple blocks, each with CRC24B, -1 filler bits in first block

        Parameters:
            input_bits: Transport block bits (after CRC24A attachment)

        Returns:
            code_blocks: List of code blocks (each may contain -1 for filler bits)
            segmentation_info: Dictionary with segmentation parameters
        """
        input_bits = np.array(input_bits, dtype=int)
        B = len(input_bits)

        # Determine if segmentation is needed
        if B <= self.Z:
            L = 0  # No CRC24B appended
            C = 1
            B_prime = B
        else:
            L = 24  # CRC24B will be appended to each block
            C = int(np.ceil(B / (self.Z - L)))
            B_prime = B + C * L

        # Find K+ and K- from table
        if C == 1:
            K_plus = min([k for k in self.K_table if k >= B], default=6144)
            K_minus = 0
            C_plus = 1
            C_minus = 0
        else:
            K_plus = min([k for k in self.K_table if C * k >= B_prime], default=6144)

            K_minus_candidates = [k for k in self.K_table if k < K_plus]
            if K_minus_candidates:
                K_minus = max(K_minus_candidates)
            else:
                K_minus = 0

            if K_minus > 0:
                delta_K = K_plus - K_minus
                C_minus = int(np.floor((C * K_plus - B_prime) / delta_K))
                C_plus = C - C_minus
            else:
                C_plus = C
                C_minus = 0

            if C_minus == 0:
                K_minus = 0

        # Calculate filler bits
        if C == 1:
            F = K_plus - B
        else:
            F = C_plus * K_plus + C_minus * K_minus - B_prime

        # Generate code blocks
        code_blocks = []
        bit_index = 0

        for r in range(C):
            if r < C_minus:
                K_r = K_minus
            else:
                K_r = K_plus

            code_block = np.zeros(K_r, dtype=int)

            # Filler bits only in first block
            if r == 0:
                filler_count = F
                data_start = F
                # MATLAB: "The <NULL> filler bits (represented by -1 at the output)"
                code_block[:F] = -1
            else:
                filler_count = 0
                data_start = 0

            data_length = K_r - L if L > 0 else K_r

            # Fill data bits
            for k in range(data_start, data_length):
                if bit_index < B:
                    code_block[k] = input_bits[bit_index]
                    bit_index += 1
                else:
                    code_block[k] = 0

            # Attach CRC24B if needed (only when B > 6144)
            if L >= 1:
                if r == 0 and F > 0:
                    data_for_crc = code_block[F:data_length]
                else:
                    data_for_crc = code_block[:data_length]

                # CRC calculation treats -1 as 0
                crc_bits = self.crc_calculate(data_for_crc, self.gCRC24B, L)
                code_block[data_length:] = crc_bits

            code_blocks.append(code_block)

        segmentation_info = {
            'B': B,
            'C': C,
            'L': L,
            'B_prime': B_prime,
            'K_plus': K_plus,
            'K_minus': K_minus,
            'C_plus': C_plus,
            'C_minus': C_minus,
            'F': F,
            'code_block_sizes': [len(cb) for cb in code_blocks]
        }

        return code_blocks, segmentation_info


# ============================================================================
# MATLAB-COMPATIBLE WRAPPER
# ============================================================================

def lteCodeBlockSegment(blk):
    """
    MATLAB lteCodeBlockSegment equivalent

    Syntax:
        cbs = lteCodeBlockSegment(blk)

    Parameters:
        blk: Data bit vector (column or row)

    Returns:
        cbs: Cell array (Python list) of code block segments
             Each block is int8 array containing -1 for filler bits

    MATLAB Documentation:
        "Splits the input data bit vector BLK into a cell array CBS of code
        block segments (with filler bits and type-24B CRC appended as appropriate)
        according to the rules of TS 36.212 5.1.2."

    Behavior:
        - If len(blk) <= 6144: Returns [1 block], no CRC24B, may have -1 filler
        - If len(blk) > 6144: Returns [multiple blocks], each with CRC24B

    Examples:
        >>> import numpy as np
        >>> from code_block_segment import lteCodeBlockSegment

        >>> # Example 1: No segmentation
        >>> cbs1 = lteCodeBlockSegment(np.ones(6144, dtype=int))
        >>> len(cbs1)
        1
        >>> len(cbs1[0])
        6144

        >>> # Example 2: With segmentation
        >>> cbs2 = lteCodeBlockSegment(np.ones(6145, dtype=int))
        >>> len(cbs2)
        2
        >>> [len(cb) for cb in cbs2]
        [3072, 3136]
    """
    segmenter = LTE_CodeBlockSegmentation()
    code_blocks, _ = segmenter.segment(blk)

    # Return only code blocks (not segmentation info) to match MATLAB
    # Convert to int8 for exact MATLAB compatibility
    return [cb.astype(np.int8) for cb in code_blocks]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE Code Block Segmentation - MATLAB lteCodeBlockSegment Equivalent")
    print("="*70)
    print()

    # Example 1: No segmentation (B <= 6144)
    print("Example 1: Input length 6144 (no segmentation)")
    print("-" * 70)
    cbs1 = lteCodeBlockSegment(np.ones(6144, dtype=int))
    print(f"Number of blocks: {len(cbs1)}")
    print(f"Block size: {len(cbs1[0])}")
    print(f"Data type: {cbs1[0].dtype}")
    print(f"CRC24B appended: No (B <= 6144)")
    print()

    # Example 2: With segmentation (B > 6144)
    print("Example 2: Input length 6145 (with segmentation)")
    print("-" * 70)
    cbs2 = lteCodeBlockSegment(np.ones(6145, dtype=int))
    print(f"Number of blocks: {len(cbs2)}")
    print(f"Block sizes: {[len(cb) for cb in cbs2]}")
    print(f"Expected: [3072, 3136] (matches MATLAB)")
    print(f"Data type: {cbs2[0].dtype}")
    print(f"CRC24B appended: Yes (B > 6144)")
    print()

    # Example 3: With filler bits
    print("Example 3: Input length 6200 (with filler bits)")
    print("-" * 70)
    cbs3 = lteCodeBlockSegment(np.ones(6200, dtype=int))
    print(f"Number of blocks: {len(cbs3)}")
    filler_count = np.sum(cbs3[0] == -1)
    print(f"Filler bits in first block: {filler_count}")
    print(f"First 10 bits of block 0: {cbs3[0][:10]}")
    print(f"Filler bits in second block: {np.sum(cbs3[1] == -1)}")
    print()

    # Example 4: Detailed segmentation info
    print("Example 4: Detailed segmentation (using internal API)")
    print("-" * 70)
    segmenter = LTE_CodeBlockSegmentation()
    code_blocks, info = segmenter.segment(np.ones(10000, dtype=int))

    print(f"Input size (B): {info['B']}")
    print(f"Number of blocks (C): {info['C']}")
    print(f"CRC length per block (L): {info['L']}")
    print(f"Filler bits (F): {info['F']}")
    print(f"K+ (larger block size): {info['K_plus']}")
    print(f"K- (smaller block size): {info['K_minus']}")
    print(f"C+ (number of K+ blocks): {info['C_plus']}")
    print(f"C- (number of K- blocks): {info['C_minus']}")
    print(f"Code block sizes: {info['code_block_sizes']}")
    print()

    print("="*70)
