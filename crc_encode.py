"""
LTE CRC Encode - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteCRCEncode

MATLAB Compatibility:
- Negative input bit values (-1) are interpreted as logical 0 for CRC calculation
- Supports CRC types: '8', '16', '24A', '24B'
- XOR masking applied MSB-first
- Filler bits preserved in output

Based on 3GPP TS 36.212 Section 5.1.1
"""

import numpy as np
from typing import Union, List

class LTE_CRC_CodeBlockSegmentation:
    """
    LTE CRC Calculation and Code Block Segmentation
    MATLAB-COMPATIBLE - Matches lteCRCEncode and lteCodeBlockSegment
    """

    def __init__(self):
        # Maximum code block size
        self.Z = 6144

        # CRC Generator Polynomials (MSB first, length includes x^n term)

        # gCRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
        self.gCRC24A = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]

        # gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
        self.gCRC24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        # gCRC16(D) = D^16 + D^12 + D^5 + 1
        self.gCRC16 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        # gCRC8(D) = D^8 + D^7 + D^4 + D^3 + D + 1
        self.gCRC8 = [1, 1, 0, 0, 1, 1, 0, 1, 1]

        # Valid turbo interleaver block sizes from table 5.1.3-3
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

        MATLAB lteCRCEncode behavior:
        "To support the correct processing of filler bits, negative input bit
        values are interpreted as logical 0 for the purposes of the CRC calculation
        (-1 is used in the LTE Toolbox to represent filler bits)."

        Parameters:
            input_bits: Input bit array (may contain -1 for filler bits)
            generator_poly: CRC generator polynomial
            L: CRC length (8, 16, or 24)

        Returns:
            CRC parity bits (L bits)
        """
        input_bits = np.array(input_bits)

        # MATLAB: "negative input bit values are interpreted as logical 0"
        # Convert -1 (NULL/filler) to 0 for CRC calculation
        input_for_crc = np.where(input_bits < 0, 0, input_bits).astype(int)

        # Polynomial division in GF(2)
        poly = np.concatenate([input_for_crc, np.zeros(L, dtype=int)])

        for i in range(len(input_for_crc)):
            if poly[i] == 1:
                for j in range(len(generator_poly)):
                    poly[i + j] = (poly[i + j] + generator_poly[j]) % 2

        return poly[-L:].astype(int)

    def crc_attach(self, input_bits, crc_type='CRC24A', mask=0):
        """
        Attach CRC to input bit sequence - MATLAB lteCRCEncode equivalent

        MATLAB Syntax:
            blkcrc = lteCRCEncode(blk, poly)
            blkcrc = lteCRCEncode(blk, poly, mask)

        Parameters:
            input_bits: Input bit vector (may contain -1 for filler bits)
            crc_type: CRC polynomial type ('8', '16', '24A', '24B')
            mask: XOR mask value (typically RNTI), applied MSB-first

        Returns:
            Bit vector with CRC appended (input preserved + CRC bits)

        Examples:
            >>> crc = LTE_CRC_CodeBlockSegmentation()

            # Example 1: CRC of all zeros is all zeros
            >>> output = crc.crc_attach(np.zeros(100, dtype=int), 'CRC24A')
            >>> len(output)
            124

            # Example 2: CRC with mask=1 (MSB-first)
            >>> output = crc.crc_attach(np.zeros(100, dtype=int), 'CRC24A', mask=1)
            >>> output[-1]  # LSB of mask is 1
            1

            # Example 3: Filler bits treated as 0
            >>> input_with_filler = np.array([-1, -1, 1, 0, 1, 1, 0])
            >>> output = crc.crc_attach(input_with_filler, 'CRC24A')
            >>> output[:2]  # Filler bits preserved
            array([-1, -1])
        """
        input_bits = np.array(input_bits)

        # Select CRC polynomial and length
        if crc_type == 'CRC24A' or crc_type == '24A':
            generator_poly = self.gCRC24A
            L = 24
        elif crc_type == 'CRC24B' or crc_type == '24B':
            generator_poly = self.gCRC24B
            L = 24
        elif crc_type == 'CRC16' or crc_type == '16':
            generator_poly = self.gCRC16
            L = 16
        elif crc_type == 'CRC8' or crc_type == '8':
            generator_poly = self.gCRC8
            L = 8
        else:
            raise ValueError(f"Unknown CRC type: {crc_type}. Valid types: '8', '16', '24A', '24B'")

        # Calculate CRC (handles -1 as 0 internally)
        parity_bits = self.crc_calculate(input_bits, generator_poly, L)

        # Apply mask if provided (MSB first)
        # MATLAB: "The MASK value is applied to the CRC bits MSB first/LSB last"
        if mask != 0:
            mask_bits = np.array([int(b) for b in format(mask, f'0{L}b')], dtype=int)
            parity_bits = (parity_bits + mask_bits) % 2

        # Return input (with -1 preserved) + CRC
        return np.concatenate([input_bits, parity_bits])

    def code_block_segmentation(self, input_bits):
        """
        Code block segmentation following 3GPP TS 36.212 Section 5.1.2

        MATLAB-COMPATIBLE: Filler bits represented as -1 (NULL)

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
            L = 0
            C = 1
            B_prime = B
        else:
            L = 24
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

            if r == 0:
                filler_count = F
                data_start = F
                # MATLAB-COMPATIBLE: Filler bits set to -1 (NULL)
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

            # Attach CRC24B if needed
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
# HELPER FUNCTIONS
# ============================================================================

def lteCodeBlockSegment(blk):
    """
    MATLAB-style wrapper for code block segmentation

    Syntax:
        cbs = lteCodeBlockSegment(blk)

    Parameters:
        blk: Data bit vector (column or row)

    Returns:
        cbs: List of code block segments (MATLAB cell array equivalent)
             Each block contains -1 for filler bits and CRC24B if applicable

    MATLAB Behavior:
        - If len(blk) <= 6144: Returns single code block (may have -1 filler bits)
        - If len(blk) > 6144: Returns multiple code blocks with CRC24B appended

    Examples:
        >>> # No segmentation (B <= 6144)
        >>> cbs1 = lteCodeBlockSegment(np.ones(6144, dtype=int))
        >>> len(cbs1)
        1
        >>> len(cbs1[0])
        6144

        >>> # With segmentation (B > 6144)
        >>> cbs2 = lteCodeBlockSegment(np.ones(6145, dtype=int))
        >>> len(cbs2)
        2
        >>> [len(cb) for cb in cbs2]
        [3072, 3136]
    """
    crc_processor = LTE_CRC_CodeBlockSegmentation()
    code_blocks, _ = crc_processor.code_block_segmentation(blk)

    # Return only code blocks (not segmentation info) to match MATLAB
    # Convert to int8 for exact MATLAB compatibility
    return [cb.astype(np.int8) for cb in code_blocks]


def lteCRCEncode(blk, poly, mask=0):
    """
    MATLAB-style wrapper function for CRC encoding

    Syntax:
        blkcrc = lteCRCEncode(blk, poly)
        blkcrc = lteCRCEncode(blk, poly, mask)

    Parameters:
        blk: Input bit vector (column or row)
        poly: CRC polynomial ('8', '16', '24A', '24B')
        mask: XOR mask value (optional)

    Returns:
        blkcrc: Input with CRC appended

    Examples:
        >>> # Example 1: CRC of all zeros
        >>> crc1 = lteCRCEncode(np.zeros(100, dtype=int), '24A')
        >>> len(crc1)
        124

        >>> # Example 2: CRC with mask
        >>> crc2 = lteCRCEncode(np.zeros(100, dtype=int), '24A', 1)
        >>> crc2[-1]
        1
    """
    crc_processor = LTE_CRC_CodeBlockSegmentation()
    return crc_processor.crc_attach(blk, crc_type=poly, mask=mask)


def binary_string_to_array(binary_string: str) -> np.ndarray:
    """Convert binary string to numpy array"""
    return np.array([int(bit) for bit in binary_string], dtype=int)


def array_to_binary_string(bit_array) -> str:
    """Convert array to binary string (showing -1 as 'N' for NULL)"""
    return ''.join(['N' if int(b) == -1 else str(int(b)) for b in bit_array])


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE CRC Encode - MATLAB lteCRCEncode Equivalent")
    print("="*70)
    print()

    # Example 1: Basic CRC24A
    print("Example 1: Basic CRC24A")
    print("-" * 70)
    input_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=int)
    output = lteCRCEncode(input_data, '24A')
    print(f"Input:  {input_data}")
    print(f"Output: {output[:10]}... (total {len(output)} bits)")
    print(f"CRC:    {output[10:]} ({len(output)-len(input_data)} bits)")
    print()

    # Example 2: CRC with mask (RNTI)
    print("Example 2: CRC24A with mask=5 (simulating RNTI)")
    print("-" * 70)
    input_data = np.zeros(100, dtype=int)
    output_no_mask = lteCRCEncode(input_data, '24A', mask=0)
    output_with_mask = lteCRCEncode(input_data, '24A', mask=5)
    print(f"CRC (no mask): {output_no_mask[-10:]}")
    print(f"CRC (mask=5):  {output_with_mask[-10:]}")
    print()

    # Example 3: Filler bits handling
    print("Example 3: CRC with filler bits (-1)")
    print("-" * 70)
    input_with_filler = np.array([-1, -1, -1, 1, 0, 1, 1, 0, 1, 0], dtype=int)
    input_with_zeros = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0], dtype=int)

    output_filler = lteCRCEncode(input_with_filler, '24A')
    output_zeros = lteCRCEncode(input_with_zeros, '24A')

    print(f"Input with -1: {input_with_filler}")
    print(f"Input with 0:  {input_with_zeros}")
    print(f"CRC from -1:   {output_filler[10:14]}... (same as below)")
    print(f"CRC from 0:    {output_zeros[10:14]}...")
    print(f"CRCs match: {np.array_equal(output_filler[10:], output_zeros[10:])}")
    print()

    # Example 4: All CRC types
    print("Example 4: All CRC types")
    print("-" * 70)
    input_data = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=int)

    for poly in ['8', '16', '24A', '24B']:
        output = lteCRCEncode(input_data, poly)
        crc = output[len(input_data):]
        print(f"CRC-{poly:4}: {crc} ({len(crc)} bits)")
    print()

    # Example 5: MATLAB-compatible code block segmentation
    print("Example 5: MATLAB lteCodeBlockSegment equivalent")
    print("-" * 70)

    # Test case 1: No segmentation (B <= 6144)
    cbs1 = lteCodeBlockSegment(np.ones(6144, dtype=int))
    print(f"Input length 6144 (no segmentation):")
    print(f"  Number of blocks: {len(cbs1)}")
    print(f"  Block size: {len(cbs1[0])}")
    print(f"  Data type: {cbs1[0].dtype}")
    print()

    # Test case 2: With segmentation (B > 6144)
    cbs2 = lteCodeBlockSegment(np.ones(6145, dtype=int))
    print(f"Input length 6145 (with segmentation):")
    print(f"  Number of blocks: {len(cbs2)}")
    print(f"  Block sizes: {[len(cb) for cb in cbs2]}")
    print(f"  Expected: [3072, 3136] (matches MATLAB)")
    print()

    # Test case 3: With filler bits
    cbs3 = lteCodeBlockSegment(np.ones(6200, dtype=int))
    print(f"Input length 6200 (with filler bits):")
    print(f"  Number of blocks: {len(cbs3)}")
    filler_count = np.sum(cbs3[0] == -1)
    print(f"  Filler bits in first block: {filler_count}")
    print(f"  First 5 bits of block 0: {cbs3[0][:5]} (should contain -1)")

    print()
    print("="*70)
