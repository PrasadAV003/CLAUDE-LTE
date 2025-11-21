"""
LTE CRC Encode - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteCRCEncode

MATLAB Compatibility:
- Negative input bit values (-1) are interpreted as logical 0 for CRC calculation
- Supports CRC types: '8', '16', '24A', '24B'
- XOR masking applied MSB-first
- Filler bits preserved in output

Based on 3GPP TS 36.212 Section 5.1.1

Note: Code block segmentation is in separate module (code_block_segment.py)
"""

import numpy as np
from typing import Union, List

class LTE_CRC:
    """
    LTE CRC Calculation ONLY
    MATLAB-COMPATIBLE - Matches lteCRCEncode

    For code block segmentation, use code_block_segment.py
    """

    def __init__(self):
        # CRC Generator Polynomials (MSB first, length includes x^n term)

        # gCRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
        self.gCRC24A = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]

        # gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
        self.gCRC24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        # gCRC16(D) = D^16 + D^12 + D^5 + 1
        self.gCRC16 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        # gCRC8(D) = D^8 + D^7 + D^4 + D^3 + D + 1
        self.gCRC8 = [1, 1, 0, 0, 1, 1, 0, 1, 1]

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
            >>> crc = LTE_CRC()

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


# ============================================================================
# MATLAB-COMPATIBLE WRAPPER FUNCTION
# ============================================================================

def lteCRCEncode(blk, poly, mask=0):
    """
    MATLAB lteCRCEncode equivalent - CRC encoding only

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

        >>> # Example 3: Filler bits treated as 0
        >>> crc3 = lteCRCEncode(np.array([-1, -1, 1, 0, 1]), '24A')
        >>> len(crc3)
        29

    Note:
        For code block segmentation, use code_block_segment.py
    """
    crc_processor = LTE_CRC()
    return crc_processor.crc_attach(blk, crc_type=poly, mask=mask)


def lteCRCDecode(blkcrc, poly, mask=0):
    """
    MATLAB lteCRCDecode equivalent - CRC decoding and removal

    Checks input data vector for CRC error assuming the vector comprises
    a block of data with CRC bits attached. Returns data part and the
    logical difference (XOR) between attached CRC and recalculated CRC.

    Syntax:
        blk, err = lteCRCDecode(blkcrc, poly)
        blk, err = lteCRCDecode(blkcrc, poly, mask)

    Parameters:
        blkcrc: Input bit vector with CRC attached (column vector)
                Can contain soft values - will be converted to hard bits
        poly: CRC polynomial ('8', '16', '24A', '24B')
        mask: XOR mask value (optional, e.g., RNTI for masking)
              Applied MSB-first before returning in err

    Returns:
        blk: Data-only part of input (int8 column vector)
        err: Logical difference (XOR) between received and calculated CRC (uint32)
             If err != 0: either error occurred or CRC was masked
             If err == 0: CRC check passed

    Examples:
        >>> # Example 1: CRC decode without mask
        >>> rnti = 8
        >>> blkcrc = lteCRCEncode(np.ones(100), '24A', rnti)
        >>> blk, err = lteCRCDecode(blkcrc, '24A')
        >>> # err = 8 (equals RNTI because CRC was masked)

        >>> # Example 2: CRC decode with mask
        >>> blk, err = lteCRCDecode(blkcrc, '24A', rnti)
        >>> # err = 0 (mask cancels out)

        >>> # Example 3: HARQ soft combining scenario
        >>> # Soft bits from multiple transmissions combined
        >>> soft_bits = np.array([1.2, -0.8, 2.3, ...])  # LLRs
        >>> hard_bits = np.where(soft_bits >= 0, 0, 1)  # Hard decision
        >>> blk, err = lteCRCDecode(hard_bits, '24B')
        >>> if err == 0:
        >>>     print("CRC passed")

    Note:
        For HARQ chase combining, soft bits should be combined using
        lteRateRecoverTurbo with cbsbuffer parameter before turbo decoding.
        This function works on hard bits after decoding.
    """
    # Determine CRC length
    crc_lengths = {'8': 8, '16': 16, '24A': 24, '24B': 24}
    if poly not in crc_lengths:
        raise ValueError(f"Invalid CRC polynomial: {poly}")

    L = crc_lengths[poly]

    # Convert to numpy array and make hard decisions if needed
    blkcrc = np.array(blkcrc, dtype=float)

    # If input contains soft values, convert to hard bits
    # LLR convention: positive = 0, negative = 1
    # But also handle direct bit values (0 or 1)
    if np.any((blkcrc < 0) | (blkcrc > 1)):
        # Contains soft values (LLRs) - make hard decision
        blkcrc_hard = np.where(blkcrc >= 0, 0, 1).astype(np.int8)
    else:
        # Already hard bits
        blkcrc_hard = blkcrc.astype(np.int8)

    # Check length
    if len(blkcrc_hard) < L:
        raise ValueError(f"Input too short for {poly} CRC (need at least {L} bits)")

    # Split data and received CRC
    data_bits = blkcrc_hard[:-L]
    received_crc_bits = blkcrc_hard[-L:]

    # Calculate expected CRC on data portion (without mask)
    crc_processor = LTE_CRC()
    calculated_with_crc = crc_processor.crc_attach(data_bits, crc_type=poly, mask=0)
    calculated_crc_bits = calculated_with_crc[-L:]

    # Convert CRC bit arrays to integers for XOR comparison
    # MSB first (leftmost bit is most significant)
    received_crc_int = int(''.join(str(b) for b in received_crc_bits), 2)
    calculated_crc_int = int(''.join(str(b) for b in calculated_crc_bits), 2)

    # Compute XOR difference
    crc_diff = received_crc_int ^ calculated_crc_int

    # Apply mask (XOR with mask)
    err = np.uint32(crc_diff ^ mask)

    return data_bits, err


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

    print("="*70)
    print("For code block segmentation, use code_block_segment.py")
    print("="*70)
