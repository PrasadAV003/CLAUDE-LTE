"""
LTE Code Block Desegmentation - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteCodeBlockDesegment

MATLAB Compatibility:
- Concatenates code block segments into single output block
- Performs CRC-24B checking and removal (if C > 1)
- Removes filler bits from beginning of first code block
- Returns per-block CRC error indicators

Based on 3GPP TS 36.212 Section 5.1.2
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from crc_encode import lteCRCDecode


def get_segmentation_params(blklen: int) -> dict:
    """
    Calculate code block segmentation parameters

    Parameters:
        blklen: Transport block size (B bits)

    Returns:
        Dictionary with segmentation parameters (C, K+, K-, C+, C-, F)
    """
    # Valid turbo interleaver sizes from TS 36.212 Table 5.1.3-3
    K_table = [
        40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
        168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272,
        280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384,
        392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496,
        504, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704,
        720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928,
        944, 960, 976, 992, 1008, 1024, 1056, 1088, 1120, 1152, 1184, 1216,
        1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600,
        1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984,
        2016, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688,
        2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456,
        3520, 3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096, 4160, 4224,
        4288, 4352, 4416, 4480, 4544, 4608, 4672, 4736, 4800, 4864, 4928, 4992,
        5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504, 5568, 5632, 5696, 5760,
        5824, 5888, 5952, 6016, 6080, 6144
    ]

    Z = 6144  # Maximum code block size

    # B is the transport block size (input data size, no CRC added yet)
    B = blklen

    if B <= Z:
        # Single code block case
        L = 0
        C = 1
        B_prime = B

        # Find minimum K+ >= B'
        K_plus = None
        for K in K_table:
            if K >= B_prime:
                K_plus = K
                break

        if K_plus is None:
            raise ValueError(f"Block size {blklen} too large")

        K_minus = 0
        C_plus = 1
        C_minus = 0
    else:
        # Multiple code blocks
        L = 24  # CRC per code block
        C = int(np.ceil(B / (Z - L)))
        B_prime = B + C * L

        # Find minimum K+ such that C*K+ >= B'
        K_plus = None
        for K in K_table:
            if C * K >= B_prime:
                K_plus = K
                break

        if K_plus is None:
            raise ValueError(f"Block size {blklen} too large")

        # Find maximum K- < K+
        K_minus = 0
        for K in reversed(K_table):
            if K < K_plus:
                K_minus = K
                break

        # Calculate number of blocks of each size
        K_delta = K_plus - K_minus
        C_minus = int(np.floor((C * K_plus - B_prime) / K_delta))
        C_plus = C - C_minus

    # Calculate filler bits
    F = C_plus * K_plus + C_minus * K_minus - B_prime

    return {
        'C': C,           # Number of code blocks
        'K_plus': K_plus,   # Larger block size
        'K_minus': K_minus, # Smaller block size
        'C_plus': C_plus,   # Number of larger blocks
        'C_minus': C_minus, # Number of smaller blocks
        'F': F,             # Number of filler bits
        'L': L              # CRC length per block
    }


def lteCodeBlockDesegment(cbs: Union[np.ndarray, List[np.ndarray]],
                          blklen: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB lteCodeBlockDesegment equivalent - Code block desegmentation

    Performs inverse of code block segmentation by concatenating code blocks,
    checking and removing CRCs (if multiple blocks), and removing filler bits.

    Syntax:
        blk, err = lteCodeBlockDesegment(cbs)
        blk, err = lteCodeBlockDesegment(cbs, blklen)

    Parameters:
        cbs: Code block segments
             - Single vector: no CRC checking
             - Cell array with 1 element: no CRC checking
             - Cell array with >1 elements: each has 24B CRC to check/remove
        blklen: Original transport block length (optional)
                If provided, used to calculate filler bits to remove
                If not provided, no filler removal

    Returns:
        blk: Desegmented output block (int8 column vector)
             Filler bits and CRCs removed
        err: CRC error indicators (int8 vector)
             - If C > 1: length C, 0=pass, 1=fail for each block
             - If C = 1: empty array

    Examples:
        # Multiple blocks with CRC checking
        cbs = lteCodeBlockSegment(ones(6145, 1))
        blk, err = lteCodeBlockDesegment(cbs, 6145)

        # Single block, no CRC
        cbs = lteCodeBlockSegment(ones(100, 1))
        blk, err = lteCodeBlockDesegment(cbs, 100)

    Note:
        Performs inverse of lteCodeBlockSegment operation
    """
    # Convert input to list format for uniform processing
    if isinstance(cbs, np.ndarray):
        # Single vector input
        code_blocks = [cbs]
    else:
        # Cell array (list) input
        code_blocks = cbs

    C = len(code_blocks)

    # Determine segmentation parameters if blklen provided
    if blklen is not None and blklen > 0:
        params = get_segmentation_params(blklen)
        F = params['F']
        L = params['L']
        K_plus = params['K_plus']
        K_minus = params['K_minus']
        C_minus = params['C_minus']

        # Validate input dimensions
        expected_C = params['C']
        if C != expected_C:
            raise ValueError(f"Expected {expected_C} code blocks but got {C}")
    else:
        # No filler removal, no dimension validation
        F = 0
        L = 24 if C > 1 else 0
        # Determine K values from actual block sizes
        if C > 1:
            # Assume all blocks same size or two different sizes
            sizes = [len(cb) for cb in code_blocks]
            K_plus = max(sizes)
            K_minus = min(sizes)
            C_minus = sum(1 for s in sizes if s == K_minus)
        else:
            K_plus = len(code_blocks[0])
            K_minus = 0
            C_minus = 0

    # Process code blocks
    output_bits = []
    crc_errors = []

    for r in range(C):
        # Get current code block
        cb = np.array(code_blocks[r], dtype=np.int8)

        # Determine size for this block
        if C_minus > 0 and r < C_minus:
            K_r = K_minus
        else:
            K_r = K_plus

        # Validate block size
        if len(cb) != K_r:
            raise ValueError(f"Code block {r} has length {len(cb)}, expected {K_r}")

        # Remove filler bits from first block
        start_idx = F if r == 0 else 0

        # Check and remove CRC if multiple blocks
        if C > 1:
            # Extract data portion (after filler, includes CRC)
            data_with_crc = cb[start_idx:K_r]

            # Check and remove CRC
            # lteCRCDecode returns (data, err) where err is uint32
            decoded, err = lteCRCDecode(data_with_crc, '24B')
            # err == 0 means CRC passed, err != 0 means failure
            crc_errors.append(0 if err == 0 else 1)

            # Append decoded data (CRC already stripped by lteCRCDecode)
            output_bits.extend(decoded.tolist())
        else:
            # Single block - no CRC checking
            data_bits = cb[start_idx:]
            output_bits.extend(data_bits.tolist())

    # Convert to numpy arrays with proper types
    blk = np.array(output_bits, dtype=np.int8)

    if C > 1:
        err = np.array(crc_errors, dtype=np.int8)
    else:
        err = np.array([], dtype=np.int8)

    return blk, err


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE Code Block Desegmentation - MATLAB Compatible")
    print("="*70)
    print()

    try:
        from code_block_segment import lteCodeBlockSegment

        # Example 1: Large block requiring segmentation
        print("Example 1: Large block (6145 bits) with segmentation")
        print("-" * 70)

        # Create test data
        test_data = np.ones(6145, dtype=int)
        print(f"Original data: {len(test_data)} bits")

        # Segment
        cbs = lteCodeBlockSegment(test_data)
        print(f"Segmented into {len(cbs)} code blocks:")
        for i, cb in enumerate(cbs):
            print(f"  Block {i}: {len(cb)} bits")
        print()

        # Desegment
        blk, err = lteCodeBlockDesegment(cbs, 6145)
        print(f"Desegmented block: {len(blk)} bits")
        print(f"CRC errors: {err}")

        # Verify
        errors = np.sum(blk != test_data)
        if errors == 0:
            print("✓ PASS: Desegmentation successful")
        else:
            print(f"✗ FAIL: {errors} bit errors")
        print()

        # Example 2: Small block (no segmentation)
        print("Example 2: Small block (100 bits) without segmentation")
        print("-" * 70)

        test_data2 = np.zeros(100, dtype=int)
        print(f"Original data: {len(test_data2)} bits")

        cbs2 = lteCodeBlockSegment(test_data2)
        print(f"Segmented: {len(cbs2)} code block(s)")
        print()

        blk2, err2 = lteCodeBlockDesegment(cbs2, 100)
        print(f"Desegmented block: {len(blk2)} bits")
        print(f"CRC errors: {err2} (empty for single block)")

        errors2 = np.sum(blk2 != test_data2)
        if errors2 == 0:
            print("✓ PASS: Desegmentation successful")
        else:
            print(f"✗ FAIL: {errors2} bit errors")
        print()

    except ImportError:
        print("code_block_segment.py not found - skipping examples")

    print("="*70)
    print("Code block desegmentation with CRC checking and filler removal")
    print("="*70)
