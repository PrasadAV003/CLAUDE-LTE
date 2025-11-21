"""
Test lteCodeBlockSegment - Verify MATLAB Compatibility

MATLAB Documentation Tests:
1. Input 6144 bits → 1 block (no segmentation)
2. Input 6145 bits → 2 blocks [3072, 3136] (with segmentation)
3. Filler bits as -1 (NULL)
4. CRC24B appended only when B > 6144
"""

import numpy as np
from code_block_segment import lteCodeBlockSegment, LTE_CodeBlockSegmentation

def test_no_segmentation():
    """
    MATLAB Example 1:
    cbs1 = lteCodeBlockSegment(ones(6144,1))
    Result: 1×1 cell array {6144×1 int8}
    """
    print("="*70)
    print("TEST 1: No Segmentation (B = 6144)")
    print("="*70)

    cbs = lteCodeBlockSegment(np.ones(6144, dtype=int))

    print(f"Input length: 6144")
    print(f"Number of blocks: {len(cbs)}")
    print(f"Block 0 size: {len(cbs[0])}")
    print(f"Data type: {cbs[0].dtype}")

    # Verify
    if len(cbs) == 1 and len(cbs[0]) == 6144 and cbs[0].dtype == np.int8:
        print("✓ PASS: Matches MATLAB (no segmentation)")
    else:
        print("✗ FAIL")

    # Check no CRC appended (B <= 6144)
    # All bits should be 1 (no CRC, no filler)
    if np.all(cbs[0] == 1):
        print("✓ PASS: No CRC appended (B <= 6144)")
    else:
        print("✗ FAIL: Expected all 1s")

    print()


def test_with_segmentation():
    """
    MATLAB Example 2:
    cbs2 = lteCodeBlockSegment(ones(6145,1))
    Result: 1×2 cell array {3072×1 int8} {3136×1 int8}
    """
    print("="*70)
    print("TEST 2: With Segmentation (B = 6145)")
    print("="*70)

    cbs = lteCodeBlockSegment(np.ones(6145, dtype=int))

    print(f"Input length: 6145")
    print(f"Number of blocks: {len(cbs)}")
    print(f"Block sizes: {[len(cb) for cb in cbs]}")
    print(f"Expected: [3072, 3136]")
    print(f"Data type: {cbs[0].dtype}")

    # Verify
    expected_sizes = [3072, 3136]
    actual_sizes = [len(cb) for cb in cbs]

    if len(cbs) == 2 and actual_sizes == expected_sizes:
        print("✓ PASS: Matches MATLAB exactly")
    else:
        print(f"✗ FAIL: Expected {expected_sizes}, got {actual_sizes}")

    # Check CRC24B appended (B > 6144)
    # Each block should have some non-1 values (CRC24B bits)
    crc_block0 = cbs[0][-24:]  # Last 24 bits should be CRC
    crc_block1 = cbs[1][-24:]

    # CRC of all 1s should not be all 1s
    has_crc0 = not np.all(crc_block0 == 1)
    has_crc1 = not np.all(crc_block1 == 1)

    if has_crc0 and has_crc1:
        print("✓ PASS: CRC24B appended to both blocks (B > 6144)")
    else:
        print("✗ FAIL: CRC24B should be appended")

    print()


def test_filler_bits():
    """
    Test: Filler bits represented as -1 and prepended to first block
    """
    print("="*70)
    print("TEST 3: Filler Bits as -1 (NULL)")
    print("="*70)

    # Input size that requires filler bits
    input_size = 6200
    cbs = lteCodeBlockSegment(np.ones(input_size, dtype=int))

    print(f"Input length: {input_size}")
    print(f"Number of blocks: {len(cbs)}")

    # Count filler bits in first block
    filler_count = np.sum(cbs[0] == -1)
    print(f"Filler bits in block 0: {filler_count}")
    print(f"First 10 bits of block 0: {cbs[0][:10]}")

    # Verify filler bits are at the beginning
    if filler_count > 0:
        first_filler_positions = cbs[0][:filler_count]
        if np.all(first_filler_positions == -1):
            print("✓ PASS: Filler bits (-1) prepended to first block")
        else:
            print("✗ FAIL: Filler bits should be -1")
    else:
        print("⚠ WARNING: No filler bits for this input size")

    # Check no filler bits in other blocks
    if len(cbs) > 1:
        filler_in_block1 = np.sum(cbs[1] == -1)
        if filler_in_block1 == 0:
            print("✓ PASS: No filler bits in block 1")
        else:
            print("✗ FAIL: Filler bits should only be in first block")

    print()


def test_detailed_comparison():
    """
    Detailed test comparing internal implementation vs wrapper
    """
    print("="*70)
    print("TEST 4: Internal Implementation vs MATLAB Wrapper")
    print("="*70)

    processor = LTE_CodeBlockSegmentation()
    input_bits = np.ones(6145, dtype=int)

    # Internal implementation
    code_blocks_internal, info = processor.segment(input_bits)

    # MATLAB wrapper
    code_blocks_wrapper = lteCodeBlockSegment(input_bits)

    print(f"Internal returns: ({len(code_blocks_internal)} blocks, info dict)")
    print(f"Wrapper returns: {len(code_blocks_wrapper)} blocks")
    print()

    # Compare outputs
    match = True
    for i, (internal, wrapper) in enumerate(zip(code_blocks_internal, code_blocks_wrapper)):
        if not np.array_equal(internal.astype(np.int8), wrapper):
            match = False
            print(f"✗ Block {i} mismatch")
        else:
            print(f"✓ Block {i} matches (size: {len(wrapper)})")

    if match:
        print("\n✓ PASS: Wrapper matches internal implementation")
    else:
        print("\n✗ FAIL: Wrapper does not match")

    print(f"\nSegmentation info:")
    print(f"  C (blocks): {info['C']}")
    print(f"  L (CRC length): {info['L']}")
    print(f"  F (filler bits): {info['F']}")
    print(f"  K+ : {info['K_plus']}")
    print(f"  K- : {info['K_minus']}")

    print()


def test_crc_presence():
    """
    Test CRC24B presence logic
    """
    print("="*70)
    print("TEST 5: CRC24B Appending Logic")
    print("="*70)

    test_cases = [
        (6144, False, "B = 6144 (at boundary)"),
        (6143, False, "B = 6143 (below boundary)"),
        (6145, True, "B = 6145 (above boundary)"),
        (10000, True, "B = 10000 (well above)")
    ]

    for input_size, should_have_crc, description in test_cases:
        cbs = lteCodeBlockSegment(np.ones(input_size, dtype=int))

        print(f"\n{description}:")
        print(f"  Blocks: {len(cbs)}")

        if should_have_crc:
            # Check if last 24 bits are different from data (indicating CRC)
            # For all-ones input, CRC should not be all ones
            has_crc = not np.all(cbs[0][-24:] == 1)
            if has_crc:
                print(f"  ✓ CRC24B appended (as expected)")
            else:
                print(f"  ✗ CRC24B should be appended")
        else:
            # Check if all bits are 1 (no CRC, no filler for these sizes)
            no_crc = np.all(cbs[0] == 1) or np.any(cbs[0] == -1)  # All 1s or has filler
            if no_crc:
                print(f"  ✓ No CRC appended (as expected)")
            else:
                print(f"  ✗ CRC should not be appended")

    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MATLAB lteCodeBlockSegment COMPATIBILITY TEST SUITE")
    print("="*70)
    print()

    test_no_segmentation()
    test_with_segmentation()
    test_filler_bits()
    test_detailed_comparison()
    test_crc_presence()

    print("="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
