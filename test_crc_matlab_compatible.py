"""
Test CRC Implementation - Verify MATLAB lteCRCEncode Compatibility

Tests key MATLAB behaviors:
1. CRC calculation on all-zero input
2. CRC with XOR masking (MSB-first)
3. CRC with -1 (NULL/filler bits) treated as 0
"""

import numpy as np
from lte_rate_matching import LTE_CRC_CodeBlockSegmentation

def test_crc_all_zeros():
    """
    MATLAB Example 1:
    crc1 = lteCRCEncode(zeros(100,1),'24A');
    Result: All zeros (124 bits total: 100 data + 24 CRC)
    """
    print("="*70)
    print("TEST 1: CRC24A on All-Zero Input (MATLAB Example 1)")
    print("="*70)

    processor = LTE_CRC_CodeBlockSegmentation()

    # 100 zeros input
    input_bits = np.zeros(100, dtype=int)

    # Attach CRC24A
    output = processor.crc_attach(input_bits, crc_type='CRC24A', mask=0)

    print(f"Input: {len(input_bits)} bits (all zeros)")
    print(f"Output: {len(output)} bits")
    print(f"CRC bits: {output[100:].tolist()}")

    # Verify all zeros
    if np.all(output == 0):
        print("✓ PASS: CRC of all zeros is all zeros (matches MATLAB)")
    else:
        print("✗ FAIL: Expected all zeros")

    print()


def test_crc_with_mask():
    """
    MATLAB Example 2:
    crc2 = lteCRCEncode(zeros(100,1),'24A',1);
    Result: All zeros except last bit = 1 (MSB-first masking)
    """
    print("="*70)
    print("TEST 2: CRC24A with XOR Mask = 1 (MATLAB Example 2)")
    print("="*70)

    processor = LTE_CRC_CodeBlockSegmentation()

    # 100 zeros input
    input_bits = np.zeros(100, dtype=int)

    # Attach CRC24A with mask = 1
    output = processor.crc_attach(input_bits, crc_type='CRC24A', mask=1)

    print(f"Input: {len(input_bits)} bits (all zeros)")
    print(f"Mask: 1 (binary: {'0'*23}1)")
    print(f"Output: {len(output)} bits")
    print(f"Last 10 CRC bits: {output[-10:].tolist()}")

    # Verify: should be all zeros except last bit
    expected = np.zeros(124, dtype=int)
    expected[-1] = 1  # MSB of mask=1 is 0, LSB is 1

    if np.array_equal(output, expected):
        print("✓ PASS: Mask applied MSB-first (matches MATLAB)")
    else:
        print("✗ FAIL: Masking order incorrect")
        print(f"Expected last 10: {expected[-10:].tolist()}")
        print(f"Got last 10: {output[-10:].tolist()}")

    print()


def test_crc_with_filler_bits():
    """
    Test: CRC calculation with -1 (filler bits) treated as 0

    MATLAB behavior: "negative input bit values are interpreted as logical 0"

    Test cases:
    1. Input with -1 should give same CRC as input with 0 in same positions
    2. Filler bits (-1) preserved in output, CRC appended after
    """
    print("="*70)
    print("TEST 3: CRC with Filler Bits (-1 treated as 0)")
    print("="*70)

    processor = LTE_CRC_CodeBlockSegmentation()

    # Test case: First 5 bits are filler (-1), rest are data
    input_with_filler = np.array([-1, -1, -1, -1, -1, 1, 0, 1, 1, 0], dtype=int)
    input_with_zeros = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 0], dtype=int)

    # Calculate CRC for both
    output_filler = processor.crc_attach(input_with_filler, crc_type='CRC24A', mask=0)
    output_zeros = processor.crc_attach(input_with_zeros, crc_type='CRC24A', mask=0)

    print(f"Input with filler: {input_with_filler.tolist()}")
    print(f"Input with zeros:  {input_with_zeros.tolist()}")
    print()
    print(f"Output with filler (first 10): {output_filler[:10].tolist()}")
    print(f"Output with zeros (first 10):  {output_zeros[:10].tolist()}")
    print()

    # CRC should be identical
    crc_filler = output_filler[10:]
    crc_zeros = output_zeros[10:]

    print(f"CRC from filler input: {crc_filler[:8].tolist()}... (24 bits)")
    print(f"CRC from zeros input:  {crc_zeros[:8].tolist()}... (24 bits)")

    if np.array_equal(crc_filler, crc_zeros):
        print("✓ PASS: -1 treated as 0 for CRC calculation (matches MATLAB)")
    else:
        print("✗ FAIL: CRC should be identical")

    # Verify filler bits preserved in output
    if np.all(output_filler[:5] == -1):
        print("✓ PASS: Filler bits (-1) preserved in output")
    else:
        print("✗ FAIL: Filler bits should remain -1 in output")

    print()


def test_code_block_segmentation():
    """
    Test: Code block segmentation with filler bits

    Verify:
    1. Filler bits set to -1 in first code block
    2. CRC24B calculated correctly (treating -1 as 0)
    """
    print("="*70)
    print("TEST 4: Code Block Segmentation with Filler Bits")
    print("="*70)

    processor = LTE_CRC_CodeBlockSegmentation()

    # Small transport block that requires segmentation
    tb_size = 6200  # Slightly larger than Z=6144
    input_bits = np.random.randint(0, 2, tb_size)

    # Segment
    code_blocks, seg_info = processor.code_block_segmentation(input_bits)

    print(f"Transport Block Size: {tb_size} bits")
    print(f"Number of Code Blocks: {seg_info['C']}")
    print(f"Filler Bits (F): {seg_info['F']}")
    print(f"K+: {seg_info['K_plus']}")
    print(f"K-: {seg_info['K_minus']}")
    print()

    # Check first code block
    if seg_info['F'] > 0:
        cb0 = code_blocks[0]
        filler_count = seg_info['F']

        print(f"First Code Block:")
        print(f"  Total length: {len(cb0)} bits")
        print(f"  First {filler_count} bits (filler): {cb0[:min(10, filler_count)].tolist()}...")

        # Verify filler bits are -1
        if np.all(cb0[:filler_count] == -1):
            print(f"  ✓ PASS: Filler bits set to -1")
        else:
            print(f"  ✗ FAIL: Filler bits should be -1")
            print(f"  Found: {np.unique(cb0[:filler_count])}")

        # Verify CRC24B is attached
        if seg_info['L'] == 24:
            crc_bits = cb0[-24:]
            if np.all(crc_bits >= 0):  # CRC should never be -1
                print(f"  ✓ PASS: CRC24B attached (no -1 in CRC)")
            else:
                print(f"  ✗ FAIL: CRC should not contain -1")
    else:
        print("  No filler bits in this test case")

    print()


def test_all_crc_types():
    """
    Test: All CRC polynomial types

    MATLAB supports: '8', '16', '24A', '24B'
    """
    print("="*70)
    print("TEST 5: All CRC Polynomial Types")
    print("="*70)

    processor = LTE_CRC_CodeBlockSegmentation()

    input_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=int)

    crc_types = [
        ('CRC8', 8),
        ('CRC16', 16),
        ('CRC24A', 24),
        ('CRC24B', 24)
    ]

    print(f"Input: {input_bits.tolist()}")
    print()

    for crc_type, expected_len in crc_types:
        try:
            output = processor.crc_attach(input_bits, crc_type=crc_type, mask=0)
            crc_bits = output[len(input_bits):]

            print(f"{crc_type:8}: {len(crc_bits)} CRC bits - {crc_bits.tolist()}")

            if len(crc_bits) == expected_len:
                print(f"         ✓ PASS: Correct length")
            else:
                print(f"         ✗ FAIL: Expected {expected_len} bits")
        except Exception as e:
            print(f"{crc_type:8}: ✗ FAIL - {e}")
        print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MATLAB lteCRCEncode COMPATIBILITY TEST SUITE")
    print("="*70)
    print()

    test_crc_all_zeros()
    test_crc_with_mask()
    test_crc_with_filler_bits()
    test_code_block_segmentation()
    test_all_crc_types()

    print("="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
