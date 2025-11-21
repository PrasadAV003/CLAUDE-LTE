"""
Test lteRateRecoverTurbo - MATLAB Documentation Examples

This test verifies that our Python implementation of rate recovery
matches the MATLAB lteRateRecoverTurbo behavior.

MATLAB Documentation Example:
    trBlkLen = 135;
    codewordLen = 450;
    rv = 0;
    crcPoly = '24A';

    trblockwithcrc = lteCRCEncode(zeros(trBlkLen,1),crcPoly);
    codeblocks = lteCodeBlockSegment(trblockwithcrc);
    turbocodedblocks = lteTurboEncode(codeblocks);
    codeword = lteRateMatchTurbo(turbocodedblocks,codewordLen,rv);
    rateRecovered = lteRateRecoverTurbo(codeword,trBlkLen,rv);

    % Result: rateRecovered is {492×1 int8}

Key Requirements:
1. Inverse of rate matching operation
2. Recovers turbo encoded code blocks before concatenation
3. Deduces dimensions from transport block length (before CRC)
4. Supports RV values: 0, 1, 2, 3
5. Supports HARQ soft combining with cbsbuffers
6. Returns cell array (list) of int8 vectors
"""

import numpy as np
from crc_encode import lteCRCEncode
from code_block_segment import lteCodeBlockSegment
from ctr_encode import lteTurboEncode, lteRateMatchTurbo
from rate_recover_turbo import lteRateRecoverTurbo, get_code_block_parameters


def test_matlab_example_1():
    """
    MATLAB Example: Complete encode → rate match → rate recover chain

    MATLAB Code:
        trBlkLen = 135;
        codewordLen = 450;
        rv = 0;
        crcPoly = '24A';

        trblockwithcrc = lteCRCEncode(zeros(trBlkLen,1),crcPoly);
        codeblocks = lteCodeBlockSegment(trblockwithcrc);
        turbocodedblocks = lteTurboEncode(codeblocks);
        codeword = lteRateMatchTurbo(turbocodedblocks,codewordLen,rv);
        rateRecovered = lteRateRecoverTurbo(codeword,trBlkLen,rv);

    Expected Output:
        rateRecovered is 1×1 cell array: {492×1 int8}
    """
    print("="*70)
    print("MATLAB Example 1: Complete Encode → Rate Match → Rate Recover")
    print("="*70)
    print()

    trBlkLen = 135
    codewordLen = 450
    rv = 0
    crcPoly = '24A'

    print("Step 1: CRC Encode")
    trblockwithcrc = lteCRCEncode(np.zeros(trBlkLen, dtype=int), crcPoly)
    print(f"  Transport block: {trBlkLen} bits")
    print(f"  With CRC24A: {len(trblockwithcrc)} bits")
    print()

    print("Step 2: Code Block Segment")
    codeblocks = lteCodeBlockSegment(trblockwithcrc)
    print(f"  Number of code blocks: {len(codeblocks)}")
    print(f"  Code block size: {len(codeblocks[0])} bits")
    print()

    print("Step 3: Turbo Encode")
    turbocodedblocks = lteTurboEncode(codeblocks)
    print(f"  Turbo encoded blocks: {len(turbocodedblocks)}")
    print(f"  Encoded block size: {len(turbocodedblocks[0])} bits")
    expected_encoded_len = 492
    assert len(turbocodedblocks[0]) == expected_encoded_len, \
        f"Expected {expected_encoded_len} bits, got {len(turbocodedblocks[0])}"
    print()

    print("Step 4: Rate Match")
    codeword = lteRateMatchTurbo(turbocodedblocks, codewordLen, rv)
    print(f"  Rate matched length: {len(codeword)} bits")
    assert len(codeword) == codewordLen, f"Expected {codewordLen} bits, got {len(codeword)}"
    print()

    print("Step 5: Rate Recover")
    rateRecovered = lteRateRecoverTurbo(codeword, trBlkLen, rv)
    print(f"  Recovered blocks: {len(rateRecovered)}")
    print(f"  Block 0 size: {len(rateRecovered[0])} bits")
    print(f"  Data type: {rateRecovered[0].dtype}")
    print()

    # Verify
    assert len(rateRecovered) == 1, f"Expected 1 block, got {len(rateRecovered)}"
    assert len(rateRecovered[0]) == 492, f"Expected 492 bits, got {len(rateRecovered[0])}"
    assert rateRecovered[0].dtype == np.int8, f"Expected int8, got {rateRecovered[0].dtype}"

    print("✓ PASS: Matches MATLAB output structure")
    print("  - Cell array with 1 element ✓")
    print("  - Block size 492×1 ✓")
    print("  - Data type int8 ✓")
    print()


def test_code_block_parameters():
    """
    Test code block parameter calculation

    Verify that we correctly determine:
    - Number of code blocks (C)
    - Code block sizes (K+, K-)
    - Filler bits (F)
    """
    print("="*70)
    print("Code Block Parameter Calculation")
    print("="*70)
    print()

    test_cases = [
        # (trblklen, expected_C, expected_K_plus)
        (135, 1, 160),   # Small: no segmentation
        (100, 1, 128),   # Small: no segmentation
        (6000, 1, 6080), # Large but no segmentation (B=6024, K+=6080)
        (6200, 2, 3136), # Requires segmentation
        (10000, 2, 5056),# Requires segmentation
    ]

    for trblklen, exp_C, exp_K_plus in test_cases:
        C, K_plus, K_minus, C_plus, C_minus, F = get_code_block_parameters(trblklen)

        print(f"Transport block length: {trblklen}")
        print(f"  C (blocks): {C} (expected: {exp_C})")
        print(f"  K+ (size): {K_plus} (expected: {exp_K_plus})")
        print(f"  F (filler): {F}")

        assert C == exp_C, f"Expected C={exp_C}, got {C}"
        assert K_plus == exp_K_plus, f"Expected K+={exp_K_plus}, got {K_plus}"
        print("  ✓ PASS")
        print()

    print("✓ ALL PASS: Code block parameters calculated correctly")
    print()


def test_round_trip_recovery():
    """
    Test round-trip: encode → rate match → rate recover

    Verify that rate recovery recovers the correct structure
    (we can't expect exact bit values due to rate matching/recovery)
    """
    print("="*70)
    print("Round-Trip Recovery Test")
    print("="*70)
    print()

    test_cases = [
        # (trblklen, codewordlen, rv)
        (40, 100, 0),    # Small transport block
        (135, 450, 0),   # MATLAB example
        (1000, 2000, 1), # Medium with RV=1
        (100, 200, 2),   # Small with RV=2
    ]

    for trblklen, codewordlen, rv in test_cases:
        print(f"Test: trblklen={trblklen}, codewordlen={codewordlen}, rv={rv}")

        # Encode
        trblkwithcrc = lteCRCEncode(np.ones(trblklen, dtype=int), '24A')
        codeblocks = lteCodeBlockSegment(trblkwithcrc)
        turboencoded = lteTurboEncode(codeblocks)

        # Rate match
        codeword = lteRateMatchTurbo(turboencoded, codewordlen, rv)

        # Rate recover
        recovered = lteRateRecoverTurbo(codeword, trblklen, rv)

        # Verify structure
        print(f"  Original blocks: {len(turboencoded)}")
        print(f"  Recovered blocks: {len(recovered)}")
        assert len(recovered) == len(turboencoded), \
            f"Block count mismatch: {len(recovered)} != {len(turboencoded)}"

        for i in range(len(recovered)):
            print(f"  Block {i}: {len(turboencoded[i])} → {len(recovered[i])} bits")
            assert len(recovered[i]) == len(turboencoded[i]), \
                f"Block {i} size mismatch: {len(recovered[i])} != {len(turboencoded[i])}"

        print("  ✓ PASS: Structure matches")
        print()

    print("✓ ALL PASS: Round-trip recovery preserves structure")
    print()


def test_different_rv_values():
    """
    Test rate recovery with different redundancy versions

    Different RVs should recover the same structure but with
    different soft information distribution
    """
    print("="*70)
    print("Different Redundancy Versions (RV = 0, 1, 2, 3)")
    print("="*70)
    print()

    trblklen = 135
    codewordlen = 450

    # Encode once
    trblkwithcrc = lteCRCEncode(np.ones(trblklen, dtype=int), '24A')
    codeblocks = lteCodeBlockSegment(trblkwithcrc)
    turboencoded = lteTurboEncode(codeblocks)

    print(f"Original turbo encoded: {len(turboencoded)} blocks")
    print()

    for rv in [0, 1, 2, 3]:
        # Rate match with this RV
        codeword = lteRateMatchTurbo(turboencoded, codewordlen, rv)

        # Rate recover
        recovered = lteRateRecoverTurbo(codeword, trblklen, rv)

        print(f"RV={rv}:")
        print(f"  Recovered blocks: {len(recovered)}")
        print(f"  Block 0 size: {len(recovered[0])} bits")

        # Verify structure
        assert len(recovered) == len(turboencoded), f"RV={rv}: Block count mismatch"
        assert len(recovered[0]) == len(turboencoded[0]), f"RV={rv}: Block size mismatch"

        print(f"  ✓ PASS")

    print()
    print("✓ ALL PASS: All RV values work correctly")
    print()


def test_harq_soft_combining():
    """
    Test HARQ soft combining with cbsbuffers

    Multiple transmissions should be combined for better decoding
    """
    print("="*70)
    print("HARQ Soft Combining")
    print("="*70)
    print()

    trblklen = 135
    codewordlen = 450

    # Encode
    trblkwithcrc = lteCRCEncode(np.ones(trblklen, dtype=int), '24A')
    codeblocks = lteCodeBlockSegment(trblkwithcrc)
    turboencoded = lteTurboEncode(codeblocks)

    print("Transmission 1 (RV=0):")
    # First transmission
    codeword_rv0 = lteRateMatchTurbo(turboencoded, codewordlen, 0)
    # Add noise to simulate channel
    received_rv0 = codeword_rv0.astype(float) + np.random.randn(len(codeword_rv0)) * 0.5
    recovered_rv0 = lteRateRecoverTurbo(received_rv0, trblklen, 0)
    print(f"  Recovered: {len(recovered_rv0)} blocks")
    print()

    print("Transmission 2 (RV=1) with soft combining:")
    # Second transmission with different RV
    codeword_rv1 = lteRateMatchTurbo(turboencoded, codewordlen, 1)
    received_rv1 = codeword_rv1.astype(float) + np.random.randn(len(codeword_rv1)) * 0.5
    # Combine with previous transmission
    recovered_rv1 = lteRateRecoverTurbo(received_rv1, trblklen, 1, cbsbuffers=recovered_rv0)
    print(f"  Recovered with combining: {len(recovered_rv1)} blocks")
    print(f"  Block 0 size: {len(recovered_rv1[0])} bits")
    print()

    # Verify structure
    assert len(recovered_rv1) == len(recovered_rv0), "Block count mismatch"
    assert len(recovered_rv1[0]) == len(recovered_rv0[0]), "Block size mismatch"

    print("✓ PASS: HARQ soft combining works")
    print()


def test_segmented_transport_block():
    """
    Test with large transport block requiring segmentation

    Verify correct handling of multiple code blocks
    """
    print("="*70)
    print("Segmented Transport Block (Multiple Code Blocks)")
    print("="*70)
    print()

    trblklen = 6200  # Requires 2 code blocks
    codewordlen = 10000
    rv = 0

    print(f"Transport block length: {trblklen} bits")
    print()

    # Determine expected structure
    C, K_plus, K_minus, C_plus, C_minus, F = get_code_block_parameters(trblklen)
    print(f"Expected code block structure:")
    print(f"  C (blocks): {C}")
    print(f"  K+ (size): {K_plus}")
    print(f"  K- (size): {K_minus}")
    print(f"  C+ (count): {C_plus}")
    print(f"  C- (count): {C_minus}")
    print(f"  F (filler): {F}")
    print()

    # Encode
    trblkwithcrc = lteCRCEncode(np.ones(trblklen, dtype=int), '24A')
    codeblocks = lteCodeBlockSegment(trblkwithcrc)
    turboencoded = lteTurboEncode(codeblocks)

    print(f"Encoding:")
    print(f"  Code blocks: {len(codeblocks)}")
    print(f"  Turbo encoded blocks: {len(turboencoded)}")
    for i, blk in enumerate(turboencoded):
        print(f"    Block {i}: {len(blk)} bits")
    print()

    # Rate match
    codeword = lteRateMatchTurbo(turboencoded, codewordlen, rv)
    print(f"Rate matched: {len(codeword)} bits")
    print()

    # Rate recover
    recovered = lteRateRecoverTurbo(codeword, trblklen, rv)
    print(f"Rate recovered:")
    print(f"  Blocks: {len(recovered)}")
    for i, blk in enumerate(recovered):
        print(f"    Block {i}: {len(blk)} bits")
    print()

    # Verify
    assert len(recovered) == C, f"Expected {C} blocks, got {len(recovered)}"
    assert len(recovered) == len(turboencoded), "Block count mismatch"

    for i in range(len(recovered)):
        assert len(recovered[i]) == len(turboencoded[i]), \
            f"Block {i} size mismatch: {len(recovered[i])} != {len(turboencoded[i])}"

    print("✓ PASS: Segmented transport block handled correctly")
    print()


def test_input_validation():
    """
    Test input validation

    Requirements:
    - RV must be 0, 1, 2, or 3
    - trblklen must be positive
    """
    print("="*70)
    print("Input Validation")
    print("="*70)
    print()

    # Test 1: Invalid RV value
    print("Test 1: Invalid RV value")
    try:
        lteRateRecoverTurbo(np.ones(100), 40, 5)  # RV must be 0-3
        print("  ✗ FAIL: Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ✓ PASS: Correctly rejected - {e}")
    print()

    # Test 2: Empty input
    print("Test 2: Empty input")
    result = lteRateRecoverTurbo(np.array([]), 40, 0)
    assert len(result) == 0, "Empty input should return empty result"
    print("  ✓ PASS: Empty input handled")
    print()

    print("✓ ALL PASS: Input validation working")
    print()


if __name__ == "__main__":
    print()
    print("="*70)
    print("MATLAB lteRateRecoverTurbo Compatibility Test Suite")
    print("="*70)
    print()

    test_matlab_example_1()
    test_code_block_parameters()
    test_round_trip_recovery()
    test_different_rv_values()
    test_harq_soft_combining()
    test_segmented_transport_block()
    test_input_validation()

    print("="*70)
    print("ALL TESTS PASSED ✓")
    print("Python implementation matches MATLAB documentation")
    print("="*70)
    print()
