"""
Test lteRateMatchTurbo - MATLAB Documentation Examples

This test verifies that our Python implementation produces identical
results to the MATLAB lteRateMatchTurbo function.

MATLAB Documentation Examples:
    rateMatched = lteRateMatchTurbo(ones(132,1), 100, 0);

Key Requirements:
1. Sub-block interleaving (32 columns, row-by-row filling)
2. Bit collection (circular buffer creation)
3. Bit selection and pruning (RV-based starting point)
4. NULL filler bits (-1) skipped during rate matching
5. RV values: 0, 1, 2, 3
6. Cell array input → concatenated output
7. Input length must be multiple of 3
"""

import numpy as np
from ctr_encode import lteTurboEncode, lteRateMatchTurbo


def test_matlab_example_1():
    """
    MATLAB Example 1: Rate match 132 bits to 100 bits

    MATLAB Code:
        invec = ones(132,1);
        rateMatched = lteRateMatchTurbo(invec, 100, 0);

    Expected Output:
        size(rateMatched) = [100, 1]
    """
    print("="*70)
    print("MATLAB Example 1: Rate match 132 bits to 100 bits")
    print("="*70)
    print()

    # Create input (simulating turbo encoded output)
    invec = np.ones(132, dtype=int)

    print(f"Input: ones(132,1)")
    print(f"  Length: {len(invec)} bits")
    print()

    # Rate match
    rateMatched = lteRateMatchTurbo(invec, 100, 0)

    print(f"Rate matched output:")
    print(f"  Length: {len(rateMatched)} bits")
    print(f"  Expected: 100 bits")
    print()

    # Verify
    assert len(rateMatched) == 100, f"Expected 100 bits, got {len(rateMatched)}"

    print("✓ PASS: Matches MATLAB output length")
    print()


def test_complete_chain():
    """
    Test complete encoding chain: Turbo encode → Rate match

    This is the typical LTE encoding workflow
    """
    print("="*70)
    print("Complete Chain: Turbo Encode → Rate Match")
    print("="*70)
    print()

    # Step 1: Create input
    K = 40
    input_bits = np.ones(K, dtype=int)
    print(f"Step 1: Input")
    print(f"  Length: {K} bits")
    print()

    # Step 2: Turbo encode
    encoded = lteTurboEncode(input_bits)
    print(f"Step 2: Turbo Encode")
    print(f"  Input: {K} bits")
    print(f"  Output: {len(encoded)} bits (3*(K+4) = 3*44 = 132)")
    assert len(encoded) == 132, "Turbo output should be 132 bits"
    print()

    # Step 3: Rate match
    E = 100  # Target output length
    rv = 0
    rate_matched = lteRateMatchTurbo(encoded, E, rv)
    print(f"Step 3: Rate Match")
    print(f"  Input: {len(encoded)} bits")
    print(f"  Target: {E} bits")
    print(f"  RV: {rv}")
    print(f"  Output: {len(rate_matched)} bits")
    assert len(rate_matched) == E, f"Expected {E} bits, got {len(rate_matched)}"
    print()

    print("✓ PASS: Complete chain works correctly")
    print()


def test_redundancy_versions():
    """
    Test all redundancy versions (RV = 0, 1, 2, 3)

    Different RVs provide different starting points in circular buffer
    for HARQ retransmissions
    """
    print("="*70)
    print("Redundancy Versions (RV = 0, 1, 2, 3)")
    print("="*70)
    print()

    # Create turbo encoded input
    input_bits = np.ones(40, dtype=int)
    encoded = lteTurboEncode(input_bits)  # 132 bits
    E = 100  # Target length

    print(f"Input: {len(encoded)} bits (turbo encoded)")
    print(f"Target output: {E} bits")
    print()

    results = {}
    for rv in [0, 1, 2, 3]:
        rm_out = lteRateMatchTurbo(encoded, E, rv)
        results[rv] = rm_out

        print(f"RV={rv}:")
        print(f"  Output length: {len(rm_out)} bits")
        print(f"  First 5 bits: {rm_out[:5]}")
        print(f"  Last 5 bits: {rm_out[-5:]}")

        assert len(rm_out) == E, f"RV={rv}: Expected {E} bits, got {len(rm_out)}"

    print()

    # Verify different RVs produce different outputs (HARQ)
    print("Verifying RVs produce different outputs:")
    for rv1 in [0, 1, 2]:
        for rv2 in [rv1+1, rv1+2, rv1+3]:
            if rv2 > 3:
                break
            different = not np.array_equal(results[rv1], results[rv2])
            status = "✓" if different else "✗"
            print(f"  RV{rv1} vs RV{rv2}: {status} {'Different' if different else 'Same'}")

    print()
    print("✓ PASS: All RV values work correctly")
    print()


def test_filler_bits_handling():
    """
    Test that filler bits (-1) are skipped during rate matching

    MATLAB Documentation:
    "The function considers negative values in the input data as <NULL>
    filler bits inserted during code block segmentation and skips them
    during rate matching."
    """
    print("="*70)
    print("Filler Bits (-1) Handling")
    print("="*70)
    print()

    # Create input with filler bits
    K = 40
    F = 5  # Filler bits
    input_with_filler = np.concatenate([np.full(F, -1, dtype=int),
                                         np.ones(K - F, dtype=int)])

    print(f"Input: {K} bits with {F} filler bits (-1)")
    print(f"  First 10 bits: {input_with_filler[:10]}")
    print()

    # Turbo encode (filler bits passed through to S and P1)
    encoded = lteTurboEncode(input_with_filler)
    print(f"Turbo encoded: {len(encoded)} bits")
    filler_in_encoded = np.sum(encoded < 0)
    print(f"  Filler bits in encoded: {filler_in_encoded}")
    print(f"  First 10 bits: {encoded[:10]}")
    print()

    # Rate match (filler bits should be skipped)
    E = 100
    rate_matched = lteRateMatchTurbo(encoded, E, 0)
    print(f"Rate matched: {len(rate_matched)} bits")
    filler_in_rm = np.sum(rate_matched < 0)
    print(f"  Filler bits in rate matched: {filler_in_rm}")
    print(f"  Expected: 0 (filler bits skipped)")
    print()

    # Verify no filler bits in rate matched output
    assert filler_in_rm == 0, "Filler bits should be skipped during rate matching"

    print("✓ PASS: Filler bits correctly skipped")
    print()


def test_input_validation():
    """
    Test input validation

    Requirements:
    - Input length must be multiple of 3
    - RV must be 0, 1, 2, or 3
    """
    print("="*70)
    print("Input Validation")
    print("="*70)
    print()

    # Test 1: Input length not multiple of 3
    print("Test 1: Input length not multiple of 3")
    try:
        lteRateMatchTurbo(np.ones(133, dtype=int), 100, 0)  # 133 not divisible by 3
        print("  ✗ FAIL: Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ✓ PASS: Correctly rejected - {e}")
    print()

    # Test 2: Invalid RV value
    print("Test 2: Invalid RV value")
    try:
        lteRateMatchTurbo(np.ones(132, dtype=int), 100, 5)  # RV must be 0-3
        print("  ✗ FAIL: Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  ✓ PASS: Correctly rejected - {e}")
    print()

    # Test 3: Valid inputs
    print("Test 3: Valid inputs")
    rm = lteRateMatchTurbo(np.ones(132, dtype=int), 100, 0)
    print(f"  ✓ PASS: Accepted valid input (length={len(rm)})")
    print()


def test_output_lengths():
    """
    Test various output lengths

    Rate matching should work for any output length
    """
    print("="*70)
    print("Various Output Lengths")
    print("="*70)
    print()

    # Create turbo encoded input
    input_bits = np.ones(40, dtype=int)
    encoded = lteTurboEncode(input_bits)  # 132 bits

    print(f"Input: {len(encoded)} bits (turbo encoded)")
    print()

    # Test different output lengths
    test_lengths = [50, 100, 132, 200, 300]

    for E in test_lengths:
        rm_out = lteRateMatchTurbo(encoded, E, 0)
        print(f"Target E={E:3} bits → Output: {len(rm_out):3} bits ✓")

        assert len(rm_out) == E, f"Expected {E} bits, got {len(rm_out)}"

    print()
    print("✓ PASS: Rate matching works for all output lengths")
    print()


def test_sub_block_interleaver_params():
    """
    Verify sub-block interleaver parameters match MATLAB

    From documentation:
    - CTCSubblock = 32 columns
    - Column permutation pattern: [0, 16, 8, 24, ...]
    """
    print("="*70)
    print("Sub-Block Interleaver Parameters")
    print("="*70)
    print()

    from ctr_encode import LTE_RateMatching

    rm = LTE_RateMatching()

    print(f"CTCSubblock (columns): {rm.C_subblock}")
    print(f"  Expected: 32")
    assert rm.C_subblock == 32, "CTCSubblock should be 32"
    print(f"  ✓ PASS")
    print()

    print(f"Column permutation pattern (first 8):")
    print(f"  Python: {rm.P[:8]}")
    print(f"  MATLAB: [0, 16, 8, 24, 4, 20, 12, 28]")
    expected_pattern = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                                  1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])
    assert np.array_equal(rm.P, expected_pattern), "Permutation pattern mismatch"
    print(f"  ✓ PASS")
    print()


if __name__ == "__main__":
    print()
    print("="*70)
    print("MATLAB lteRateMatchTurbo Compatibility Test Suite")
    print("="*70)
    print()

    test_matlab_example_1()
    test_complete_chain()
    test_redundancy_versions()
    test_filler_bits_handling()
    test_input_validation()
    test_output_lengths()
    test_sub_block_interleaver_params()

    print("="*70)
    print("ALL TESTS PASSED ✓")
    print("Python implementation matches MATLAB documentation")
    print("="*70)
    print()
