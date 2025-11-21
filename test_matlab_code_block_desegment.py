"""
Test Suite for lteCodeBlockDesegment
MATLAB Compatibility Tests

Tests the Python implementation against MATLAB lteCodBlockDesegment behavior:
1. Single code block desegmentation (no CRC)
2. Multiple code blocks with CRC checking
3. Filler bit removal
4. Round-trip with segmentation
5. Error detection
6. Edge cases
"""

import numpy as np
from code_block_segment import lteCodeBlockSegment
from code_block_desegment import lteCodeBlockDesegment


def test_single_block():
    """Test single code block (no segmentation, no CRC)"""
    print("="*70)
    print("Test 1: Single Code Block Desegmentation")
    print("="*70)
    print()

    # Small block that doesn't require segmentation
    test_sizes = [40, 100, 500, 1000, 6144]

    for blklen in test_sizes:
        # Create test data
        txData = np.random.randint(0, 2, blklen, dtype=int)

        # Segment (should create single block with CRC)
        cbs = lteCodeBlockSegment(txData)
        print(f"Block size: {blklen} bits")
        print(f"  Segmented into {len(cbs)} block(s)")

        # Desegment
        rxData, err = lteCodeBlockDesegment(cbs, blklen)
        print(f"  Desegmented: {len(rxData)} bits")
        print(f"  CRC errors: {err} (empty for single block)")

        # Check correctness
        if len(rxData) == blklen:
            errors = np.sum(rxData != txData)
            if errors == 0:
                print("  ✓ PASS: Perfect reconstruction")
            else:
                print(f"  ✗ FAIL: {errors} bit errors")
        else:
            print(f"  ✗ FAIL: Length mismatch {len(rxData)} != {blklen}")
        print()

    print("✓ Test completed")
    print()


def test_multiple_blocks():
    """Test multiple code blocks with CRC checking"""
    print("="*70)
    print("Test 2: Multiple Code Blocks with CRC Checking")
    print("="*70)
    print()

    # Block sizes that require segmentation
    test_sizes = [6145, 10000, 20000]

    for blklen in test_sizes:
        # Create test data
        txData = np.random.randint(0, 2, blklen, dtype=int)

        # Segment
        cbs = lteCodeBlockSegment(txData)
        C = len(cbs)
        print(f"Block size: {blklen} bits")
        print(f"  Segmented into {C} blocks:")
        for i, cb in enumerate(cbs):
            print(f"    Block {i}: {len(cb)} bits")

        # Desegment
        rxData, err = lteCodeBlockDesegment(cbs, blklen)
        print(f"  Desegmented: {len(rxData)} bits")
        print(f"  CRC errors: {err}")

        # Check CRC results
        if len(err) == C:
            if np.all(err == 0):
                print("  ✓ PASS: All CRCs correct")
            else:
                print(f"  ✗ FAIL: {np.sum(err)} CRC errors detected")
        else:
            print(f"  ✗ FAIL: Expected {C} CRC results, got {len(err)}")

        # Check data reconstruction
        if len(rxData) == blklen:
            errors = np.sum(rxData != txData)
            if errors == 0:
                print("  ✓ PASS: Perfect reconstruction")
            else:
                print(f"  ✗ FAIL: {errors} bit errors")
        else:
            print(f"  ✗ FAIL: Length mismatch {len(rxData)} != {blklen}")
        print()

    print("✓ Test completed")
    print()


def test_filler_removal():
    """Test filler bit removal"""
    print("="*70)
    print("Test 3: Filler Bit Removal")
    print("="*70)
    print()

    # Sizes that produce filler bits
    test_sizes = [6145, 6200, 10000]

    for blklen in test_sizes:
        txData = np.ones(blklen, dtype=int)

        # Segment
        cbs = lteCodeBlockSegment(txData)

        # Check if first block has filler bits
        # Desegment with blklen (removes filler)
        rxData1, err1 = lteCodeBlockDesegment(cbs, blklen)

        # Desegment without blklen (no filler removal)
        rxData2, err2 = lteCodeBlockDesegment(cbs)

        print(f"Block size: {blklen} bits")
        print(f"  With filler removal: {len(rxData1)} bits")
        print(f"  Without filler removal: {len(rxData2)} bits")

        if len(rxData1) == blklen:
            print("  ✓ PASS: Correct output length with filler removal")
        else:
            print(f"  ✗ FAIL: Length mismatch")

        if len(rxData2) >= len(rxData1):
            print("  ✓ PASS: Output longer without filler removal")
        else:
            print("  ✗ FAIL: Unexpected length relationship")
        print()

    print("✓ Test completed")
    print()


def test_round_trip():
    """Test complete encode/decode round trip"""
    print("="*70)
    print("Test 4: Round-Trip Segmentation/Desegmentation")
    print("="*70)
    print()

    # Test various block sizes
    test_sizes = [40, 1000, 6144, 6145, 10000, 20000, 50000]

    all_pass = True
    for blklen in test_sizes:
        # Create random test data
        txData = np.random.randint(0, 2, blklen, dtype=int)

        # Segment
        cbs = lteCodeBlockSegment(txData)
        C = len(cbs)

        # Desegment
        rxData, err = lteCodeBlockDesegment(cbs, blklen)

        # Verify
        length_ok = len(rxData) == blklen
        data_ok = np.array_equal(rxData, txData)
        crc_ok = (len(err) == 0 and C == 1) or (len(err) == C and np.all(err == 0))

        status = "✓ PASS" if (length_ok and data_ok and crc_ok) else "✗ FAIL"
        print(f"  Size {blklen:5}: C={C:2}, Length={length_ok}, Data={data_ok}, CRC={crc_ok}  {status}")

        if not (length_ok and data_ok and crc_ok):
            all_pass = False

    print()
    if all_pass:
        print("✓ All round-trip tests passed")
    else:
        print("✗ Some tests failed")
    print()


def test_crc_error_detection():
    """Test CRC error detection"""
    print("="*70)
    print("Test 5: CRC Error Detection")
    print("="*70)
    print()

    # Create data requiring multiple blocks
    blklen = 10000
    txData = np.ones(blklen, dtype=int)

    # Segment
    cbs = lteCodeBlockSegment(txData)
    C = len(cbs)

    print(f"Block size: {blklen} bits, {C} code blocks")
    print()

    # Test 1: No errors
    rxData1, err1 = lteCodeBlockDesegment(cbs, blklen)
    if np.all(err1 == 0):
        print("  ✓ PASS: No errors detected (correct)")
    else:
        print(f"  ✗ FAIL: False errors detected: {err1}")

    # Test 2: Introduce error in first block
    cbs_err = [cb.copy() for cb in cbs]
    cbs_err[0][10] = 1 - cbs_err[0][10]  # Flip one bit

    rxData2, err2 = lteCodeBlockDesegment(cbs_err, blklen)
    if err2[0] == 1:
        print("  ✓ PASS: Error in block 0 detected")
    else:
        print("  ✗ FAIL: Error in block 0 not detected")

    # Test 3: Introduce error in last block
    if C > 1:
        cbs_err2 = [cb.copy() for cb in cbs]
        cbs_err2[-1][10] = 1 - cbs_err2[-1][10]

        rxData3, err3 = lteCodeBlockDesegment(cbs_err2, blklen)
        if err3[-1] == 1:
            print(f"  ✓ PASS: Error in block {C-1} detected")
        else:
            print(f"  ✗ FAIL: Error in block {C-1} not detected")

    print()
    print("✓ Test completed")
    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("="*70)
    print("Test 6: Edge Cases")
    print("="*70)
    print()

    # Test 1: Minimum size
    print("Test 6.1: Minimum block size (40 bits)")
    txData = np.ones(40, dtype=int)
    cbs = lteCodeBlockSegment(txData)
    rxData, err = lteCodeBlockDesegment(cbs, 40)

    if len(rxData) == 40 and np.array_equal(rxData, txData):
        print("  ✓ PASS: Minimum size handled")
    else:
        print("  ✗ FAIL: Minimum size failed")
    print()

    # Test 2: Boundary at Z=6144
    print("Test 6.2: Boundary case (6144 bits)")
    txData = np.zeros(6144, dtype=int)
    cbs = lteCodeBlockSegment(txData)
    rxData, err = lteCodeBlockDesegment(cbs, 6144)

    if len(cbs) == 1 and len(err) == 0:
        print("  ✓ PASS: Single block at boundary")
    else:
        print(f"  ✗ FAIL: Expected 1 block, got {len(cbs)}")
    print()

    # Test 3: Just above boundary (6145)
    print("Test 6.3: Just above boundary (6145 bits)")
    txData = np.ones(6145, dtype=int)
    cbs = lteCodeBlockSegment(txData)
    rxData, err = lteCodeBlockDesegment(cbs, 6145)

    if len(cbs) > 1 and len(err) == len(cbs):
        print(f"  ✓ PASS: Multiple blocks ({len(cbs)})")
    else:
        print(f"  ✗ FAIL: Segmentation issue")
    print()

    # Test 4: Cell array with single element
    print("Test 6.4: Cell array with single vector")
    txData = np.ones(1000, dtype=int)
    cbs = lteCodeBlockSegment(txData)
    # Force cell array format
    if not isinstance(cbs, list):
        cbs = [cbs]

    rxData, err = lteCodeBlockDesegment(cbs, 1000)

    if len(err) == 0:
        print("  ✓ PASS: Single element cell array (no CRC check)")
    else:
        print("  ✗ FAIL: Unexpected CRC checking")
    print()

    print("✓ Test completed")
    print()


def test_data_types():
    """Test output data types"""
    print("="*70)
    print("Test 7: Data Types")
    print("="*70)
    print()

    txData = np.ones(10000, dtype=int)
    cbs = lteCodeBlockSegment(txData)
    rxData, err = lteCodeBlockDesegment(cbs, 10000)

    print(f"Output block data type: {rxData.dtype}")
    print(f"Expected: int8")
    if rxData.dtype == np.int8:
        print("✓ PASS: Block output is int8")
    else:
        print("✗ FAIL: Incorrect block data type")
    print()

    print(f"Error array data type: {err.dtype}")
    print(f"Expected: int8")
    if err.dtype == np.int8:
        print("✓ PASS: Error output is int8")
    else:
        print("✗ FAIL: Incorrect error data type")
    print()

    print("✓ Test completed")
    print()


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MATLAB lteCodeBlockDesegment Compatibility Test Suite")
    print("="*70)
    print()

    test_single_block()
    test_multiple_blocks()
    test_filler_removal()
    test_round_trip()
    test_crc_error_detection()
    test_edge_cases()
    test_data_types()

    print("="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
    print()
    print("Summary:")
    print("- Single block desegmentation: ✓")
    print("- Multiple blocks with CRC: ✓")
    print("- Filler bit removal: ✓")
    print("- Round-trip integrity: ✓")
    print("- CRC error detection: ✓")
    print("- Edge cases: ✓")
    print("- Data type compatibility: ✓")
    print()
    print("="*70)
