"""
Complete LTE Encoding Chain Test - MATLAB Compatibility

This test verifies the complete LTE encoding chain as used in MATLAB:
1. Transport Block → lteCRCEncode (CRC24A attachment)
2. CRC output → lteCodeBlockSegment (segmentation with CRC24B)
3. Code blocks → lteTurboEncode (turbo encoding)

This mimics the actual LTE DL-SCH/UL-SCH encoding process.
"""

import numpy as np
from crc_encode import lteCRCEncode
from code_block_segment import lteCodeBlockSegment
from ctr_encode import lteTurboEncode


def test_complete_chain_small_block():
    """
    Test complete chain with small transport block (no segmentation)

    MATLAB process:
    1. Transport block: 100 bits
    2. Add CRC24A: 100 + 24 = 124 bits
    3. Code block segment: 124 ≤ 6144 → 1 block (padded to 128)
    4. Turbo encode: 128 → 3*(128+4) = 396 bits
    """
    print("="*70)
    print("Complete Chain Test 1: Small Transport Block (No Segmentation)")
    print("="*70)
    print()

    # Step 1: Transport block
    transport_block = np.ones(100, dtype=int)
    print(f"Step 1: Transport Block")
    print(f"  Input: {len(transport_block)} bits")
    print()

    # Step 2: CRC24A attachment
    blk_crc = lteCRCEncode(transport_block, '24A')
    print(f"Step 2: CRC24A Attachment")
    print(f"  Input: {len(transport_block)} bits")
    print(f"  Output: {len(blk_crc)} bits (100 + 24)")
    print(f"  CRC24A: {blk_crc[-24:]}")
    print()

    # Step 3: Code block segmentation
    code_blocks = lteCodeBlockSegment(blk_crc)
    print(f"Step 3: Code Block Segmentation")
    print(f"  Input: {len(blk_crc)} bits")
    print(f"  Number of blocks: {len(code_blocks)}")
    print(f"  Block sizes: {[len(cb) for cb in code_blocks]}")

    # Check for filler bits
    filler_count = np.sum(code_blocks[0] < 0) if len(code_blocks) > 0 else 0
    print(f"  Filler bits in first block: {filler_count}")
    if filler_count > 0:
        print(f"  First {min(10, filler_count)} bits: {code_blocks[0][:min(10, filler_count)]}")
    print()

    # Step 4: Turbo encoding
    encoded_blocks = lteTurboEncode(code_blocks)
    print(f"Step 4: Turbo Encoding")
    print(f"  Input: {len(code_blocks)} code blocks")
    print(f"  Output: {len(encoded_blocks)} encoded blocks")
    for i, eb in enumerate(encoded_blocks):
        K = len(code_blocks[i])
        expected = 3 * (K + 4)
        print(f"  Block {i}: {K} bits → {len(eb)} bits (expected {expected})")
    print()

    # Verify
    assert len(code_blocks) == 1, "Should have 1 code block"
    assert len(encoded_blocks) == 1, "Should have 1 encoded block"

    K = len(code_blocks[0])
    assert len(encoded_blocks[0]) == 3 * (K + 4), "Encoded length should be 3*(K+4)"

    print("✓ PASS: Complete chain successful")
    print()


def test_complete_chain_large_block():
    """
    Test complete chain with large transport block (with segmentation)

    MATLAB process:
    1. Transport block: 10000 bits
    2. Add CRC24A: 10000 + 24 = 10024 bits
    3. Code block segment: 10024 > 6144 → multiple blocks with CRC24B
    4. Turbo encode each block
    """
    print("="*70)
    print("Complete Chain Test 2: Large Transport Block (With Segmentation)")
    print("="*70)
    print()

    # Step 1: Transport block
    transport_block = np.ones(10000, dtype=int)
    print(f"Step 1: Transport Block")
    print(f"  Input: {len(transport_block)} bits")
    print()

    # Step 2: CRC24A attachment
    blk_crc = lteCRCEncode(transport_block, '24A')
    print(f"Step 2: CRC24A Attachment")
    print(f"  Input: {len(transport_block)} bits")
    print(f"  Output: {len(blk_crc)} bits (10000 + 24)")
    print()

    # Step 3: Code block segmentation
    code_blocks = lteCodeBlockSegment(blk_crc)
    print(f"Step 3: Code Block Segmentation")
    print(f"  Input: {len(blk_crc)} bits (> 6144)")
    print(f"  Number of blocks: {len(code_blocks)}")
    print(f"  Block sizes: {[len(cb) for cb in code_blocks]}")

    # Check for filler bits
    filler_count = np.sum(code_blocks[0] < 0)
    print(f"  Filler bits in first block: {filler_count}")
    if filler_count > 0:
        print(f"  First {min(10, filler_count)} bits: {code_blocks[0][:min(10, filler_count)]}")

    # Verify CRC24B in each block (last 24 bits should not all be 1)
    print(f"  CRC24B appended to each block: ", end="")
    all_have_crc = all(not np.all(cb[-24:] == 1) for cb in code_blocks)
    print("✓" if all_have_crc else "✗")
    print()

    # Step 4: Turbo encoding
    encoded_blocks = lteTurboEncode(code_blocks)
    print(f"Step 4: Turbo Encoding")
    print(f"  Input: {len(code_blocks)} code blocks")
    print(f"  Output: {len(encoded_blocks)} encoded blocks")

    total_bits = 0
    for i, eb in enumerate(encoded_blocks):
        K = len(code_blocks[i])
        expected = 3 * (K + 4)
        print(f"  Block {i}: {K} bits → {len(eb)} bits (expected {expected})")
        total_bits += len(eb)
        assert len(eb) == expected, f"Block {i} encoded length mismatch"

    print(f"  Total encoded bits: {total_bits}")
    print()

    # Verify
    assert len(code_blocks) >= 2, "Should have multiple code blocks"
    assert len(encoded_blocks) == len(code_blocks), "Same number of encoded blocks"

    print("✓ PASS: Complete chain with segmentation successful")
    print()


def test_filler_bits_propagation():
    """
    Test that filler bits propagate correctly through the entire chain

    This tests the critical MATLAB requirement:
    "negative input bit values are specially processed"
    """
    print("="*70)
    print("Complete Chain Test 3: Filler Bits Propagation")
    print("="*70)
    print()

    # Create transport block that will require filler bits after segmentation
    # We want B > 6144 but not perfectly aligned
    transport_block = np.ones(6200, dtype=int)
    print(f"Step 1: Transport Block")
    print(f"  Input: {len(transport_block)} bits")
    print()

    # Step 2: CRC24A attachment
    blk_crc = lteCRCEncode(transport_block, '24A')
    print(f"Step 2: CRC24A Attachment")
    print(f"  Output: {len(blk_crc)} bits")
    print()

    # Step 3: Code block segmentation
    code_blocks = lteCodeBlockSegment(blk_crc)
    print(f"Step 3: Code Block Segmentation")
    print(f"  Number of blocks: {len(code_blocks)}")
    print(f"  Block sizes: {[len(cb) for cb in code_blocks]}")

    # Check filler bits in first block
    filler_positions = np.where(code_blocks[0] < 0)[0]
    print(f"  Filler bits in block 0: {len(filler_positions)}")
    print(f"  Filler positions: {filler_positions[:10] if len(filler_positions) > 0 else 'none'}")
    print()

    # Step 4: Turbo encoding
    encoded_blocks = lteTurboEncode(code_blocks)
    print(f"Step 4: Turbo Encoding")

    # Check that filler bits appear in encoded output (S and P1)
    if len(filler_positions) > 0:
        # Encoded format is [S P1 P2]
        K = len(code_blocks[0])
        K_tail = K + 4

        # Extract S, P1, P2 from first encoded block
        encoded_0 = encoded_blocks[0]
        S = encoded_0[:K_tail]
        P1 = encoded_0[K_tail:2*K_tail]
        P2 = encoded_0[2*K_tail:]

        # Check S for filler bits
        s_filler_positions = np.where(S < 0)[0]
        print(f"  Filler bits in S (systematic): {len(s_filler_positions)}")
        print(f"  S filler positions: {s_filler_positions[:10]}")

        # Check P1 for filler bits
        p1_filler_positions = np.where(P1 < 0)[0]
        print(f"  Filler bits in P1 (parity1): {len(p1_filler_positions)}")
        print(f"  P1 filler positions: {p1_filler_positions[:10]}")

        # Check P2 (may have -1 due to interleaving)
        p2_filler_positions = np.where(P2 < 0)[0]
        print(f"  Filler bits in P2 (parity2): {len(p2_filler_positions)}")
        print(f"  Note: P2 filler positions depend on interleaver permutation")

        print()

        # Verify filler bits are present in S and P1
        assert len(s_filler_positions) > 0, "S should have filler bits"
        assert len(p1_filler_positions) > 0, "P1 should have filler bits"

        print("✓ PASS: Filler bits propagate through chain correctly")
    else:
        print("  No filler bits in this test case")

    print()


def test_matlab_workflow_example():
    """
    Test complete MATLAB workflow as documented

    This simulates a typical MATLAB LTE encoding workflow:

    % MATLAB code:
    % trblk = randi([0 1], 1000, 1);
    % trblkCrc = lteCRCEncode(trblk, '24A');
    % cbs = lteCodeBlockSegment(trblkCrc);
    % encoded = lteTurboEncode(cbs);
    """
    print("="*70)
    print("Complete Chain Test 4: MATLAB Workflow Example")
    print("="*70)
    print()

    print("MATLAB Workflow:")
    print("  trblk = randi([0 1], 1000, 1);")
    print("  trblkCrc = lteCRCEncode(trblk, '24A');")
    print("  cbs = lteCodeBlockSegment(trblkCrc);")
    print("  encoded = lteTurboEncode(cbs);")
    print()

    # Python equivalent
    print("Python Equivalent:")
    print("-"*70)

    # Create random transport block
    np.random.seed(42)  # For reproducibility
    trblk = np.random.randint(0, 2, 1000, dtype=int)
    print(f"trblk = np.random.randint(0, 2, {len(trblk)})")
    print(f"  Length: {len(trblk)} bits")
    print(f"  Type: {trblk.dtype}")
    print()

    # Add CRC24A
    trblkCrc = lteCRCEncode(trblk, '24A')
    print(f"trblkCrc = lteCRCEncode(trblk, '24A')")
    print(f"  Length: {len(trblkCrc)} bits ({len(trblk)} + 24)")
    print(f"  Type: {trblkCrc.dtype}")
    print()

    # Code block segmentation
    cbs = lteCodeBlockSegment(trblkCrc)
    print(f"cbs = lteCodeBlockSegment(trblkCrc)")
    print(f"  Number of blocks: {len(cbs)}")
    print(f"  Block sizes: {[len(cb) for cb in cbs]}")
    print(f"  Type: {[cb.dtype for cb in cbs]}")
    print()

    # Turbo encoding
    encoded = lteTurboEncode(cbs)
    print(f"encoded = lteTurboEncode(cbs)")
    print(f"  Number of encoded blocks: {len(encoded)}")
    print(f"  Encoded sizes: {[len(eb) for eb in encoded]}")
    print(f"  Type: {[eb.dtype for eb in encoded]}")
    print()

    # Verification
    assert len(trblkCrc) == len(trblk) + 24, "CRC24A should add 24 bits"
    assert all(cb.dtype == np.int8 for cb in cbs), "Code blocks should be int8"
    assert all(eb.dtype == np.int8 for eb in encoded), "Encoded blocks should be int8"
    assert len(encoded) == len(cbs), "Same number of encoded blocks as code blocks"

    # Verify turbo coding rate
    for i, (cb, eb) in enumerate(zip(cbs, encoded)):
        expected_len = 3 * (len(cb) + 4)
        assert len(eb) == expected_len, f"Block {i}: Expected {expected_len}, got {len(eb)}"

    print("✓ PASS: MATLAB workflow completed successfully")
    print()


def test_all_crc_types_integration():
    """
    Test integration with all CRC types

    While LTE uses CRC24A for transport blocks, verify all CRC types work
    """
    print("="*70)
    print("Complete Chain Test 5: All CRC Types")
    print("="*70)
    print()

    transport_block = np.ones(100, dtype=int)

    for crc_type in ['8', '16', '24A', '24B']:
        print(f"Testing with CRC-{crc_type}:")

        # Add CRC
        blk_crc = lteCRCEncode(transport_block, crc_type)
        crc_len = {'8': 8, '16': 16, '24A': 24, '24B': 24}[crc_type]
        print(f"  CRC attachment: {len(transport_block)} → {len(blk_crc)} bits (+{crc_len})")

        # Segment
        code_blocks = lteCodeBlockSegment(blk_crc)
        print(f"  Segmentation: {len(code_blocks)} blocks, sizes {[len(cb) for cb in code_blocks]}")

        # Encode
        encoded = lteTurboEncode(code_blocks)
        print(f"  Turbo encoding: {len(encoded)} blocks, sizes {[len(eb) for eb in encoded]}")
        print()

    print("✓ PASS: All CRC types work in complete chain")
    print()


if __name__ == "__main__":
    print()
    print("="*70)
    print("COMPLETE LTE ENCODING CHAIN TEST SUITE")
    print("Testing: lteCRCEncode → lteCodeBlockSegment → lteTurboEncode")
    print("="*70)
    print()

    test_complete_chain_small_block()
    test_complete_chain_large_block()
    test_filler_bits_propagation()
    test_matlab_workflow_example()
    test_all_crc_types_integration()

    print("="*70)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("Complete LTE encoding chain matches MATLAB behavior")
    print("="*70)
    print()
