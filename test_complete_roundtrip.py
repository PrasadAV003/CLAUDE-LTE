"""
Complete LTE Channel Coding Round-Trip Test
Tests encode-decode pipeline for 40, 1088, and 6145 bit sequences
Saves ALL intermediate results to a single comprehensive file
"""

import numpy as np
from ctr_encode import (
    LTE_CRC, LTE_CodeBlockSegmentation,
    lteTurboEncode, lteRateMatchTurbo
)
from ctr_decode import (
    lteCRCDecode, lteCodeBlockDesegment,
    lteTurboDecode, lteRateRecoverTurbo
)
from datetime import datetime


def bits_to_string(bits, max_display=100):
    """Convert bits array to string for display"""
    bits_str = ''.join(str(int(b)) for b in bits[:max_display])
    if len(bits) > max_display:
        bits_str += f'... ({len(bits)} total bits)'
    return bits_str


def save_array_to_file(f, name, data, show_all=False):
    """Save array data to file with formatting"""
    f.write(f"\n{name}:\n")
    f.write(f"  Length: {len(data)}\n")
    f.write(f"  Data type: {data.dtype}\n")

    if show_all or len(data) <= 200:
        # Show all data for short sequences
        f.write(f"  Data: {data.tolist()}\n")
    else:
        # Show first and last 100 for long sequences
        f.write(f"  First 100: {data[:100].tolist()}\n")
        f.write(f"  Last 100: {data[-100:].tolist()}\n")
        f.write(f"  Binary (first 100): {bits_to_string(data, 100)}\n")


def create_test_sequence(base_sequence, target_length):
    """Create test sequence by repeating base sequence"""
    base_len = len(base_sequence)

    if target_length <= base_len:
        return base_sequence[:target_length]
    else:
        repeats = (target_length + base_len - 1) // base_len
        extended = np.tile(base_sequence, repeats)
        return extended[:target_length]


def run_complete_roundtrip_test(test_bits, test_name, output_file):
    """
    Complete round-trip test with all intermediate stages logged

    ENCODE CHAIN:
    1. Original data
    2. CRC-24A attachment (transport block CRC)
    3. Code block segmentation
    4. Turbo encoding
    5. Rate matching

    CHANNEL:
    Perfect channel (no noise)

    DECODE CHAIN:
    6. Rate recovery (de-rate-match)
    7. Turbo decoding
    8. Code block desegmentation
    9. CRC-24A decoding
    10. Verification
    """

    f = output_file

    f.write("\n" + "="*80 + "\n")
    f.write(f"TEST: {test_name}\n")
    f.write("="*80 + "\n")
    f.write(f"Test date: {datetime.now()}\n")
    f.write(f"Input length: {len(test_bits)} bits\n")
    f.write("\n")

    # ========================================================================
    # ENCODE CHAIN
    # ========================================================================

    f.write("-"*80 + "\n")
    f.write("ENCODING CHAIN\n")
    f.write("-"*80 + "\n\n")

    # Stage 1: Original input
    f.write("STAGE 1: ORIGINAL INPUT\n")
    f.write("-"*40 + "\n")
    save_array_to_file(f, "Original bits", test_bits, show_all=True)
    f.write(f"  Ones: {np.sum(test_bits == 1)}\n")
    f.write(f"  Zeros: {np.sum(test_bits == 0)}\n")

    # Stage 2: CRC-24A attachment
    f.write("\n\nSTAGE 2: CRC-24A ATTACHMENT (Transport Block CRC)\n")
    f.write("-"*40 + "\n")
    crc_processor = LTE_CRC()
    data_with_crc = crc_processor.crc_attach(test_bits, crc_type='24A', mask=0)

    f.write(f"Input length: {len(test_bits)} bits\n")
    f.write(f"CRC length: 24 bits\n")
    f.write(f"Output length: {len(data_with_crc)} bits\n")
    save_array_to_file(f, "Data with CRC", data_with_crc)
    save_array_to_file(f, "CRC bits only", data_with_crc[-24:], show_all=True)

    # Stage 3: Code block segmentation
    f.write("\n\nSTAGE 3: CODE BLOCK SEGMENTATION\n")
    f.write("-"*40 + "\n")
    segmenter = LTE_CodeBlockSegmentation()
    code_blocks, seg_info = segmenter.segment(data_with_crc)

    f.write(f"Number of blocks (C): {seg_info['C']}\n")
    f.write(f"K_plus: {seg_info['K_plus']}\n")
    f.write(f"K_minus: {seg_info['K_minus']}\n")
    f.write(f"C_plus: {seg_info['C_plus']}\n")
    f.write(f"C_minus: {seg_info['C_minus']}\n")
    f.write(f"Filler bits (F): {seg_info['F']}\n")
    f.write(f"CRC per block (L): {seg_info['L']}\n")
    f.write(f"Block sizes: {seg_info['code_block_sizes']}\n")
    f.write(f"Total input bits: {seg_info['B']}\n")
    f.write(f"B_prime: {seg_info['B_prime']}\n")

    for idx, cb in enumerate(code_blocks):
        f.write(f"\n  Code Block {idx}:\n")
        save_array_to_file(f, f"    Block {idx} data", cb)
        f.write(f"    Filler bits (-1): {np.sum(cb == -1)}\n")

    # Stage 4: Turbo encoding
    f.write("\n\nSTAGE 4: TURBO ENCODING\n")
    f.write("-"*40 + "\n")
    turbo_encoded = lteTurboEncode(code_blocks)

    if isinstance(turbo_encoded, list):
        total_turbo_bits = sum(len(cb) for cb in turbo_encoded)
        turbo_lengths = [len(cb) for cb in turbo_encoded]

        f.write(f"Number of blocks: {len(turbo_encoded)}\n")
        f.write(f"Block lengths: {turbo_lengths}\n")
        f.write(f"Total output bits: {total_turbo_bits}\n")
        f.write(f"Rate: 1/3 (each K bits → 3*(K+4) bits)\n")

        for idx, te in enumerate(turbo_encoded):
            f.write(f"\n  Turbo Encoded Block {idx}:\n")
            save_array_to_file(f, f"    Block {idx} encoded", te)
            f.write(f"    NULL bits (-1): {np.sum(te == -1)}\n")
    else:
        total_turbo_bits = len(turbo_encoded)
        f.write(f"Output length: {len(turbo_encoded)} bits\n")
        save_array_to_file(f, "Turbo encoded data", turbo_encoded)
        f.write(f"NULL bits (-1): {np.sum(turbo_encoded == -1)}\n")

    # Stage 5: Rate matching
    f.write("\n\nSTAGE 5: RATE MATCHING\n")
    f.write("-"*40 + "\n")

    # Calculate rate matched output length
    if isinstance(turbo_encoded, list):
        E_total = int(total_turbo_bits * 1.2)
    else:
        E_total = int(len(turbo_encoded) * 1.2)

    rate_matched = lteRateMatchTurbo(turbo_encoded, E_total, rv=0)

    f.write(f"Redundancy version (rv): 0\n")
    f.write(f"Target output length (E): {E_total}\n")
    f.write(f"Actual output length: {len(rate_matched)}\n")
    save_array_to_file(f, "Rate matched data", rate_matched)
    f.write(f"NULL bits (-1): {np.sum(rate_matched == -1) if isinstance(rate_matched, np.ndarray) else 0}\n")

    # ========================================================================
    # CHANNEL SIMULATION
    # ========================================================================

    f.write("\n\n" + "-"*80 + "\n")
    f.write("CHANNEL TRANSMISSION\n")
    f.write("-"*80 + "\n\n")

    # Convert to soft values (LLRs)
    # LLR convention: positive = bit 0, negative = bit 1
    received_soft = np.where(rate_matched == 0, 5.0, -5.0).astype(float)

    f.write("Channel type: Perfect (no noise)\n")
    f.write("LLR mapping: bit 0 → +5.0, bit 1 → -5.0\n")
    f.write(f"Transmitted bits: {len(rate_matched)}\n")
    f.write(f"Received soft values: {len(received_soft)}\n")
    save_array_to_file(f, "Transmitted hard bits", rate_matched)
    save_array_to_file(f, "Received soft LLRs", received_soft)

    # ========================================================================
    # DECODE CHAIN
    # ========================================================================

    f.write("\n\n" + "-"*80 + "\n")
    f.write("DECODING CHAIN\n")
    f.write("-"*80 + "\n\n")

    # Stage 6: Rate recovery
    f.write("STAGE 6: RATE RECOVERY (De-Rate-Match)\n")
    f.write("-"*40 + "\n")
    rate_recovered = lteRateRecoverTurbo(received_soft, len(test_bits), rv=0, cbsbuffers=None)

    if isinstance(rate_recovered, list):
        total_recovered = sum(len(cb) for cb in rate_recovered)
        recovered_lengths = [len(cb) for cb in rate_recovered]

        f.write(f"Number of blocks: {len(rate_recovered)}\n")
        f.write(f"Block lengths: {recovered_lengths}\n")
        f.write(f"Total soft bits: {total_recovered}\n")

        for idx, rr in enumerate(rate_recovered):
            f.write(f"\n  Rate Recovered Block {idx}:\n")
            f.write(f"    Length: {len(rr)}\n")
            if len(rr) <= 200:
                f.write(f"    Data: {rr.tolist()}\n")
            else:
                f.write(f"    First 50: {rr[:50].tolist()}\n")
                f.write(f"    Last 50: {rr[-50:].tolist()}\n")
    else:
        f.write(f"Output length: {len(rate_recovered)} soft bits\n")
        if len(rate_recovered) <= 200:
            f.write(f"Data: {rate_recovered.tolist()}\n")
        else:
            f.write(f"First 50: {rate_recovered[:50].tolist()}\n")
            f.write(f"Last 50: {rate_recovered[-50:].tolist()}\n")

    # Stage 7: Turbo decoding
    f.write("\n\nSTAGE 7: TURBO DECODING (Max-Log-MAP)\n")
    f.write("-"*40 + "\n")
    turbo_decoded = lteTurboDecode(rate_recovered, nturbodecits=5)

    if isinstance(turbo_decoded, list):
        total_decoded_bits = sum(len(cb) for cb in turbo_decoded)
        decoded_lengths = [len(cb) for cb in turbo_decoded]

        f.write(f"Number of blocks: {len(turbo_decoded)}\n")
        f.write(f"Block lengths: {decoded_lengths}\n")
        f.write(f"Total decoded bits: {total_decoded_bits}\n")
        f.write(f"Iterations: 5\n")
        f.write(f"Algorithm: Max-Log-MAP\n")

        for idx, td in enumerate(turbo_decoded):
            f.write(f"\n  Turbo Decoded Block {idx}:\n")
            save_array_to_file(f, f"    Block {idx} decoded", td)
    else:
        f.write(f"Output length: {len(turbo_decoded)} bits\n")
        f.write(f"Iterations: 5\n")
        f.write(f"Algorithm: Max-Log-MAP\n")
        save_array_to_file(f, "Turbo decoded data", turbo_decoded)

    # Stage 8: Code block desegmentation
    f.write("\n\nSTAGE 8: CODE BLOCK DESEGMENTATION\n")
    f.write("-"*40 + "\n")
    desegmented, crc_errors = lteCodeBlockDesegment(turbo_decoded, blklen=len(test_bits))

    f.write(f"Input blocks: {seg_info['C']}\n")
    f.write(f"Expected output length: {len(test_bits) + 24} bits (data + transport CRC)\n")
    f.write(f"Actual output length: {len(desegmented)} bits\n")

    if len(crc_errors) > 0:
        f.write(f"CRC errors per block: {crc_errors.tolist()}\n")
        if np.all(crc_errors == 0):
            f.write("✓ All block CRCs passed\n")
        else:
            f.write("✗ Some block CRCs failed\n")
    else:
        f.write("CRC errors: N/A (single block, no per-block CRC)\n")

    save_array_to_file(f, "Desegmented data (with transport CRC)", desegmented)

    # Stage 9: CRC-24A decoding
    f.write("\n\nSTAGE 9: CRC-24A DECODING (Transport Block CRC)\n")
    f.write("-"*40 + "\n")

    decoded_data, crc_error = lteCRCDecode(desegmented, '24A', mask=0)

    f.write(f"Input length: {len(desegmented)} bits\n")
    f.write(f"CRC length: 24 bits\n")
    f.write(f"Output length: {len(decoded_data)} bits\n")
    f.write(f"CRC error value: {int(crc_error)}\n")
    f.write(f"CRC status: {'PASS' if crc_error == 0 else 'FAIL'}\n")
    save_array_to_file(f, "Final decoded data", decoded_data, show_all=True)

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    f.write("\n\n" + "="*80 + "\n")
    f.write("VERIFICATION RESULTS\n")
    f.write("="*80 + "\n\n")

    if len(decoded_data) == len(test_bits):
        errors = np.sum(decoded_data != test_bits)
        ber = errors / len(test_bits)

        f.write(f"Original length: {len(test_bits)} bits\n")
        f.write(f"Decoded length: {len(decoded_data)} bits\n")
        f.write(f"Bit errors: {errors}\n")
        f.write(f"Bit Error Rate (BER): {ber:.6f}\n")

        if errors == 0:
            f.write("\n✓ PERFECT MATCH - NO ERRORS!\n")
            result = "PASS"
        else:
            f.write(f"\n✗ ERRORS DETECTED - {errors} bit errors\n")
            error_positions = np.where(decoded_data != test_bits)[0]
            f.write(f"Error positions (first 20): {error_positions[:20].tolist()}\n")

            # Show comparison for first 10 errors
            f.write("\nError details (first 10):\n")
            for i, pos in enumerate(error_positions[:10]):
                f.write(f"  Position {pos}: Expected {test_bits[pos]}, Got {decoded_data[pos]}\n")
            result = "FAIL"
    else:
        f.write(f"✗ LENGTH MISMATCH!\n")
        f.write(f"Original length: {len(test_bits)} bits\n")
        f.write(f"Decoded length: {len(decoded_data)} bits\n")
        errors = -1
        result = "LENGTH_MISMATCH"

    f.write("\n" + "="*80 + "\n\n")

    return decoded_data, errors, result


def main():
    """Main test function"""

    # Base sequence (64 bits)
    binary_str = '0000000100110010010001010111011011001101111111101000100110111010'
    base_seq = np.array([int(b) for b in binary_str], dtype=np.int8)

    # Test sizes: 40, 1088, 6145 (not 6144 to trigger multi-block segmentation)
    test_sizes = [40, 1088, 6145]

    # Open single output file for all results
    output_filename = "complete_roundtrip_results.txt"

    with open(output_filename, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("LTE CHANNEL CODING - COMPLETE ROUND-TRIP TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test date: {datetime.now()}\n")
        f.write(f"Test cases: {len(test_sizes)}\n")
        f.write(f"Test sizes: {test_sizes}\n")
        f.write("\n")
        f.write("Base sequence (64 bits):\n")
        f.write(f"  Binary: {binary_str}\n")
        f.write(f"  Decimal: {base_seq.tolist()}\n")
        f.write("\n")
        f.write("Complete pipeline tested:\n")
        f.write("  ENCODE: CRC → Segment → Turbo → Rate Match\n")
        f.write("  CHANNEL: Perfect (no noise)\n")
        f.write("  DECODE: Rate Recover → Turbo → Desegment → CRC\n")
        f.write("\n")
        f.write("="*80 + "\n")

        # Run tests
        results_summary = []

        for size in test_sizes:
            print(f"\nRunning test for {size} bits...")

            # Create test sequence
            test_seq = create_test_sequence(base_seq, size)

            # Run round-trip test
            test_name = f"{size}_bits"
            decoded, errors, status = run_complete_roundtrip_test(test_seq, test_name, f)

            results_summary.append({
                'size': size,
                'errors': errors,
                'status': status
            })

            print(f"  Completed: {status} ({errors} errors)" if errors >= 0 else f"  Completed: {status}")

        # Write summary at the end
        f.write("\n\n" + "="*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Size':<15} {'Errors':<15} {'Status':<15}\n")
        f.write("-"*45 + "\n")

        for result in results_summary:
            error_str = str(result['errors']) if result['errors'] >= 0 else "N/A"
            f.write(f"{result['size']:<15} {error_str:<15} {result['status']:<15}\n")

        f.write("\n")

        # Overall result
        all_passed = all(r['errors'] == 0 for r in results_summary if r['errors'] >= 0)
        if all_passed:
            f.write("✓ ALL TESTS PASSED - PERFECT RECONSTRUCTION!\n")
        else:
            f.write("✗ SOME TESTS FAILED - See details above\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\n\nAll results saved to: {output_filename}")
    print("\nSummary:")
    print(f"{'Size':<15} {'Errors':<15} {'Status':<15}")
    print("-"*45)
    for result in results_summary:
        error_str = str(result['errors']) if result['errors'] >= 0 else "N/A"
        print(f"{result['size']:<15} {error_str:<15} {result['status']:<15}")


if __name__ == "__main__":
    main()
