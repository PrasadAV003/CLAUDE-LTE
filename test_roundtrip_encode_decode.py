"""
LTE Channel Coding Round-Trip Test
Tests complete encode-decode pipeline for 40, 1088, and 6144 bit sequences
Saves detailed results at each stage to text files
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
import os

# Create output directory for results
os.makedirs("roundtrip_results", exist_ok=True)


def save_stage_results(filename, stage_name, data_dict):
    """Save stage results to text file"""
    with open(f"roundtrip_results/{filename}", 'a') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{stage_name}\n")
        f.write("=" * 80 + "\n")
        for key, value in data_dict.items():
            f.write(f"{key}:\n")
            if isinstance(value, np.ndarray):
                f.write(f"  Length: {len(value)}\n")
                f.write(f"  Type: {value.dtype}\n")
                if len(value) <= 200:
                    # Show full data for short sequences
                    f.write(f"  Data: {value.tolist()}\n")
                else:
                    # Show first/last for long sequences
                    f.write(f"  First 100: {value[:100].tolist()}\n")
                    f.write(f"  Last 100: {value[-100:].tolist()}\n")
            elif isinstance(value, list):
                f.write(f"  Length: {len(value)}\n")
                f.write(f"  Data: {value}\n")
            elif isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {value}\n")
        f.write("\n")


def create_test_sequence(base_sequence, target_length):
    """Create test sequence by repeating base sequence"""
    base_len = len(base_sequence)

    if target_length <= base_len:
        return base_sequence[:target_length]
    else:
        # Repeat and truncate
        repeats = (target_length + base_len - 1) // base_len
        extended = np.tile(base_sequence, repeats)
        return extended[:target_length]


def test_roundtrip(test_bits, test_name, log_file):
    """
    Complete round-trip test for encode-decode pipeline

    ENCODE CHAIN:
    1. CRC-24A attachment (transport block CRC)
    2. Code block segmentation
    3. Turbo encoding
    4. Rate matching

    DECODE CHAIN:
    1. Rate recovery
    2. Turbo decoding
    3. Code block desegmentation
    4. CRC-24A decoding
    """

    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")

    # Clear log file for this test
    with open(f"roundtrip_results/{log_file}", 'w') as f:
        f.write(f"LTE CHANNEL CODING ROUND-TRIP TEST\n")
        f.write(f"Test: {test_name}\n")
        f.write(f"Date: {np.datetime64('now')}\n")
        f.write("\n")

    # Save original input
    save_stage_results(log_file, "STAGE 0: ORIGINAL INPUT", {
        "Input bits": test_bits,
        "Length": len(test_bits),
        "Ones": np.sum(test_bits == 1),
        "Zeros": np.sum(test_bits == 0)
    })

    print(f"Input: {len(test_bits)} bits")
    print(f"  Ones: {np.sum(test_bits == 1)}, Zeros: {np.sum(test_bits == 0)}")

    # ========================================================================
    # ENCODE CHAIN
    # ========================================================================

    print("\n" + "─"*80)
    print("ENCODING CHAIN")
    print("─"*80)

    # Stage 1: CRC-24A attachment
    print("\n[1] CRC-24A Attachment...")
    crc_processor = LTE_CRC()
    data_with_crc = crc_processor.crc_attach(test_bits, crc_type='24A', mask=0)

    save_stage_results(log_file, "STAGE 1: CRC-24A ATTACHMENT", {
        "Input length": len(test_bits),
        "CRC length": 24,
        "Output length": len(data_with_crc),
        "Output data": data_with_crc,
        "CRC bits": data_with_crc[-24:]
    })

    print(f"  Input: {len(test_bits)} bits")
    print(f"  Output: {len(data_with_crc)} bits (data + 24-bit CRC)")

    # Stage 2: Code block segmentation
    print("\n[2] Code Block Segmentation...")
    segmenter = LTE_CodeBlockSegmentation()
    code_blocks, seg_info = segmenter.segment(data_with_crc)

    save_stage_results(log_file, "STAGE 2: CODE BLOCK SEGMENTATION", {
        "Number of blocks (C)": seg_info['C'],
        "K_plus": seg_info['K_plus'],
        "K_minus": seg_info['K_minus'],
        "C_plus": seg_info['C_plus'],
        "C_minus": seg_info['C_minus'],
        "Filler bits (F)": seg_info['F'],
        "CRC per block (L)": seg_info['L'],
        "Block sizes": seg_info['code_block_sizes'],
        "Total input bits": seg_info['B'],
        "B_prime": seg_info['B_prime']
    })

    for idx, cb in enumerate(code_blocks):
        save_stage_results(log_file, f"  Code Block {idx}", {
            "Length": len(cb),
            "Data": cb,
            "Filler bits (-1)": np.sum(cb == -1)
        })

    print(f"  Blocks: {seg_info['C']}")
    print(f"  Block sizes: {seg_info['code_block_sizes']}")
    print(f"  Filler bits: {seg_info['F']}")

    # Stage 3: Turbo encoding
    print("\n[3] Turbo Encoding...")
    turbo_encoded = lteTurboEncode(code_blocks)

    if isinstance(turbo_encoded, list):
        total_turbo_bits = sum(len(cb) for cb in turbo_encoded)
        turbo_lengths = [len(cb) for cb in turbo_encoded]

        save_stage_results(log_file, "STAGE 3: TURBO ENCODING", {
            "Number of blocks": len(turbo_encoded),
            "Block lengths": turbo_lengths,
            "Total output bits": total_turbo_bits,
            "Rate": "1/3 (each K bits → 3*(K+4) bits)"
        })

        for idx, te in enumerate(turbo_encoded):
            save_stage_results(log_file, f"  Turbo Encoded Block {idx}", {
                "Length": len(te),
                "Data": te,
                "NULL bits (-1)": np.sum(te == -1)
            })

        print(f"  Output: {len(turbo_encoded)} blocks")
        print(f"  Lengths: {turbo_lengths}")
        print(f"  Total: {total_turbo_bits} bits")
    else:
        save_stage_results(log_file, "STAGE 3: TURBO ENCODING", {
            "Output length": len(turbo_encoded),
            "Data": turbo_encoded,
            "NULL bits (-1)": np.sum(turbo_encoded == -1)
        })
        print(f"  Output: {len(turbo_encoded)} bits")

    # Stage 4: Rate matching
    print("\n[4] Rate Matching...")

    # Calculate rate matched output length (E)
    # For this test, use moderate puncturing/repetition
    if isinstance(turbo_encoded, list):
        # Multiple blocks or cell array format
        E_total = int(total_turbo_bits * 1.2)
    else:
        # Single array
        E_total = int(len(turbo_encoded) * 1.2)

    rate_matched = lteRateMatchTurbo(turbo_encoded, E_total, rv=0)

    save_stage_results(log_file, "STAGE 4: RATE MATCHING", {
        "Redundancy version (rv)": 0,
        "Output length (E)": E_total,
        "Actual output": len(rate_matched),
        "Data": rate_matched,
        "NULL bits (-1)": np.sum(rate_matched == -1) if isinstance(rate_matched, np.ndarray) else 0
    })

    print(f"  RV: 0")
    print(f"  Output: {len(rate_matched)} bits")

    # ========================================================================
    # CHANNEL SIMULATION (Perfect - no noise)
    # ========================================================================

    print("\n" + "─"*80)
    print("CHANNEL: Perfect (no noise)")
    print("─"*80)

    # Convert to soft values (LLRs)
    # LLR convention: positive = bit 0, negative = bit 1
    # Use large magnitude for perfect channel
    received_soft = np.where(rate_matched == 0, 5.0, -5.0).astype(float)

    save_stage_results(log_file, "CHANNEL TRANSMISSION", {
        "Channel type": "Perfect (no noise)",
        "LLR mapping": "bit 0 → +5.0, bit 1 → -5.0",
        "Transmitted bits": rate_matched,
        "Received soft values": received_soft
    })

    print(f"  Transmitted: {len(rate_matched)} bits")
    print(f"  Received: {len(received_soft)} soft values (LLRs)")

    # ========================================================================
    # DECODE CHAIN
    # ========================================================================

    print("\n" + "─"*80)
    print("DECODING CHAIN")
    print("─"*80)

    # Stage 5: Rate recovery
    print("\n[5] Rate Recovery...")
    rate_recovered = lteRateRecoverTurbo(received_soft, len(test_bits), rv=0, cbsbuffers=None)

    if isinstance(rate_recovered, list):
        total_recovered = sum(len(cb) for cb in rate_recovered)
        recovered_lengths = [len(cb) for cb in rate_recovered]

        save_stage_results(log_file, "STAGE 5: RATE RECOVERY", {
            "Number of blocks": len(rate_recovered),
            "Block lengths": recovered_lengths,
            "Total soft bits": total_recovered
        })

        for idx, rr in enumerate(rate_recovered):
            save_stage_results(log_file, f"  Rate Recovered Block {idx}", {
                "Length": len(rr),
                "First 50 values": rr[:50].tolist() if len(rr) >= 50 else rr.tolist()
            })

        print(f"  Output: {len(rate_recovered)} blocks")
        print(f"  Lengths: {recovered_lengths}")
    else:
        save_stage_results(log_file, "STAGE 5: RATE RECOVERY", {
            "Output length": len(rate_recovered),
            "First 50 values": rate_recovered[:50].tolist() if len(rate_recovered) >= 50 else rate_recovered.tolist()
        })
        print(f"  Output: {len(rate_recovered)} soft bits")

    # Stage 6: Turbo decoding
    print("\n[6] Turbo Decoding (Max-Log-MAP, 5 iterations)...")
    turbo_decoded = lteTurboDecode(rate_recovered, nturbodecits=5)

    if isinstance(turbo_decoded, list):
        total_decoded_bits = sum(len(cb) for cb in turbo_decoded)
        decoded_lengths = [len(cb) for cb in turbo_decoded]

        save_stage_results(log_file, "STAGE 6: TURBO DECODING", {
            "Number of blocks": len(turbo_decoded),
            "Block lengths": decoded_lengths,
            "Total decoded bits": total_decoded_bits,
            "Iterations": 5,
            "Algorithm": "Max-Log-MAP"
        })

        for idx, td in enumerate(turbo_decoded):
            save_stage_results(log_file, f"  Turbo Decoded Block {idx}", {
                "Length": len(td),
                "Data": td
            })

        print(f"  Output: {len(turbo_decoded)} blocks")
        print(f"  Lengths: {decoded_lengths}")
    else:
        save_stage_results(log_file, "STAGE 6: TURBO DECODING", {
            "Output length": len(turbo_decoded),
            "Data": turbo_decoded,
            "Iterations": 5,
            "Algorithm": "Max-Log-MAP"
        })
        print(f"  Output: {len(turbo_decoded)} bits")

    # Stage 7: Code block desegmentation
    print("\n[7] Code Block Desegmentation...")
    desegmented, crc_errors = lteCodeBlockDesegment(turbo_decoded, blklen=len(test_bits))

    save_stage_results(log_file, "STAGE 7: CODE BLOCK DESEGMENTATION", {
        "Input blocks": seg_info['C'],
        "Expected block length": len(test_bits),
        "Output length": len(desegmented),
        "CRC errors per block": crc_errors.tolist() if len(crc_errors) > 0 else "N/A (single block)",
        "Data": desegmented
    })

    print(f"  Output: {len(desegmented)} bits")
    if len(crc_errors) > 0:
        print(f"  Block CRC errors: {crc_errors.tolist()}")
        if np.all(crc_errors == 0):
            print(f"  ✓ All block CRCs passed")
        else:
            print(f"  ✗ Some block CRCs failed!")

    # Stage 8: CRC-24A decoding
    print("\n[8] CRC-24A Decoding...")

    # Desegmented output includes transport CRC (data + 24-bit CRC)
    # Decode to remove CRC and check for errors
    decoded_data, crc_error = lteCRCDecode(desegmented, '24A', mask=0)

    save_stage_results(log_file, "STAGE 8: CRC-24A DECODING", {
        "Input length": len(desegmented),
        "CRC length": 24,
        "Output length": len(decoded_data),
        "CRC error value": int(crc_error),
        "CRC status": "PASS" if crc_error == 0 else "FAIL",
        "Decoded data": decoded_data
    })

    print(f"  CRC error: {crc_error}")
    if crc_error == 0:
        print(f"  ✓ Transport block CRC passed")
    else:
        print(f"  ✗ Transport block CRC failed!")

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    # Compare decoded with original
    if len(decoded_data) == len(test_bits):
        errors = np.sum(decoded_data != test_bits)
        ber = errors / len(test_bits)

        verification = {
            "Original length": len(test_bits),
            "Decoded length": len(decoded_data),
            "Bit errors": int(errors),
            "Bit Error Rate (BER)": float(ber),
            "Match": "PERFECT" if errors == 0 else "FAILED"
        }

        if errors == 0:
            print(f"✓ PERFECT MATCH!")
            print(f"  Original: {len(test_bits)} bits")
            print(f"  Decoded:  {len(decoded_data)} bits")
            print(f"  Errors:   {errors} bits")
            print(f"  BER:      {ber:.2e}")
        else:
            print(f"✗ ERRORS DETECTED!")
            print(f"  Original: {len(test_bits)} bits")
            print(f"  Decoded:  {len(decoded_data)} bits")
            print(f"  Errors:   {errors} bits")
            print(f"  BER:      {ber:.2e}")

            # Show first few errors
            error_positions = np.where(decoded_data != test_bits)[0]
            print(f"  First 10 error positions: {error_positions[:10].tolist()}")
            verification["Error positions (first 10)"] = error_positions[:10].tolist()
    else:
        print(f"✗ LENGTH MISMATCH!")
        print(f"  Original: {len(test_bits)} bits")
        print(f"  Decoded:  {len(decoded_data)} bits")
        verification = {
            "Original length": len(test_bits),
            "Decoded length": len(decoded_data),
            "Match": "LENGTH MISMATCH"
        }

    save_stage_results(log_file, "FINAL VERIFICATION", verification)

    print(f"\nDetailed results saved to: roundtrip_results/{log_file}")

    return decoded_data, errors if len(decoded_data) == len(test_bits) else -1


def main():
    """Main test function"""

    print("="*80)
    print("LTE CHANNEL CODING ROUND-TRIP TEST")
    print("Testing complete encode-decode pipeline")
    print("="*80)

    # Base sequence (64 bits)
    binary_str = '0000000100110010010001010111011011001101111111101000100110111010'
    base_seq = np.array([int(b) for b in binary_str], dtype=np.int8)

    print(f"\nBase sequence: {len(base_seq)} bits")
    print(f"Binary: {binary_str}")

    # Test sizes
    test_sizes = [40, 1088, 6144]

    results_summary = []

    for size in test_sizes:
        # Create test sequence
        test_seq = create_test_sequence(base_seq, size)

        # Run round-trip test
        test_name = f"{size}_bits"
        log_file = f"roundtrip_{size}_bits.txt"

        decoded, errors = test_roundtrip(test_seq, test_name, log_file)

        results_summary.append({
            'size': size,
            'errors': errors,
            'status': 'PASS' if errors == 0 else 'FAIL'
        })

    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)

    summary_file = "roundtrip_results/summary.txt"
    with open(summary_file, 'w') as f:
        f.write("LTE CHANNEL CODING ROUND-TRIP TEST SUMMARY\n")
        f.write("="*80 + "\n\n")

        for result in results_summary:
            line = f"Size: {result['size']:5d} bits | Errors: {result['errors']:5d} | Status: {result['status']}"
            print(line)
            f.write(line + "\n")

        f.write("\n")

        # Overall result
        all_passed = all(r['errors'] == 0 for r in results_summary)
        if all_passed:
            line = "\n✓ ALL TESTS PASSED!"
            print(line)
            f.write(line + "\n")
        else:
            line = "\n✗ SOME TESTS FAILED!"
            print(line)
            f.write(line + "\n")

    print(f"\nSummary saved to: {summary_file}")
    print("\nAll detailed results saved in: roundtrip_results/")


if __name__ == "__main__":
    main()
