"""
Simple test for turbo encode/decode without rate matching
"""

import numpy as np
from ctr_encode import lteTurboEncode
from ctr_decode import lteTurboDecode

# Simple 40-bit test
test_bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                      0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                      1, 1, 0, 0, 1, 1, 0, 1], dtype=np.int8)

print("Simple Turbo Encode/Decode Test")
print("="*60)
print(f"Input: {len(test_bits)} bits")
print(f"Data: {test_bits.tolist()}")

# Encode
print("\nEncoding...")
encoded = lteTurboEncode(test_bits)
print(f"Encoded: {len(encoded)} bits (expected {3*(40+4)}={3*44})")

# Perfect channel: convert to soft values (LLRs)
# LLR convention: positive = bit 0, negative = bit 1
llr = np.where(encoded == 0, 5.0, -5.0).astype(float)
print(f"LLRs: {len(llr)} soft values")

# Decode
print("\nDecoding...")
decoded = lteTurboDecode(llr, nturbodecits=5)
print(f"Decoded: {len(decoded)} bits")
print(f"Data: {decoded.tolist()}")

# Compare
errors = np.sum(decoded != test_bits)
print(f"\nErrors: {errors}/{len(test_bits)}")

if errors == 0:
    print("✓ PERFECT MATCH!")
else:
    print("✗ ERRORS DETECTED!")
    print(f"Error positions: {np.where(decoded != test_bits)[0].tolist()}")
