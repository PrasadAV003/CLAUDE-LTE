"""
SC-FDMA Round-trip Test with Fixed Binary Data
Standard LTE uplink modulation/demodulation test
"""

import numpy as np
from lte_scfdma_modulate import lteSCFDMAModulate
from lte_scfdma_demodulate import lteSCFDMADemodulate

print('=' * 60)
print('PYTHON SC-FDMA Debug Output (Fixed Binary Data)')
print('=' * 60)
print()

# Fixed binary data sequence
binary_str = '0000000100110010010001010111011011001101111111101000100110111010'
print(f'Binary sequence: {binary_str}')
print(f'Binary length: {len(binary_str)} bits')
print()

# UE configuration - Standard LTE
ue = {
    'NULRB': 1,                    # 1 RB = 12 subcarriers
    'CyclicPrefixUL': 'Normal',
    'NTxAnts': 1
}

# Resource grid dimensions
nSC = ue['NULRB'] * 12    # 12 subcarriers
nSymbols = 7              # 7 OFDM symbols (1 slot, Normal CP)

# Convert binary string to array
nBits = nSC * nSymbols * 2  # QPSK: 2 bits per symbol
print(f'Required bits for QPSK: {nBits}')

# Use the binary sequence (repeat if needed)
bits_needed = nBits
binary_data = (binary_str * (bits_needed // len(binary_str) + 1))[:bits_needed]

# Convert to numeric array
bits = np.array([int(b) for b in binary_data])

print(f'Using first {bits_needed} bits from sequence')
print()

# QPSK modulation - Gray coding
# Bit pairs: 00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
# Normalized by 1/sqrt(2)
bit_pairs = bits.reshape(-1, 2)
qpsk_map = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j
}
symbols = np.array([qpsk_map[tuple(pair)] for pair in bit_pairs]) / np.sqrt(2)

# Reshape into grid [nSC x nSymbols]
grid_in = symbols.reshape(nSC, nSymbols)

print('=== INPUT GRID ===')
print(f'Grid shape: [{grid_in.shape[0]}, {grid_in.shape[1]}]')
print(f'Grid power: {np.sum(np.abs(grid_in)**2):.6f}')
print('First 5 subcarriers of symbol 1:')
for i in range(5):
    print(f'  grid({i+1},1) = {grid_in[i,0].real:.6f}{grid_in[i,0].imag:+.6f}i')
print()

# SC-FDMA MODULATION
print('=== MODULATION ===')
result = lteSCFDMAModulate(ue, grid_in)

# Handle tuple return (waveform, info)
if isinstance(result, tuple):
    waveform, info = result
else:
    waveform = result

print(f'Waveform length: {len(waveform)} samples')
print(f'Waveform power: {np.sum(np.abs(waveform)**2):.6f}')
print('First 5 waveform samples:')
for i in range(5):
    sample = waveform[i, 0] if waveform.ndim > 1 else waveform[i]
    print(f'  waveform({i+1}) = {sample.real:.6f}{sample.imag:+.6f}i')
print()

# SC-FDMA DEMODULATION
print('=== DEMODULATION ===')
grid_rx = lteSCFDMADemodulate(ue, waveform)

# Squeeze antenna dimension if present
if grid_rx.ndim == 3:
    grid_rx = grid_rx.squeeze(axis=2)

print(f'Recovered grid shape: [{grid_rx.shape[0]}, {grid_rx.shape[1]}]')
print(f'Recovered power: {np.sum(np.abs(grid_rx)**2):.6f}')
print('First 5 subcarriers of symbol 1:')
for i in range(5):
    print(f'  grid_rx({i+1},1) = {grid_rx[i,0].real:.6f}{grid_rx[i,0].imag:+.6f}i')
print()

# ERROR ANALYSIS
print('=== ERROR ANALYSIS ===')
error = grid_in - grid_rx
mse = np.mean(np.abs(error)**2)
max_error = np.max(np.abs(error))
print(f'MSE: {mse:.6e}')
print(f'Max error: {max_error:.6e}')
print('First 5 errors (magnitude):')
for i in range(5):
    print(f'  |error({i+1},1)| = {np.abs(error[i,0]):.6e}')
print()

if mse < 1e-20:
    print('✓ PERFECT round-trip!')
elif mse < 1e-10:
    print('✓ Good round-trip (acceptable numerical error)')
else:
    print('✗ FAILED - Round-trip has significant errors')

print()
print('=' * 60)
