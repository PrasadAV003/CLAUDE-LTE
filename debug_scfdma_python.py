"""
Simple SC-FDMA Round-trip Test - Matching MATLAB Example
Based on MATLAB lteSCFDMAModulate documentation
"""

import numpy as np
from lte_scfdma_modulate import lteSCFDMAModulate
from lte_scfdma_demodulate import lteSCFDMADemodulate

print('=' * 60)
print('PYTHON SC-FDMA Debug Output')
print('=' * 60)
print()

# UE configuration - Basic LTE
ue = {
    'NULRB': 6,              # 6 RBs = 72 subcarriers
    'CyclicPrefixUL': 'Normal',
    'NTxAnts': 1
}

# Create resource grid
print('=== INPUT ===')
nSC = ue['NULRB'] * 12    # 72 subcarriers
nSymbols = 14             # 1 subframe (2 slots)

# Generate QPSK symbols - matching MATLAB rng(42)
np.random.seed(42)
nBits = nSC * nSymbols * 2  # 2 bits per QPSK symbol
bits = np.random.randint(0, 2, nBits)

# QPSK modulation - Python style (matching MATLAB lteSymbolModulate)
# QPSK mapping: [00->-1-1j, 01->-1+1j, 10->1-1j, 11->1+1j] / sqrt(2)
bit_pairs = bits.reshape(-1, 2)
qpsk_map = {
    (0, 0): -1-1j,
    (0, 1): -1+1j,
    (1, 0): 1-1j,
    (1, 1): 1+1j
}
symbols = np.array([qpsk_map[tuple(pair)] for pair in bit_pairs]) / np.sqrt(2)

# Reshape into grid [nSC x nSymbols]
grid_in = symbols.reshape(nSC, nSymbols)

print(f'Grid shape: [{grid_in.shape[0]}, {grid_in.shape[1]}]')
print(f'First 5 subcarriers of symbol 1:')
print(grid_in[0:5, 0])
print(f'Grid power: {np.sum(np.abs(grid_in)**2):.6f}')
print()

# MODULATE
print('=== MODULATION ===')
result = lteSCFDMAModulate(ue, grid_in)
print(f'Result type: {type(result)}')
print(f'Result: {result}')

# Handle both tuple return (waveform, info) and direct waveform return
if isinstance(result, tuple):
    waveform = result[0]
    info = result[1]
    print(f'Info: {info}')
else:
    waveform = result

print(f'Waveform shape: {waveform.shape}')
print(f'First 5 samples:')
print(waveform[0:5, 0] if waveform.ndim > 1 else waveform[0:5])
print(f'Waveform power: {np.sum(np.abs(waveform)**2):.6f}')
print()

# DEMODULATE
print('=== DEMODULATION ===')
grid_out = lteSCFDMADemodulate(ue, waveform)

# Squeeze antenna dimension if present
if grid_out.ndim == 3:
    grid_out = grid_out.squeeze(axis=2)

print(f'Output grid shape: [{grid_out.shape[0]}, {grid_out.shape[1]}]')
print(f'First 5 subcarriers of symbol 1:')
print(grid_out[0:5, 0])
print(f'Output power: {np.sum(np.abs(grid_out)**2):.6f}')
print()

# ERROR CALCULATION
print('=== ERROR ===')
error = grid_in - grid_out
mse = np.mean(np.abs(error)**2)
max_error = np.max(np.abs(error))
print(f'MSE: {mse:.6e}')
print(f'Max error: {max_error:.6e}')
print(f'First 5 errors (symbol 1): {np.abs(error[0:5, 0])}')
print()

if mse < 1e-20:
    print('✓ Perfect round-trip!')
elif mse < 1e-10:
    print('✓ Good round-trip (acceptable numerical error)')
else:
    print('✗ Round-trip has significant errors')

print()
print('=' * 60)
