"""
Python equivalent of MATLAB lteSCFDMAModulate documentation example
Adapted for standard LTE (our implementation uses standard LTE, not NB-IoT)
"""

import numpy as np
from lte_scfdma_modulate import lteSCFDMAModulate

# UE configuration - Standard LTE uplink
# (MATLAB example uses NB-IoT, but we adapt to standard LTE)
ue = {
    'NULRB': 6,                    # 6 resource blocks = 72 subcarriers
    'CyclicPrefixUL': 'Normal',    # Normal CP
    'NTxAnts': 1                   # Single antenna
}

# Resource grid dimensions
nSubcarriers = ue['NULRB'] * 12   # 72 subcarriers (12 per RB)
nSymbols = 14                      # 1 subframe = 14 OFDM symbols (Normal CP)

# Generate random QPSK symbols
# QPSK: 2 bits per symbol
np.random.seed(42)  # For reproducibility
nBits = nSubcarriers * nSymbols * 2
bits = np.random.randint(0, 2, nBits)

# QPSK modulation - Gray coding
# Bit pairs: 00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
# Then normalize by 1/sqrt(2)
bit_pairs = bits.reshape(-1, 2)
qpsk_map = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j
}
symbols = np.array([qpsk_map[tuple(pair)] for pair in bit_pairs]) / np.sqrt(2)

# Create resource grid [nSubcarriers x nSymbols]
grid = symbols.reshape(nSubcarriers, nSymbols)

print("Resource Grid Configuration:")
print(f"  Subcarriers: {nSubcarriers}")
print(f"  Symbols: {nSymbols}")
print(f"  Grid shape: {grid.shape}")
print()

# Perform SC-FDMA modulation
result = lteSCFDMAModulate(ue, grid)

# Handle tuple return (waveform, info)
if isinstance(result, tuple):
    waveform, info = result
    print("Modulation Info:")
    print(f"  NFFT: {info.Nfft}")
    print(f"  Windowing: {info.Windowing}")
    print(f"  Sampling Rate: {info.SamplingRate/1e6:.2f} MHz")
    print(f"  CP Lengths: {info.CyclicPrefixLengths}")
    print()
else:
    waveform = result

# Display first 5 samples (matching MATLAB example)
print("First 5 waveform samples:")
if waveform.ndim > 1:
    for i in range(5):
        sample = waveform[i, 0]
        print(f"  {sample.real:8.4f} {sample.imag:+8.4f}i")
else:
    for i in range(5):
        sample = waveform[i]
        print(f"  {sample.real:8.4f} {sample.imag:+8.4f}i")

print()
print(f"Total waveform length: {len(waveform)} samples")
print(f"Waveform power: {np.sum(np.abs(waveform)**2):.6f}")
