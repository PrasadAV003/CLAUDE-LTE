"""
Python SC-FDMA Debug Script
Run this and compare outputs with MATLAB version
"""
import numpy as np
from lte_scfdma_modulate import lteSCFDMAModulate
from lte_scfdma_demodulate import lteSCFDMADemodulate

print('='*60)
print('PYTHON SC-FDMA Debug Output')
print('='*60)
print()

# Configuration
ue = {
    'NULRB': 6,
    'CyclicPrefixUL': 'Normal',
    'Windowing': 0  # No windowing for simplicity
}

# Simple test input: 14 symbols (2 slots), using fixed binary data
nSC = 72
nSymbols = 14  # 2 slots for Normal CP (7 symbols per slot)

# Fixed binary sequence for reproducibility
binary_str = '0000000100110010010001010111011011001101111111101000100110111010'
base_seq = np.array([int(b) for b in binary_str], dtype=np.int8)

# Convert binary to BPSK symbols (0→-1, 1→+1)
bpsk_seq = 2 * base_seq - 1

# Replicate to fill nSC subcarriers
full_seq = np.tile(bpsk_seq, int(np.ceil(nSC / len(bpsk_seq))))
full_seq = full_seq[:nSC]

# Create grid with same sequence for all symbols (as float, not complex for now)
grid_in = np.tile(full_seq.reshape(-1, 1), (1, nSymbols)).astype(np.complex128)

print('=== INPUT ===')
print(f'Grid shape: [{grid_in.shape[0]}, {grid_in.shape[1]}]')
print(f'First 5 subcarriers of symbol 1: {grid_in[0:5, 0]}')
print()

# MODULATE using Python
print('=== MODULATION ===')
waveform, info = lteSCFDMAModulate(ue, grid_in)

print(f'NFFT: {info.Nfft}')
print(f'Windowing: {info.Windowing}')
print(f'CP lengths: {info.CyclicPrefixLengths}')
print(f'Waveform shape: [{waveform.shape[0]}, {waveform.shape[1]}]')
print(f'Waveform[0:5]: {waveform[0:5, 0]}')
print(f'Waveform power: {np.sum(np.abs(waveform)**2):.6f}')
print()

# DEMODULATE using Python
print('=== DEMODULATION ===')
grid_out = lteSCFDMADemodulate(ue, waveform)

# Squeeze antenna dimension if present
if grid_out.ndim == 3:
    grid_out = grid_out.squeeze(axis=2)

print(f'Output grid shape: [{grid_out.shape[0]}, {grid_out.shape[1]}]')
print(f'First 5 subcarriers of symbol 1: {grid_out[0:5, 0]}')
print()

# ERROR CALCULATION
print('=== ERROR ===')
error = grid_in - grid_out
mse = np.mean(np.abs(error)**2)
print(f'MSE: {mse:.6e}')
print(f'Max error: {np.max(np.abs(error)):.6e}')
print(f'First 5 errors (symbol 1): {np.abs(error[0:5, 0])}')
print()

if mse < 1e-20:
    print('✓ Perfect round-trip!')
elif mse < 1e-10:
    print('✓ Excellent round-trip!')
else:
    print('✗ Round-trip has errors')

# MANUAL STEP-BY-STEP (Single symbol, no windowing)
print()
print('='*60)
print('MANUAL STEP-BY-STEP (Symbol 1)')
print('='*60)
print()

nFFT = info.Nfft
cpLength = info.CyclicPrefixLengths[0]
firstSC = nFFT//2 - nSC//2  # Python 0-indexed

print('--- TX: Subcarrier Mapping ---')
freq_in = np.zeros(nFFT, dtype=np.complex128)
freq_in[firstSC:firstSC+nSC] = grid_in[:, 0]
print(f'Mapped to bins {firstSC} to {firstSC+nSC-1} (Python 0-indexed)')
print(f'Non-zero bins: {np.where(np.abs(freq_in) > 0.5)[0]}')
print(f'freq_in[{firstSC}] = {freq_in[firstSC]}')
print()

print('--- TX: fftshift ---')
freq_shifted = np.fft.fftshift(freq_in)
print(f'After fftshift, non-zero bins: {np.where(np.abs(freq_shifted) > 0.5)[0]}')
print(f'freq_shifted[0] = {freq_shifted[0]}')
print()

print('--- TX: IFFT ---')
time_out = np.fft.ifft(freq_shifted)
print(f'time_out[0:3] = {time_out[0:3]}')
print(f'Time power: {np.sum(np.abs(time_out)**2):.6f}')
print()

print('--- TX: Add CP with half-SC shift ---')
extended = np.concatenate([time_out[-cpLength:], time_out])
phase_idx = np.arange(-cpLength, nFFT)
phase = np.exp(1j * np.pi * phase_idx / nFFT)
tx_waveform = extended * phase
print(f'CP length: {cpLength}')
print(f'Extended length: {len(extended)}')
print(f'tx_waveform[0:3] = {tx_waveform[0:3]}')
print()

print('--- RX: Skip CP ---')
cpFraction = 0.55
fftStart = int(cpLength * cpFraction)  # Python 0-indexed
samples = tx_waveform[fftStart:fftStart+nFFT]
print(f'fftStart: {fftStart} (Python 0-indexed)')
print(f'Extract indices {fftStart} to {fftStart+nFFT-1}')
print(f'samples[0:3] = {samples[0:3]}')
print()

print('--- RX: Half-SC shift correction ---')
idx = np.arange(nFFT)
halfsc = np.exp(1j * np.pi / nFFT * (idx + fftStart - cpLength))
samples = samples * halfsc
print(f'samples after halfsc[0:3] = {samples[0:3]}')
print()

print('--- RX: FFT ---')
freq_rx = np.fft.fft(samples)
print(f'After FFT[0:3] = {freq_rx[0:3]}')
print()

print('--- RX: Phase correction (BEFORE fftshift) ---')
phaseCorr = np.exp(-1j * 2 * np.pi * (cpLength - fftStart) / nFFT * idx)
freq_rx = freq_rx * phaseCorr
print(f'After phase corr[0:3] = {freq_rx[0:3]}')
print()

print('--- RX: fftshift ---')
freq_rx_shifted = np.fft.fftshift(freq_rx)
print(f'After fftshift[{firstSC}] = {freq_rx_shifted[firstSC]}')
print()

print('--- RX: Extract subcarriers ---')
grid_rx = freq_rx_shifted[firstSC:firstSC+nSC]
print(f'Extracted bins {firstSC} to {firstSC+nSC-1}')
print(f'grid_rx[0:5] = {grid_rx[0:5]}')
print()

print('--- COMPARISON ---')
manual_error = np.abs(grid_in[:, 0] - grid_rx)
print(f'Manual MSE: {np.mean(manual_error**2):.6e}')
print(f'Manual max error: {np.max(manual_error):.6e}')
print(f'First 5 errors: {manual_error[0:5]}')
