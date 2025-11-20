"""
Debug SC-FDMA modulator/demodulator
"""
import numpy as np

# Simple test without using the classes
nFFT = 128
nSC = 72
nSymbols = 1  # Just one symbol
cpLength = 10

# Create simple test signal
grid_in = np.ones((nSC, 1), dtype=np.complex128)
grid_in[0] = 1+0j  # DC component

# Modulate manually
freq_array = np.zeros(nFFT, dtype=np.complex128)
firstSC = nFFT//2 - nSC//2  # 28
freq_array[firstSC:firstSC+nSC] = grid_in[:, 0]

print(f"First active SC: {firstSC}")
print(f"Last active SC: {firstSC+nSC-1}")
print(f"Non-zero bins: {np.where(np.abs(freq_array) > 0)[0]}")

# fftshift before IFFT (MATLAB style)
freq_shifted = np.fft.fftshift(freq_array)
print(f"\nAfter fftshift, non-zero bins: {np.where(np.abs(freq_shifted) > 0)[0]}")

# IFFT with normalization
iffout = np.fft.ifft(freq_shifted)  # Use numpy's normalized ifft
print(f"\nIFFT output magnitude: mean={np.mean(np.abs(iffout)):.6f}, max={np.max(np.abs(iffout)):.6f}")

# Add CP (no windowing)
extended = np.concatenate([iffout[-cpLength:], iffout])
print(f"Extended length: {len(extended)} (should be {cpLength + nFFT})")

# Apply half-SC shift
phase_indices = np.arange(-cpLength, nFFT)
phase_shift = np.exp(1j * np.pi * phase_indices / nFFT)
extended_shifted = extended * phase_shift

# This is the transmitted waveform
waveform = extended_shifted

# Demodulate
cpFraction = 0.55
fftStart = int(cpLength * cpFraction)  # 5
print(f"\nfftStart: {fftStart}")

# Extract samples for FFT
samples = waveform[fftStart:fftStart+nFFT]
print(f"Extracted samples length: {len(samples)}")

# Apply half-SC shift correction
idx = np.arange(nFFT)
halfsc = np.exp(1j * np.pi / nFFT * (idx + fftStart - cpLength))
samples_corrected = samples * halfsc

# FFT
fftOutput = np.fft.fft(samples_corrected)

# fftshift
fftOutput = np.fft.fftshift(fftOutput)

# Phase correction
phaseCorrection = np.exp(-1j * 2 * np.pi * (cpLength - fftStart) / nFFT * idx)
fftOutput = fftOutput * phaseCorrection

# Extract active subcarriers
firstActiveSC = nFFT//2 - nSC//2
grid_out = fftOutput[firstActiveSC:firstActiveSC+nSC]

# Compare
print(f"\nInput grid[0]: {grid_in[0, 0]}")
print(f"Output grid[0]: {grid_out[0]}")
print(f"Difference: {grid_in[0, 0] - grid_out[0]}")

mse = np.mean(np.abs(grid_in[:, 0] - grid_out)**2)
print(f"\nMSE: {mse:.6e}")

# Check magnitudes
print(f"\nInput magnitude: mean={np.mean(np.abs(grid_in)):.6f}")
print(f"Output magnitude: mean={np.mean(np.abs(grid_out)):.6f}")
print(f"Ratio: {np.mean(np.abs(grid_out)) / np.mean(np.abs(grid_in)):.6f}")
