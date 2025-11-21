"""
Very simple debug of SC-FDMA without ANY phase shifts or windowing
"""
import numpy as np

# Simplest possible test
nFFT = 128
nSC = 72
cpLength = 10

# Simple input: single pulse
grid_in = np.zeros(nSC, dtype=np.complex128)
grid_in[0] = 1.0 + 0j

print("="*60)
print("ULTRA-SIMPLE TEST: No phase shifts, no windowing")
print("="*60)
print(f"Input: {grid_in[:5]}")

# TX: Map to frequency domain
freq = np.zeros(nFFT, dtype=np.complex128)
firstSC = nFFT//2 - nSC//2  # 28
freq[firstSC:firstSC+nSC] = grid_in

print(f"\nMapped to bins {firstSC} to {firstSC+nSC-1}")
print(f"Non-zero before fftshift: {np.where(np.abs(freq) > 0.5)[0]}")

# fftshift
freq_shifted = np.fft.fftshift(freq)
print(f"Non-zero after fftshift: {np.where(np.abs(freq_shifted) > 0.5)[0]}")

# IFFT
time = np.fft.ifft(freq_shifted)
print(f"\nTime domain: max={np.max(np.abs(time)):.6f}, energy={np.sum(np.abs(time)**2):.6f}")

# Add CP (no phase shift)
waveform = np.concatenate([time[-cpLength:], time])
print(f"Waveform length: {len(waveform)}")

# RX: Skip CP
cpFraction = 0.55
fftStart = int(cpLength * cpFraction)
samples = waveform[fftStart:fftStart+nFFT]

print(f"\nRX: Extract from index {fftStart} to {fftStart+nFFT-1}")

# FFT (no phase correction)
freq_rx = np.fft.fft(samples)
print(f"After FFT, max bin: {np.argmax(np.abs(freq_rx))}, max val: {np.max(np.abs(freq_rx)):.6f}")

# fftshift
freq_rx_shifted = np.fft.fftshift(freq_rx)
print(f"After fftshift, max bin: {np.argmax(np.abs(freq_rx_shifted))}, max val: {np.max(np.abs(freq_rx_shifted)):.6f}")

# Extract
grid_out = freq_rx_shifted[firstSC:firstSC+nSC]

print(f"\nOutput: {grid_out[:5]}")
print(f"Input[0]: {grid_in[0]}")
print(f"Output[0]: {grid_out[0]}")

mse = np.mean(np.abs(grid_in - grid_out)**2)
print(f"\nMSE: {mse:.6e}")

if mse < 1e-10:
    print("✓ Simple test works!")
else:
    print("✗ Simple test fails")
    print(f"Ratio output/input: {np.abs(grid_out[0])/np.abs(grid_in[0]):.6f}")
    print(f"Phase error: {np.angle(grid_out[0]) - np.angle(grid_in[0]):.6f} rad")
