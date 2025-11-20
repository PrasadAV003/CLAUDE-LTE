"""
Minimal SC-FDMA test without windowing or complex phase shifts
"""
import numpy as np

# Test parameters
nFFT = 128
nSC = 72
cpLength = 10

# Simple input: all ones
grid_in = np.ones(nSC, dtype=np.complex128)

print("="*60)
print("TEST: Basic SC-FDMA without phase shifts or windowing")
print("="*60)

# MODULATE
# Step 1: Map to FFT input
freq_array = np.zeros(nFFT, dtype=np.complex128)
firstSC = nFFT//2 - nSC//2  # 28
freq_array[firstSC:firstSC+nSC] = grid_in

print(f"Subcarrier mapping: bins {firstSC} to {firstSC+nSC-1}")

# Step 2: fftshift then IFFT (MATLAB style)
freq_shifted = np.fft.fftshift(freq_array)
iffout = np.fft.ifft(freq_shifted)  # numpy normalizes by default

print(f"IFFT output: mean magnitude = {np.mean(np.abs(iffout)):.6f}")

# Step 3: Add CP (no phase shift, no windowing)
waveform = np.concatenate([iffout[-cpLength:], iffout])

print(f"Waveform length: {len(waveform)} (CP={cpLength} + FFT={nFFT})")

# DEMODULATE
# Step 4: Skip CP and extract samples
cpFraction = 0.55
fftStart = int(cpLength * cpFraction)

print(f"\nDemodulator: fftStart = {fftStart}")

samples = waveform[fftStart:fftStart+nFFT]

print(f"Extracted samples: from index {fftStart} to {fftStart+nFFT-1}")

# Step 5: FFT then fftshift (no phase correction)
fftout = np.fft.fft(samples)
fftout = np.fft.fftshift(fftout)

print(f"FFT output: mean magnitude = {np.mean(np.abs(fftout)):.6f}")

# Step 6: Extract active subcarriers
firstActiveSC = nFFT//2 - nSC//2
grid_out = fftout[firstActiveSC:firstActiveSC+nSC]

# Compare
print(f"\nResults:")
print(f"Input[0]: {grid_in[0]:.6f}")
print(f"Output[0]: {grid_out[0]:.6f}")

mse = np.mean(np.abs(grid_in - grid_out)**2)
print(f"MSE: {mse:.6e}")

if mse < 1e-10:
    print("✓ Basic SC-FDMA works!")
else:
    print("✗ Basic SC-FDMA has errors")
    print(f"\nInput mean magnitude: {np.mean(np.abs(grid_in)):.6f}")
    print(f"Output mean magnitude: {np.mean(np.abs(grid_out)):.6f}")
    print(f"Magnitude ratio: {np.mean(np.abs(grid_out))/np.mean(np.abs(grid_in)):.6f}")
