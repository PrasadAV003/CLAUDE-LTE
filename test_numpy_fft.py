"""
Test SC-FDMA with pure numpy FFT (no PyFFTW) to isolate PyFFTW issues
"""
import numpy as np

def scfdma_modulate_numpy(grid, nFFT, cpLength):
    """Simple SC-FDMA modulator using numpy"""
    nSC = len(grid)
    firstSC = nFFT//2 - nSC//2

    # Map to IFFT input
    freq = np.zeros(nFFT, dtype=np.complex128)
    freq[firstSC:firstSC+nSC] = grid

    # fftshift then IFFT
    freq = np.fft.fftshift(freq)
    time = np.fft.ifft(freq)

    # Add CP with half-SC shift
    extended = np.concatenate([time[-cpLength:], time])
    phase_idx = np.arange(-cpLength, nFFT)
    phase = np.exp(1j * np.pi * phase_idx / nFFT)
    waveform = extended * phase

    return waveform

def scfdma_demodulate_numpy(waveform, nSC, nFFT, cpLength, cpFraction=0.55):
    """Simple SC-FDMA demodulator using numpy"""
    firstSC = nFFT//2 - nSC//2
    fftStart = int(cpLength * cpFraction)

    # Extract samples
    samples = waveform[fftStart:fftStart+nFFT]

    # Half-SC shift correction
    idx = np.arange(nFFT)
    halfsc = np.exp(1j * np.pi / nFFT * (idx + fftStart - cpLength))
    samples = samples * halfsc

    # FFT
    freq = np.fft.fft(samples)

    # Phase correction BEFORE fftshift
    phaseCorr = np.exp(-1j * 2 * np.pi * (cpLength - fftStart) / nFFT * idx)
    freq = freq * phaseCorr

    # fftshift
    freq = np.fft.fftshift(freq)

    # Extract
    grid = freq[firstSC:firstSC+nSC]

    return grid

# Test
print("="*60)
print("Pure NumPy FFT Test")
print("="*60)

nFFT = 128
nSC = 72
cpLength = 10

# Random input
np.random.seed(42)
grid_in = np.random.randn(nSC) + 1j * np.random.randn(nSC)

# Modulate
waveform = scfdma_modulate_numpy(grid_in, nFFT, cpLength)
print(f"Input shape: {grid_in.shape}")
print(f"Waveform shape: {waveform.shape}")

# Demodulate
grid_out = scfdma_demodulate_numpy(waveform, nSC, nFFT, cpLength)
print(f"Output shape: {grid_out.shape}")

# Check
mse = np.mean(np.abs(grid_in - grid_out)**2)
print(f"\nMSE: {mse:.6e}")

if mse < 1e-15:
    print("✓ NumPy FFT works perfectly!")
elif mse < 1e-10:
    print("✓ NumPy FFT works well!")
else:
    print("✗ NumPy FFT also fails")
    print(f"\nInput magnitude: mean={np.mean(np.abs(grid_in)):.6f}")
    print(f"Output magnitude: mean={np.mean(np.abs(grid_out)):.6f}")
    print(f"Ratio: {np.mean(np.abs(grid_out))/np.mean(np.abs(grid_in)):.6f}")
