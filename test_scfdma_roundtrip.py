"""
Test SC-FDMA round-trip with windowing
"""
import numpy as np
from lte_scfdma_modulate import lteSCFDMAModulate
from lte_scfdma_demodulate import lteSCFDMADemodulate

# Test with windowing=0 first
print("="*60)
print("Test 1: Without windowing")
print("="*60)
ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal', 'Windowing': 0}
grid_original = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

waveform, info = lteSCFDMAModulate(ue, grid_original)
print(f"Modulator info: NFFT={info.Nfft}, Windowing={info.Windowing}, SamplingRate={info.SamplingRate/1e6:.2f}MHz")
print(f"Waveform shape: {waveform.shape}")

grid_recovered = lteSCFDMADemodulate(ue, waveform)
print(f"Recovered shape: {grid_recovered.shape}")

mse = np.mean(np.abs(grid_original - grid_recovered)**2)
print(f"MSE: {mse:.2e}")
if mse < 1e-15:
    print("✓ Excellent round-trip!")
else:
    print("✗ Round-trip has errors")

# Test with default windowing
print("\n" + "="*60)
print("Test 2: With default windowing")
print("="*60)
ue2 = {'NULRB': 6, 'CyclicPrefixUL': 'Normal'}
grid_original2 = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

waveform2, info2 = lteSCFDMAModulate(ue2, grid_original2)
print(f"Modulator info: NFFT={info2.Nfft}, Windowing={info2.Windowing}, SamplingRate={info2.SamplingRate/1e6:.2f}MHz")
print(f"Waveform shape: {waveform2.shape}")

grid_recovered2 = lteSCFDMADemodulate(ue2, waveform2)
print(f"Recovered shape: {grid_recovered2.shape}")

mse2 = np.mean(np.abs(grid_original2 - grid_recovered2)**2)
print(f"MSE: {mse2:.2e}")
if mse2 < 1e-15:
    print("✓ Excellent round-trip!")
else:
    print("✗ Round-trip has errors")

# Test with different cpFraction values
print("\n" + "="*60)
print("Test 3: With windowing and different cpFraction")
print("="*60)
for cpf in [0.0, 0.25, 0.5, 0.55, 0.75, 1.0]:
    grid_recovered3 = lteSCFDMADemodulate(ue2, waveform2, cpf)
    mse3 = np.mean(np.abs(grid_original2 - grid_recovered3)**2)
    status = "✓" if mse3 < 1e-10 else "✗"
    print(f"  cpFraction={cpf:.2f}: MSE={mse3:.2e} {status}")
