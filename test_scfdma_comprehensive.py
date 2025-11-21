"""
Comprehensive SC-FDMA Modulator/Demodulator Test Suite
=======================================================

Tests for MATLAB-exact SC-FDMA implementation including:
- Basic round-trip without windowing
- Round-trip with default windowing
- Different cpFraction values
- Multiple CP configurations
- Extended CP support
"""
import numpy as np
from lte_scfdma_modulate import lteSCFDMAModulate
from lte_scfdma_demodulate import lteSCFDMADemodulate


def test_basic_no_windowing():
    """Test 1: Basic round-trip without windowing"""
    print("="*60)
    print("TEST 1: Basic round-trip without windowing")
    print("="*60)

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal', 'Windowing': 0}
    grid_original = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Modulator info: NFFT={info.Nfft}, Windowing={info.Windowing}, "
          f"SamplingRate={info.SamplingRate/1e6:.2f}MHz")
    print(f"Waveform shape: {waveform.shape}")

    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered shape: {grid_recovered.shape}")

    mse = np.mean(np.abs(grid_original - grid_recovered)**2)
    print(f"MSE: {mse:.2e}")

    if mse < 1e-15:
        print("✓ Excellent round-trip!")
        return True
    else:
        print("✗ Round-trip has errors")
        return False


def test_with_default_windowing():
    """Test 2: Round-trip with default windowing"""
    print("\n" + "="*60)
    print("TEST 2: Round-trip with default windowing")
    print("="*60)

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal'}
    grid_original = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Modulator info: NFFT={info.Nfft}, Windowing={info.Windowing}, "
          f"SamplingRate={info.SamplingRate/1e6:.2f}MHz")
    print(f"Waveform shape: {waveform.shape}")

    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered shape: {grid_recovered.shape}")

    mse = np.mean(np.abs(grid_original - grid_recovered)**2)
    print(f"MSE: {mse:.2e}")

    if mse < 1e-15:
        print("✓ Excellent round-trip!")
        return True
    else:
        print("✗ Round-trip has errors")
        return False


def test_different_cp_fractions():
    """Test 3: Try different cpFraction values"""
    print("\n" + "="*60)
    print("TEST 3: Different cpFraction values")
    print("="*60)

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal'}
    grid_original = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Testing with Windowing={info.Windowing}")

    best_cpf = None
    best_mse = float('inf')

    for cpf in [0.0, 0.25, 0.5, 0.55, 0.75, 1.0]:
        grid_recovered = lteSCFDMADemodulate(ue, waveform, cpf)
        mse = np.mean(np.abs(grid_original - grid_recovered)**2)
        status = "✓" if mse < 1e-10 else "✗"
        print(f"  cpFraction={cpf:.2f}: MSE={mse:.2e} {status}")

        if mse < best_mse:
            best_mse = mse
            best_cpf = cpf

    print(f"\nBest cpFraction: {best_cpf} with MSE={best_mse:.2e}")
    return best_mse < 1e-15


def test_extended_cp():
    """Test 4: Extended CP configuration"""
    print("\n" + "="*60)
    print("TEST 4: Extended CP configuration")
    print("="*60)

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Extended', 'Windowing': 0}
    grid_original = np.random.randn(72, 12, 1) + 1j * np.random.randn(72, 12, 1)

    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Modulator info: NFFT={info.Nfft}, CP={ue['CyclicPrefixUL']}, "
          f"Symbols={grid_original.shape[1]}")
    print(f"Waveform shape: {waveform.shape}")

    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered shape: {grid_recovered.shape}")

    mse = np.mean(np.abs(grid_original - grid_recovered)**2)
    print(f"MSE: {mse:.2e}")

    if mse < 1e-15:
        print("✓ Excellent round-trip!")
        return True
    else:
        print("✗ Round-trip has errors")
        return False


def test_different_bandwidth():
    """Test 5: Different bandwidth (25 RBs)"""
    print("\n" + "="*60)
    print("TEST 5: Different bandwidth (25 RBs)")
    print("="*60)

    ue = {'NULRB': 25, 'CyclicPrefixUL': 'Normal', 'Windowing': 0}
    grid_original = np.random.randn(300, 14, 1) + 1j * np.random.randn(300, 14, 1)

    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Modulator info: NFFT={info.Nfft}, NULRB={ue['NULRB']}")
    print(f"Waveform shape: {waveform.shape}")

    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered shape: {grid_recovered.shape}")

    mse = np.mean(np.abs(grid_original - grid_recovered)**2)
    print(f"MSE: {mse:.2e}")

    if mse < 1e-15:
        print("✓ Excellent round-trip!")
        return True
    else:
        print("✗ Round-trip has errors")
        return False


def test_minimal_manual():
    """Test 6: Minimal manual step-by-step test"""
    print("\n" + "="*60)
    print("TEST 6: Minimal manual SC-FDMA (single symbol)")
    print("="*60)

    # Test parameters
    nFFT = 128
    nSC = 72
    cpLength = 10

    # Simple input: all ones
    grid_in = np.ones(nSC, dtype=np.complex128)

    print(f"Parameters: nFFT={nFFT}, nSC={nSC}, cpLength={cpLength}")

    # MODULATE
    # SC-FDMA: DFT spreading
    dftOut = np.fft.fft(grid_in, nSC)

    freq_array = np.zeros(nFFT, dtype=np.complex128)
    firstSC = nFFT//2 - nSC//2
    freq_array[firstSC:firstSC+nSC] = dftOut

    freq_shifted = np.fft.fftshift(freq_array)
    iffout = np.fft.ifft(freq_shifted)

    # Add CP with half-SC shift
    extended = np.concatenate([iffout[-cpLength:], iffout])
    phase_indices = np.arange(-cpLength, nFFT)
    phase_shift = np.exp(1j * np.pi * phase_indices / nFFT)
    waveform = extended * phase_shift

    print(f"Waveform length: {len(waveform)}")

    # DEMODULATE
    cpFraction = 0.55
    fftStart = int(cpLength * cpFraction)

    samples = waveform[fftStart:fftStart+nFFT]

    idx = np.arange(nFFT)
    halfsc = np.exp(1j * np.pi / nFFT * (idx + fftStart - cpLength))
    samples = samples * halfsc

    fftout = np.fft.fft(samples)

    phaseCorrection = np.exp(-1j * 2 * np.pi * (cpLength - fftStart) / nFFT * idx)
    fftout = fftout * phaseCorrection

    fftout = np.fft.fftshift(fftout)

    firstActiveSC = nFFT//2 - nSC//2
    activeSCs = fftout[firstActiveSC:firstActiveSC+nSC]

    # SC-FDMA: IFFT despreading
    grid_out = np.fft.ifft(activeSCs, nSC)

    # Compare
    mse = np.mean(np.abs(grid_in - grid_out)**2)
    print(f"MSE: {mse:.2e}")

    if mse < 1e-15:
        print("✓ Manual test passed!")
        return True
    else:
        print("✗ Manual test failed")
        return False


def test_multiple_antennas():
    """Test 7: Multiple antenna ports"""
    print("\n" + "="*60)
    print("TEST 7: Multiple antenna ports (2 antennas)")
    print("="*60)

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal', 'Windowing': 0}
    grid_original = np.random.randn(72, 14, 2) + 1j * np.random.randn(72, 14, 2)

    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Input grid shape: {grid_original.shape}")
    print(f"Waveform shape: {waveform.shape}")

    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered shape: {grid_recovered.shape}")

    mse = np.mean(np.abs(grid_original - grid_recovered)**2)
    print(f"MSE: {mse:.2e}")

    if mse < 1e-15:
        print("✓ Multiple antennas work!")
        return True
    else:
        print("✗ Multiple antennas have errors")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("COMPREHENSIVE SC-FDMA TEST SUITE")
    print("="*60 + "\n")

    tests = [
        ("Basic (no windowing)", test_basic_no_windowing),
        ("Default windowing", test_with_default_windowing),
        ("CP fraction sweep", test_different_cp_fractions),
        ("Extended CP", test_extended_cp),
        ("25 RBs bandwidth", test_different_bandwidth),
        ("Manual step-by-step", test_minimal_manual),
        ("Multiple antennas", test_multiple_antennas),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")

    return passed_count == total_count


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
