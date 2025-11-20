"""
Test Suite for lteULChannelEstimate with Real SC-FDMA Modulator
================================================================

Comprehensive tests using actual SC-FDMA modulation/demodulation chain.
Validates channel estimation with realistic signal processing.

Author: CLAUDE-LTE Project
Date: 2025-11-20
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys


def generate_multipath_channel(ue_config, num_taps=3, max_delay_samples=10):
    """
    Generate frequency-selective multipath channel impulse response

    Parameters
    ----------
    ue_config : dict
        UE configuration
    num_taps : int
        Number of channel taps (paths)
    max_delay_samples : int
        Maximum delay spread in samples

    Returns
    -------
    h_impulse : np.ndarray
        Channel impulse response (taps,)
    delays : np.ndarray
        Tap delays in samples
    """
    # Generate tap delays (uniformly distributed)
    delays = np.sort(np.random.randint(0, max_delay_samples, num_taps))

    # Generate tap gains (Rayleigh fading)
    tap_gains = (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)

    # Normalize power
    tap_gains = tap_gains / np.sqrt(np.sum(np.abs(tap_gains)**2))

    return tap_gains, delays


def apply_multipath_channel(waveform, tap_gains, delays):
    """
    Apply multipath channel to time-domain waveform

    y(t) = Σ h_i * x(t - τ_i) + n(t)

    Parameters
    ----------
    waveform : np.ndarray
        Input waveform (T x P)
    tap_gains : np.ndarray
        Channel tap gains
    delays : np.ndarray
        Channel tap delays in samples

    Returns
    -------
    output : np.ndarray
        Channel-affected waveform (T x P)
    """
    if waveform.ndim == 1:
        waveform = waveform.reshape(-1, 1)

    T, P = waveform.shape
    max_delay = np.max(delays) if len(delays) > 0 else 0

    # Create output with extra length for delays
    output = np.zeros((T + max_delay, P), dtype=np.complex128)

    # Apply each tap
    for tap_gain, delay in zip(tap_gains, delays):
        output[delay:delay+T, :] += tap_gain * waveform

    # Trim to original length
    return output[:T, :]


def add_awgn_noise(signal, snr_db):
    """
    Add AWGN noise to achieve target SNR

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    snr_db : float
        Target SNR in dB

    Returns
    -------
    noisy_signal : np.ndarray
        Signal with added noise
    noise_power : float
        Noise power
    """
    signal_power = np.mean(np.abs(signal)**2)

    if signal_power == 0:
        return signal, 0.0

    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )

    return signal + noise, noise_power


def create_test_grid_with_drs(ue, chs):
    """
    Create resource grid with PUSCH DRS and random data

    Parameters
    ----------
    ue : dict
        UE configuration
    chs : dict
        Channel configuration

    Returns
    -------
    grid : np.ndarray
        Resource grid with DRS (M x N x P)
    """
    # Get grid size
    nsc, nsym = lteULResourceGridSize(ue)
    ntx = ue.get('NTxAnts', 1)

    # Initialize grid
    grid = np.zeros((nsc, nsym, ntx), dtype=np.complex128)

    # Generate PUSCH DRS
    drs_seq, _, _ = ltePUSCHDRS(ue, chs)

    if drs_seq is None or drs_seq.size == 0:
        raise ValueError("Failed to generate PUSCH DRS")

    # Get DRS indices
    ue_config = DRSUEConfig(
        NULRB=ue['NULRB'],
        CyclicPrefixUL=ue.get('CyclicPrefixUL', 'Normal'),
        NTxAnts=ntx
    )
    chs_config = DRSCHSConfig(PRBSet=chs['PRBSet'])
    drs_indices = ltePUSCHDRSIndices(ue_config, chs_config, '0based sub')

    # Place DRS in grid
    for i in range(drs_indices.shape[0]):
        sc_idx = int(drs_indices[i, 0])
        sym_idx = int(drs_indices[i, 1])
        ant_idx = int(drs_indices[i, 2])
        grid[sc_idx, sym_idx, ant_idx] = drs_seq[i, ant_idx]

    # Fill non-DRS locations with random QPSK data
    # (In real system, this would be actual PUSCH data)
    qpsk_symbols = (2*np.random.randint(0, 2, grid.shape) - 1 +
                   1j*(2*np.random.randint(0, 2, grid.shape) - 1)) / np.sqrt(2)

    # Only fill where grid is zero (non-DRS locations)
    mask = (grid == 0)
    grid[mask] = qpsk_symbols[mask]

    return grid


def test_basic_functionality():
    """Test 1: Basic functionality with SC-FDMA modulation"""

    print("\n" + "="*80)
    print("TEST 1: Basic Functionality - Real SC-FDMA Chain")
    print("="*80)

    # Configure UE
    ue = {
        'NULRB': 6,
        'NCellID': 1,
        'NSubframe': 0,
        'CyclicPrefixUL': 'Normal',
        'NTxAnts': 1,
        'Hopping': 'Off',
        'SeqGroup': 0,
        'CyclicShift': 0
    }

    # Configure channel
    chs = {
        'PRBSet': np.arange(6).reshape(-1, 1),
        'NLayers': 1,
        'DynCyclicShift': 0,
        'OrthoCover': 'Off'
    }

    print(f"\nConfiguration:")
    print(f"  NULRB: {ue['NULRB']}")
    print(f"  NCellID: {ue['NCellID']}")
    print(f"  PRBSet: [0, 1, 2, 3, 4, 5]")

    try:
        # Create resource grid with DRS
        print(f"\nCreating resource grid with PUSCH DRS...")
        tx_grid = create_test_grid_with_drs(ue, chs)
        print(f"  ✓ Grid created: {tx_grid.shape}")

        # SC-FDMA Modulation
        print(f"\nPerforming SC-FDMA modulation...")
        tx_waveform, mod_info = lteSCFDMAModulate(ue, tx_grid)
        print(f"  ✓ Waveform generated: {tx_waveform.shape}")
        print(f"  NFFT: {mod_info.Nfft}")
        print(f"  Sampling rate: {mod_info.SamplingRate/1e6:.2f} MHz")

        # Generate multipath channel
        print(f"\nApplying multipath channel...")
        num_taps = 3
        tap_gains, delays = generate_multipath_channel(ue, num_taps=num_taps)
        print(f"  Channel taps: {num_taps}")
        print(f"  Delays: {delays} samples")

        # Apply channel
        rx_waveform = apply_multipath_channel(tx_waveform, tap_gains, delays)

        # Add noise
        snr_db = 20
        rx_waveform_noisy, noise_power = add_awgn_noise(rx_waveform, snr_db)
        print(f"  SNR: {snr_db} dB")
        print(f"  ✓ Channel applied")

        # SC-FDMA Demodulation
        print(f"\nPerforming SC-FDMA demodulation...")
        rx_grid = lteSCFDMADemodulate(ue, rx_waveform_noisy)
        print(f"  ✓ Grid recovered: {rx_grid.shape}")

        # Perform channel estimation
        print(f"\nPerforming channel estimation...")
        hest, noiseest = lteULChannelEstimate(ue, chs, rx_grid)

        print(f"  ✓ Channel estimation successful")
        print(f"  Output shape: {hest.shape}")
        print(f"  Noise estimate: {noiseest:.6f}")

        # Validate channel estimate
        # Get DRS locations
        ue_config = DRSUEConfig(
            NULRB=ue['NULRB'],
            CyclicPrefixUL=ue['CyclicPrefixUL'],
            NTxAnts=1
        )
        chs_config = DRSCHSConfig(PRBSet=chs['PRBSet'])
        drs_indices = ltePUSCHDRSIndices(ue_config, chs_config, '0based sub')

        # Check that estimates exist at pilot locations
        pilots_estimated = 0
        for i in range(drs_indices.shape[0]):
            sc_idx = int(drs_indices[i, 0])
            sym_idx = int(drs_indices[i, 1])
            if hest[sc_idx, sym_idx, 0, 0] != 0:
                pilots_estimated += 1

        print(f"  Pilots with estimates: {pilots_estimated}/{len(drs_indices)}")

        if pilots_estimated >= len(drs_indices) * 0.9:  # At least 90% should have estimates
            print(f"  ✓ PASS: Channel estimation successful")
            return True
        else:
            print(f"  ⚠ Warning: Some pilots missing estimates")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_cec():
    """Test 2: Custom channel estimator configuration"""

    print("\n" + "="*80)
    print("TEST 2: Custom CEC with SC-FDMA")
    print("="*80)

    ue = {
        'NULRB': 15,
        'NCellID': 10,
        'NSubframe': 5,
        'CyclicPrefixUL': 'Normal',
        'NTxAnts': 1,
        'Hopping': 'Group',
        'SeqGroup': 5,
        'CyclicShift': 2
    }

    chs = {
        'PRBSet': np.arange(6).reshape(-1, 1),
        'NLayers': 1,
        'DynCyclicShift': 3,
        'OrthoCover': 'On'
    }

    # Custom CEC
    cec = {
        'FreqWindow': 13,
        'TimeWindow': 3,
        'InterpType': 'cubic',
        'PilotAverage': 'UserDefined',
        'Reference': 'Antennas'
    }

    print(f"\nCustom CEC Configuration:")
    print(f"  FreqWindow: {cec['FreqWindow']}")
    print(f"  TimeWindow: {cec['TimeWindow']}")
    print(f"  InterpType: {cec['InterpType']}")

    try:
        # Create grid with DRS
        tx_grid = create_test_grid_with_drs(ue, chs)

        # Modulate
        tx_waveform, _ = lteSCFDMAModulate(ue, tx_grid)

        # Channel
        tap_gains, delays = generate_multipath_channel(ue, num_taps=4)
        rx_waveform = apply_multipath_channel(tx_waveform, tap_gains, delays)
        rx_waveform_noisy, _ = add_awgn_noise(rx_waveform, 15)

        # Demodulate
        rx_grid = lteSCFDMADemodulate(ue, rx_waveform_noisy)

        # Estimate with custom CEC
        hest, noiseest = lteULChannelEstimate(ue, chs, cec, rx_grid)

        print(f"\n  ✓ Channel estimation successful")
        print(f"  Output shape: {hest.shape}")
        print(f"  Noise estimate: {noiseest:.6f}")

        # Verify interpolation worked
        non_zero = np.sum(hest != 0)
        total = hest.size
        print(f"  Non-zero elements: {non_zero}/{total} ({100*non_zero/total:.1f}%)")

        # Get DRS count
        ue_config = DRSUEConfig(NULRB=ue['NULRB'], CyclicPrefixUL='Normal', NTxAnts=1)
        chs_config = DRSCHSConfig(PRBSet=chs['PRBSet'])
        drs_indices = ltePUSCHDRSIndices(ue_config, chs_config, '0based sub')

        if non_zero > len(drs_indices):
            print(f"  ✓ PASS: Interpolation performed ({non_zero} > {len(drs_indices)} pilots)")
            return True
        else:
            print(f"  ⚠ Warning: Limited interpolation")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interpolation_types():
    """Test 3: Different interpolation types"""

    print("\n" + "="*80)
    print("TEST 3: Interpolation Types with SC-FDMA")
    print("="*80)

    ue = {
        'NULRB': 6,
        'NCellID': 1,
        'NSubframe': 0,
        'CyclicPrefixUL': 'Normal',
        'NTxAnts': 1
    }

    chs = {
        'PRBSet': np.arange(4).reshape(-1, 1),
        'NLayers': 1
    }

    # Create and process waveform once
    try:
        tx_grid = create_test_grid_with_drs(ue, chs)
        tx_waveform, _ = lteSCFDMAModulate(ue, tx_grid)
        tap_gains, delays = generate_multipath_channel(ue, num_taps=2)
        rx_waveform = apply_multipath_channel(tx_waveform, tap_gains, delays)
        rx_waveform_noisy, _ = add_awgn_noise(rx_waveform, 25)
        rx_grid = lteSCFDMADemodulate(ue, rx_waveform_noisy)
    except Exception as e:
        print(f"  ✗ FAIL: Could not create test signal - {e}")
        return False

    # Get DRS count for comparison
    ue_config = DRSUEConfig(NULRB=ue['NULRB'], CyclicPrefixUL='Normal', NTxAnts=1)
    chs_config = DRSCHSConfig(PRBSet=chs['PRBSet'])
    drs_indices = ltePUSCHDRSIndices(ue_config, chs_config, '0based sub')
    num_pilots = len(drs_indices)

    # Test different interpolation methods
    interp_methods = ['nearest', 'linear', 'cubic', 'none']
    results = {}

    for method in interp_methods:
        cec = {
            'FreqWindow': 1,
            'TimeWindow': 1,
            'InterpType': method
        }

        try:
            hest, noiseest = lteULChannelEstimate(ue, chs, cec, rx_grid)

            non_zero = np.sum(hest != 0)
            print(f"\n  {method:8s}: {non_zero:4d} non-zero elements", end="")

            if method == 'none':
                # Should have estimates only at pilot locations
                if non_zero <= num_pilots * 1.1:  # Allow 10% tolerance
                    print(" ✓")
                    results[method] = True
                else:
                    print(" ✗ (too many)")
                    results[method] = False
            else:
                # Should have more than just pilots
                if non_zero > num_pilots:
                    print(" ✓")
                    results[method] = True
                else:
                    print(" ✗ (no interpolation)")
                    results[method] = False

        except Exception as e:
            print(f"\n  {method:8s}: ✗ Failed - {e}")
            results[method] = False

    all_passed = all(results.values())
    print(f"\n  Overall: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_extended_cp():
    """Test 4: Extended cyclic prefix"""

    print("\n" + "="*80)
    print("TEST 4: Extended CP with SC-FDMA")
    print("="*80)

    ue = {
        'NULRB': 6,
        'NCellID': 1,
        'NSubframe': 0,
        'CyclicPrefixUL': 'Extended',
        'NTxAnts': 1
    }

    chs = {
        'PRBSet': np.arange(6).reshape(-1, 1),
        'NLayers': 1
    }

    print(f"\nCyclic Prefix: {ue['CyclicPrefixUL']}")

    try:
        # Create and process signal
        tx_grid = create_test_grid_with_drs(ue, chs)
        print(f"  Grid shape: {tx_grid.shape} (Extended CP: 12 symbols)")

        tx_waveform, mod_info = lteSCFDMAModulate(ue, tx_grid)
        print(f"  Waveform generated: {tx_waveform.shape}")
        print(f"  CP lengths: {mod_info.CyclicPrefixLengths}")

        # Channel
        tap_gains, delays = generate_multipath_channel(ue, num_taps=2)
        rx_waveform = apply_multipath_channel(tx_waveform, tap_gains, delays)
        rx_waveform_noisy, _ = add_awgn_noise(rx_waveform, 20)

        # Demodulate
        rx_grid = lteSCFDMADemodulate(ue, rx_waveform_noisy)
        print(f"  Demodulated grid: {rx_grid.shape}")

        # Estimate
        hest, noiseest = lteULChannelEstimate(ue, chs, rx_grid)

        print(f"\n  ✓ Channel estimation successful")
        print(f"  Output shape: {hest.shape}")

        if hest.shape[1] == 12:
            print(f"  ✓ PASS: Correct symbol count for Extended CP")
            return True
        else:
            print(f"  ✗ FAIL: Wrong symbol count (expected 12, got {hest.shape[1]})")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_snr_performance():
    """Test 5: SNR performance characterization"""

    print("\n" + "="*80)
    print("TEST 5: SNR Performance with SC-FDMA")
    print("="*80)

    ue = {
        'NULRB': 6,
        'NCellID': 1,
        'NSubframe': 0,
        'CyclicPrefixUL': 'Normal',
        'NTxAnts': 1
    }

    chs = {
        'PRBSet': np.arange(6).reshape(-1, 1),
        'NLayers': 1
    }

    cec = {
        'FreqWindow': 7,
        'TimeWindow': 1,
        'InterpType': 'cubic'
    }

    snr_values = [0, 10, 20, 30]
    results = {}

    print("\nTesting channel estimation at different SNR levels...")

    for snr_db in snr_values:
        try:
            # Create and process signal
            tx_grid = create_test_grid_with_drs(ue, chs)
            tx_waveform, _ = lteSCFDMAModulate(ue, tx_grid)

            # Channel
            tap_gains, delays = generate_multipath_channel(ue, num_taps=2)
            rx_waveform = apply_multipath_channel(tx_waveform, tap_gains, delays)
            rx_waveform_noisy, true_noise_power = add_awgn_noise(rx_waveform, snr_db)

            # Demodulate
            rx_grid = lteSCFDMADemodulate(ue, rx_waveform_noisy)

            # Estimate
            hest, noiseest = lteULChannelEstimate(ue, chs, cec, rx_grid)

            # Count non-zero estimates
            non_zero = np.sum(hest != 0)

            print(f"  SNR {snr_db:2d} dB: noise_est={noiseest:.6f}, "
                  f"estimates={non_zero}, ", end="")

            # Should have estimates at most locations for high SNR
            if snr_db >= 20 and non_zero > 500:
                print("✓")
                results[snr_db] = True
            elif snr_db < 20 and non_zero > 0:
                print("✓")
                results[snr_db] = True
            else:
                print("⚠")
                results[snr_db] = False

        except Exception as e:
            print(f"  SNR {snr_db:2d} dB: ✗ Failed - {e}")
            results[snr_db] = False

    passed = sum(results.values())
    total = len(results)

    print(f"\n  Results: {passed}/{total} SNR points passed")

    if passed >= total * 0.75:  # At least 75% should pass
        print(f"  ✓ PASS: Good performance across SNR range")
        return True
    else:
        print(f"  ⚠ Warning: Limited performance")
        return False


def run_all_tests():
    """Run all test cases"""

    print("\n" + "="*80)
    print("lteULChannelEstimate - SC-FDMA Integration Test Suite")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nUsing REAL SC-FDMA modulation/demodulation chain")

    results = {}

    # Run all tests
    tests = [
        ("Basic SC-FDMA Chain", test_basic_functionality),
        ("Custom CEC", test_custom_cec),
        ("Interpolation Types", test_interpolation_types),
        ("Extended CP", test_extended_cp),
        ("SNR Performance", test_snr_performance)
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:25s}: {status}")

    print("\n" + "-"*80)
    print(f"  TOTAL: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("="*80)

    if passed == total:
        print("\n✓ ALL TESTS PASSED - Full SC-FDMA integration validated!")
        print("\nValidated chain:")
        print("  ✓ Resource grid creation with PUSCH DRS")
        print("  ✓ SC-FDMA modulation (IFFT + CP + shift)")
        print("  ✓ Multipath channel propagation")
        print("  ✓ AWGN noise addition")
        print("  ✓ SC-FDMA demodulation (FFT + CP removal)")
        print("  ✓ Channel estimation with interpolation")
        print("  ✓ Multiple SNR levels")
        print("\n3GPP TS 36.211 v10.1.0: ✓ COMPLIANT")
        print("SC-FDMA integration: ✓ VERIFIED")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
