"""
SC-FDMA Demodulator for LTE Uplink
===================================

Reverses SC-FDMA modulation to recover resource grid from time-domain waveform.
Companion to lte_scfdma_modulator.py

Author: CLAUDE-LTE Project
Date: 2025-11-20
"""

import numpy as np
import pyfftw
from typing import Dict, Union, Tuple
from dataclasses import dataclass

# Enable PyFFTW optimizations
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

# Zero threshold for signed zero handling
ZERO_THRESHOLD = 1e-15


def apply_zero_threshold(arr: np.ndarray) -> np.ndarray:
    """Apply zero threshold to avoid signed zero issues"""
    arr = arr.copy()
    arr.real[np.abs(arr.real) < ZERO_THRESHOLD] = 0.0
    arr.imag[np.abs(arr.imag) < ZERO_THRESHOLD] = 0.0
    return arr


@dataclass
class UEConfig:
    """User Equipment Configuration"""
    NULRB: int = None
    CyclicPrefixUL: str = 'Normal'
    NBULSubcarrierSpacing: str = None
    FrameStructureType: int = 1


class LTESCFDMADemodulator:
    """
    LTE SC-FDMA Demodulator

    Reverses the SC-FDMA modulation process:
    1. Remove cyclic prefix
    2. Remove half-subcarrier shift
    3. FFT
    4. Extract resource grid
    """

    # Physical constants (match modulator)
    DELTA_F = 15000  # Hz
    N_FFT_BASE = 2048
    BASE_SAMPLING_RATE = 30.72e6  # 30.72 MHz
    NB_IOT_SAMPLING_RATE = 1.92e6  # 1.92 MHz for NB-IoT

    # NFFT mapping
    NFFT_MAP = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 2048, 100: 2048}

    def __init__(self):
        """Initialize demodulator with FFT plans"""
        self._fft_plans = {}

    def _get_fft_plan(self, nfft: int):
        """Get or create cached FFT plan"""
        if nfft not in self._fft_plans:
            time_array = pyfftw.empty_aligned(nfft, dtype='complex128')
            freq_array = pyfftw.empty_aligned(nfft, dtype='complex128')

            fft_plan = pyfftw.FFTW(
                time_array,
                freq_array,
                direction='FFTW_FORWARD',
                flags=('FFTW_MEASURE',),
                threads=pyfftw.config.NUM_THREADS
            )

            self._fft_plans[nfft] = {
                'plan': fft_plan,
                'time_array': time_array,
                'freq_array': freq_array
            }

        return self._fft_plans[nfft]

    def _determine_nfft(self, ue: UEConfig, M: int) -> int:
        """Determine NFFT size (match modulator logic)"""
        if ue.NBULSubcarrierSpacing is not None:
            if ue.NBULSubcarrierSpacing == '15kHz':
                return 128
            elif ue.NBULSubcarrierSpacing == '3.75kHz':
                return 512

        if M == 12:
            return 128

        if M >= 72:
            NRB = M // 12
            if NRB in self.NFFT_MAP:
                return self.NFFT_MAP[NRB]
            else:
                min_fft = 12 * NRB / 0.85
                nfft = int(2 ** np.ceil(np.log2(min_fft)))
                return nfft

        return 128

    def _get_cyclic_prefix_lengths(self, nfft: int, cp_type: str,
                                   nb_iot_spacing: str = None) -> np.ndarray:
        """Get cyclic prefix lengths (match modulator)"""
        if nb_iot_spacing == '3.75kHz':
            return np.array([16] * 14, dtype=np.int32)

        if nb_iot_spacing == '15kHz' or nfft == 128:
            if cp_type == 'Normal':
                return np.array([10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9], dtype=np.int32)
            else:
                return np.array([32] * 12, dtype=np.int32)

        scale_factor = nfft / 2048.0

        if cp_type == 'Normal':
            base_cp = np.array([160, 144, 144, 144, 144, 144, 144,
                               160, 144, 144, 144, 144, 144, 144], dtype=np.float64)
        else:
            base_cp = np.array([512] * 12, dtype=np.float64)

        cp_lengths = np.round(base_cp * scale_factor).astype(np.int32)
        return cp_lengths

    def _remove_half_subcarrier_shift(self, time_with_cp: np.ndarray,
                                     n_cp: int, nfft: int) -> np.ndarray:
        """
        Remove half-subcarrier shift: exp(-j*2*pi*0.5*Δf*t)

        This reverses the SC-FDMA phase shift applied in modulation
        """
        # Remove the phase shift that was applied in modulation
        t_indices = np.arange(-n_cp, nfft)
        phase_shift = np.exp(-1j * np.pi * t_indices / nfft)  # Negative to reverse
        unshifted_signal = time_with_cp * phase_shift

        # Remove CP (keep only IFFT portion)
        time_signal = unshifted_signal[n_cp:]

        return time_signal

    def _extract_from_fft_output(self, fft_output: np.ndarray, M: int) -> np.ndarray:
        """
        Extract resource grid from FFT output

        Reverses the mapping from _map_to_fft_input in modulator
        """
        nfft = len(fft_output)

        # CORRECTED: Extract contiguous block centered around DC
        # This matches the modulator's mapping
        firstSC = (nfft // 2) - (M // 2)

        # Extract grid from fft_output[firstSC:firstSC+M]
        grid_symbol = fft_output[firstSC:firstSC+M]

        return grid_symbol

    def demodulate(self, ue: Union[UEConfig, Dict],
                  waveform: np.ndarray) -> np.ndarray:
        """
        Main SC-FDMA demodulation function

        Args:
            ue: UE configuration
            waveform: Time-domain waveform (T x P)

        Returns:
            grid: Resource grid (M x N x P)
        """
        # Convert dict to dataclass if needed
        if isinstance(ue, dict):
            ue = UEConfig(**{k: v for k, v in ue.items()
                            if k in ['NULRB', 'CyclicPrefixUL',
                                   'NBULSubcarrierSpacing', 'FrameStructureType']})

        # Handle 1D waveform
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)

        T, P = waveform.shape

        # Determine M (number of subcarriers in resource grid)
        if ue.NBULSubcarrierSpacing == '15kHz':
            M = 12
        elif ue.NBULSubcarrierSpacing == '3.75kHz':
            M = 48
        elif ue.NULRB is not None:
            M = ue.NULRB * 12
        else:
            raise ValueError("Cannot determine M: specify NULRB or NBULSubcarrierSpacing")

        # Determine NFFT
        nfft = self._determine_nfft(ue, M)

        # Get cyclic prefix lengths
        cp_lengths = self._get_cyclic_prefix_lengths(
            nfft, ue.CyclicPrefixUL, ue.NBULSubcarrierSpacing)

        # Calculate number of symbols
        symbol_length = nfft + cp_lengths[0]  # Approximate
        N = len(cp_lengths)  # Symbols per slot/subframe

        # Get FFT plan
        fft_plan_dict = self._get_fft_plan(nfft)
        fft_plan = fft_plan_dict['plan']
        time_array = fft_plan_dict['time_array']
        freq_array = fft_plan_dict['freq_array']

        # Initialize output grid
        grid = np.zeros((M, N, P), dtype=np.complex128)

        # Process each antenna port
        for p in range(P):
            position = 0

            # Process each symbol
            for n in range(N):
                cp_idx = n % len(cp_lengths)
                n_cp = cp_lengths[cp_idx]

                # Extract symbol from waveform (CP + IFFT samples)
                symbol_len = n_cp + nfft

                if position + symbol_len > T:
                    # Not enough samples, pad with zeros
                    time_with_cp = np.zeros(symbol_len, dtype=np.complex128)
                    available = T - position
                    if available > 0:
                        time_with_cp[:available] = waveform[position:position+available, p]
                else:
                    time_with_cp = waveform[position:position+symbol_len, p]

                position += symbol_len

                # Remove half-subcarrier shift and CP
                time_signal = self._remove_half_subcarrier_shift(
                    time_with_cp, n_cp, nfft)

                # Apply zero threshold
                time_signal = apply_zero_threshold(time_signal)

                # FFT using PyFFTW
                time_array[:] = time_signal
                fft_plan()
                freq_signal = freq_array.copy()

                # Apply fftshift to center DC
                freq_signal = np.fft.fftshift(freq_signal)

                # Apply zero threshold
                freq_signal = apply_zero_threshold(freq_signal)

                # Extract resource grid
                grid_symbol = self._extract_from_fft_output(freq_signal, M)

                grid[:, n, p] = grid_symbol

        return grid


def lteSCFDMADemodulate(ue: Union[UEConfig, Dict],
                       waveform: np.ndarray) -> np.ndarray:
    """
    Main function interface for SC-FDMA demodulation

    Syntax:
        grid = lteSCFDMADemodulate(ue, waveform)

    Args:
        ue: UE configuration (dict or UEConfig)
        waveform: Time-domain waveform (T x P)

    Returns:
        grid: Resource grid (M x N x P)
    """
    demodulator = LTESCFDMADemodulator()
    return demodulator.demodulate(ue, waveform)


if __name__ == '__main__':
    """Quick test"""
    print("SC-FDMA Demodulator - Quick Test")
    print("=" * 60)

    # Test round-trip: Grid → Modulate → Demodulate → Grid
    from lte_scfdma_modulator import lteSCFDMAModulate

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal'}

    # Create test grid
    M, N = 72, 14
    grid_original = np.random.randn(M, N, 1) + 1j * np.random.randn(M, N, 1)

    # Modulate
    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Original grid:     {grid_original.shape}")
    print(f"Waveform:          {waveform.shape}")

    # Demodulate
    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered grid:    {grid_recovered.shape}")

    # Check error
    error = np.mean(np.abs(grid_original - grid_recovered)**2)
    print(f"\nMSE: {error:.2e}")

    if error < 1e-20:
        print("✓ Perfect round-trip!")
    elif error < 1e-10:
        print("✓ Excellent round-trip!")
    else:
        print("⚠ Some error in round-trip")

    print("\n✓ Test complete!")
