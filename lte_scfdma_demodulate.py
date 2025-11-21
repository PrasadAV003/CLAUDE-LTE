"""
SC-FDMA Demodulator - MATLAB-Exact Implementation
==================================================

Complete Python implementation matching MATLAB lteSCFDMADemodulate exactly.
Based on 3GPP TS 36.211 and MATLAB R2022b implementation.

Key Features:
- CP fraction positioning (default 0.55 for LTE)
- Exact phase correction formulas from MATLAB
- Half-subcarrier shift with correct phase
- FFT then fftshift (opposite of modulator)
- Active subcarrier extraction

Author: CLAUDE-LTE Project
Date: 2025-11-20
"""

import numpy as np
import pyfftw
from typing import Dict, Tuple, Optional

# Enable PyFFTW optimizations
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

ZERO_THRESHOLD = 1e-15


def apply_zero_threshold(arr: np.ndarray) -> np.ndarray:
    """Apply zero threshold"""
    arr = arr.copy()
    arr.real[np.abs(arr.real) < ZERO_THRESHOLD] = 0.0
    arr.imag[np.abs(arr.imag) < ZERO_THRESHOLD] = 0.0
    return arr


class LTESCFDMADemodulator:
    """
    LTE SC-FDMA Demodulator - MATLAB-Exact Implementation

    Matches MATLAB lteSCFDMADemodulate behavior exactly:
    - CP fraction positioning (0.55 default for LTE)
    - Phase correction: exp(-1j*2*pi*(cpLength-fftStart)/nFFT*idx)
    - Half-SC shift: exp(1j*pi/nFFT*(idx+fftStart-cpLength))
    - FFT then fftshift
    - Extract active subcarriers
    """

    # Constants
    NFFT_MAP = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 2048, 100: 2048}
    BASE_SAMPLING_RATE = 30.72e6
    N_FFT_BASE = 2048

    def __init__(self):
        """Initialize demodulator with FFT plans"""
        self._fft_plans = {}

    def _get_fft_plan(self, nfft: int):
        """Get or create cached FFT plan"""
        if nfft not in self._fft_plans:
            time_array = pyfftw.empty_aligned(nfft, dtype='complex128')
            freq_array = pyfftw.empty_aligned(nfft, dtype='complex128')

            fft_plan = pyfftw.FFTW(
                time_array, freq_array,
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

    def _determine_nfft(self, nsc: int, ue: Dict) -> int:
        """Determine NFFT size (match modulator logic)"""
        if ue.get('NBULSubcarrierSpacing') == '15kHz':
            return 128
        elif ue.get('NBULSubcarrierSpacing') == '3.75kHz':
            return 512

        log2nsc = np.log2(nsc)
        if log2nsc == int(log2nsc) and log2nsc > 6:
            nrb = int(0.85 * nsc / 12)
        else:
            nrb = int(nsc / 12)

        if nrb < 6:
            nrb = 6
        if nrb > 110:
            nrb = 110

        if nrb in self.NFFT_MAP:
            return self.NFFT_MAP[nrb]
        else:
            min_fft = 12 * nrb / 0.85
            return int(2 ** np.ceil(np.log2(min_fft)))

    def _get_cyclic_prefix_lengths(self, nfft: int, cp_type: str,
                                   nb_spacing: Optional[str] = None) -> np.ndarray:
        """Get CP lengths (match modulator)"""
        if nb_spacing == '3.75kHz':
            return np.array([16] * 14, dtype=np.int32)

        if nb_spacing == '15kHz' or nfft == 128:
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

        return np.round(base_cp * scale_factor).astype(np.int32)

    def _get_default_cp_fraction(self, ue: Dict) -> float:
        """
        Get default CP fraction matching MATLAB

        MATLAB defaults:
        - 0.55 for LTE
        - 0.22 for NB-IoT 15kHz
        - 0.18 for NB-IoT 3.75kHz
        """
        nb_spacing = ue.get('NBULSubcarrierSpacing')

        if nb_spacing == '15kHz':
            return 0.22
        elif nb_spacing == '3.75kHz':
            return 0.18
        else:
            return 0.55  # LTE default

    def demodulate(self, ue: Dict, waveform: np.ndarray,
                  chs: Optional[Dict] = None,
                  cpFraction: Optional[float] = None) -> np.ndarray:
        """
        SC-FDMA demodulation matching MATLAB lteSCFDMADemodulate

        MATLAB algorithm:
        1. Position FFT: fftStart = fix(cpLength * cpFraction)
        2. Phase correction: exp(-1j*2*pi*(cpLength-fftStart)/nFFT*idx)
        3. Half-SC shift: exp(1j*pi/nFFT*(idx+fftStart-cpLength))
        4. Extract samples: waveform(offset+fftStart+(1:nFFT),:)
        5. Apply half-SC shift: samples .* halfsc
        6. FFT: fft(...)
        7. fftshift
        8. Apply phase correction
        9. Extract active subcarriers

        Args:
            ue: UE configuration dict
            waveform: Time-domain waveform (T x P)
            chs: Optional channel config
            cpFraction: Optional CP fraction (0 to 1)

        Returns:
            grid: Resource grid (M x N x P)
        """
        # Handle 1D waveform
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)

        T, nAnts = waveform.shape

        # Get CP fraction
        if cpFraction is None:
            cpFraction = self._get_default_cp_fraction(ue)

        if not (0 <= cpFraction <= 1):
            raise ValueError("cpFraction must be between 0 and 1")

        # Determine parameters
        nulrb = ue.get('NULRB')
        nb_spacing = ue.get('NBULSubcarrierSpacing')

        if nb_spacing == '15kHz':
            totalActiveSC = 12
        elif nb_spacing == '3.75kHz':
            totalActiveSC = 48
        elif nulrb is not None:
            totalActiveSC = nulrb * 12
        else:
            raise ValueError("Cannot determine number of subcarriers")

        # Determine NFFT
        nFFT = self._determine_nfft(totalActiveSC, ue)

        # Get CP lengths
        cp_type = ue.get('CyclicPrefixUL', 'Normal')
        cpLengths = self._get_cyclic_prefix_lengths(nFFT, cp_type, nb_spacing)

        symbols_per_slot = len(cpLengths) // 2

        # Gap samples
        if nb_spacing == '3.75kHz':
            gapSamples = 144
        else:
            gapSamples = 0

        # Calculate samples per slot
        samplesPerSlot = int(np.sum(cpLengths[:symbols_per_slot]) +
                            nFFT * symbols_per_slot + gapSamples)
        nSlots = T // samplesPerSlot

        if nSlots == 0:
            raise ValueError("Waveform too short for even one slot")

        # MATLAB: firstActiveSC = (nFFT/2) - floor(totalActiveSC/2) + 1 (1-indexed)
        # Python (0-indexed): firstActiveSC = nFFT//2 - totalActiveSC//2
        firstActiveSC = (nFFT // 2) - (totalActiveSC // 2)

        # Create output grid
        dims = (totalActiveSC, symbols_per_slot * nSlots, nAnts)
        reGrid = np.zeros(dims, dtype=np.complex128)

        # Pre-calculate phase vectors
        # MATLAB: idx = 0:nFFT-1
        idx = np.arange(nFFT)

        # Get FFT plan
        fft_dict = self._get_fft_plan(nFFT)
        fft_plan = fft_dict['plan']
        time_array = fft_dict['time_array']
        freq_array = fft_dict['freq_array']

        # Process each antenna
        for ant in range(nAnts):
            offset = 0
            symbol_idx = 0

            # For each slot
            for slot in range(nSlots):
                # For each symbol in slot
                for sym_in_slot in range(symbols_per_slot):
                    # Get CP length for this symbol
                    cp_idx = sym_in_slot + slot * symbols_per_slot
                    cpLength = cpLengths[cp_idx % len(cpLengths)]

                    # MATLAB: fftStart = fix(cpLength * cpFraction)
                    fftStart = int(cpLength * cpFraction)

                    # MATLAB: phaseCorrection = exp(-1i*2*pi*(cpLength-fftStart)/nFFT*idx)'
                    phaseCorrection = np.exp(-1j * 2 * np.pi * (cpLength - fftStart) / nFFT * idx)

                    # MATLAB: halfsc = exp(1i*pi/nFFT*(idx+fftStart-cpLength))'
                    halfsc = np.exp(1j * np.pi / nFFT * (idx + fftStart - cpLength))

                    # Extract samples
                    # MATLAB: waveform(offset+fftStart+(1:nFFT),:)
                    # Python: waveform[offset+fftStart:offset+fftStart+nFFT, ant]
                    start = offset + fftStart
                    end = start + nFFT

                    if end > T:
                        # Not enough samples, pad with zeros
                        samples = np.zeros(nFFT, dtype=np.complex128)
                        available = T - start
                        if available > 0:
                            samples[:available] = waveform[start:start+available, ant]
                    else:
                        samples = waveform[start:end, ant]

                    # Apply half-subcarrier shift
                    # MATLAB: samples .* halfsc
                    samples = samples * halfsc

                    # FFT
                    time_array[:] = samples
                    fft_plan()
                    fftOutput = freq_array.copy()

                    # Apply phase correction BEFORE fftshift
                    fftOutput = fftOutput * phaseCorrection

                    # fftshift
                    # MATLAB: fftshift(...,1)
                    fftOutput = np.fft.fftshift(fftOutput)

                    # Apply zero threshold
                    fftOutput = apply_zero_threshold(fftOutput)

                    # Extract active subcarriers
                    # MATLAB: fftOutput(firstActiveSC:firstActiveSC+totalActiveSC-1,:)
                    activeSCs = fftOutput[firstActiveSC:firstActiveSC+totalActiveSC]

                    # Assign to grid
                    reGrid[:, symbol_idx, ant] = activeSCs

                    # Update offset
                    if (sym_in_slot + 1) == symbols_per_slot:
                        offset += nFFT + cpLength + gapSamples
                    else:
                        offset += nFFT + cpLength

                    symbol_idx += 1

        return reGrid


def lteSCFDMADemodulate(ue: Dict, *args) -> np.ndarray:
    """
    Main function interface matching MATLAB lteSCFDMADemodulate

    Syntax:
        grid = lteSCFDMADemodulate(ue, waveform)
        grid = lteSCFDMADemodulate(ue, waveform, cpFraction)
        grid = lteSCFDMADemodulate(ue, chs, waveform)
        grid = lteSCFDMADemodulate(ue, chs, waveform, cpFraction)
    """
    demodulator = LTESCFDMADemodulator()

    if len(args) == 1:
        # lteSCFDMADemodulate(ue, waveform)
        waveform = args[0]
        return demodulator.demodulate(ue, waveform)

    elif len(args) == 2:
        if isinstance(args[0], dict):
            # lteSCFDMADemodulate(ue, chs, waveform)
            chs = args[0]
            waveform = args[1]
            return demodulator.demodulate(ue, waveform, chs=chs)
        else:
            # lteSCFDMADemodulate(ue, waveform, cpFraction)
            waveform = args[0]
            cpFraction = args[1]
            return demodulator.demodulate(ue, waveform, cpFraction=cpFraction)

    elif len(args) == 3:
        # lteSCFDMADemodulate(ue, chs, waveform, cpFraction)
        chs = args[0]
        waveform = args[1]
        cpFraction = args[2]
        return demodulator.demodulate(ue, waveform, chs=chs, cpFraction=cpFraction)

    else:
        raise ValueError("Invalid number of arguments")


if __name__ == '__main__':
    print("SC-FDMA Demodulator - MATLAB-Exact Implementation")
    print("="*60)

    # Test round-trip
    from lte_scfdma_modulate import lteSCFDMAModulate

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal'}

    # Create test grid
    grid_original = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

    # Modulate
    waveform, info = lteSCFDMAModulate(ue, grid_original)
    print(f"Original grid: {grid_original.shape}")
    print(f"Waveform:      {waveform.shape}")

    # Demodulate
    grid_recovered = lteSCFDMADemodulate(ue, waveform)
    print(f"Recovered:     {grid_recovered.shape}")

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
