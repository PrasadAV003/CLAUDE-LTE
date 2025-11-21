"""
SC-FDMA Modulator - MATLAB-Exact Implementation
================================================

Complete Python implementation matching MATLAB lteSCFDMAModulate exactly.
Based on 3GPP TS 36.211 and MATLAB R2022b implementation.

Key Features:
- Exact IFFT input mapping (contiguous block centered at DC)
- fftshift BEFORE ifft (as per MATLAB)
- Correct CP addition with windowing samples
- Half-subcarrier shift with exact phase formula
- Raised-cosine windowing with overlap-add
- "Head" chopping and final overlap for seamless looping

Author: CLAUDE-LTE Project
Date: 2025-11-20
"""

import numpy as np
import pyfftw
from typing import Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Enable PyFFTW optimizations
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

ZERO_THRESHOLD = 1e-15


def apply_zero_threshold(arr: np.ndarray) -> np.ndarray:
    """Apply zero threshold to avoid signed zero issues"""
    arr = arr.copy()
    arr.real[np.abs(arr.real) < ZERO_THRESHOLD] = 0.0
    arr.imag[np.abs(arr.imag) < ZERO_THRESHOLD] = 0.0
    return arr


@dataclass
class SCFDMAInfo:
    """SC-FDMA Modulation Information"""
    SamplingRate: float
    Nfft: int
    Windowing: int
    CyclicPrefixLengths: np.ndarray
    NBULGapSamples: int = 0


class LTESCFDMAModulator:
    """
    LTE SC-FDMA Modulator - MATLAB-Exact Implementation

    Matches MATLAB lteSCFDMAModulate behavior exactly:
    - IFFT input mapping: contiguous block at (nFFT/2) - (nSC/2) + 1
    - fftshift applied BEFORE ifft
    - CP with windowing: [tail(cpLength+N)... ifft_output]
    - Half-SC shift: exp(1j*pi*((-cpLength-N):(nFFT-1))/nFFT)
    - Windowing with raised cosine
    - Overlap-add with "head" chopping
    """

    # Constants
    DELTA_F = 15000  # Hz
    N_FFT_BASE = 2048
    BASE_SAMPLING_RATE = 30.72e6  # 30.72 MHz
    NB_IOT_SAMPLING_RATE = 1.92e6  # 1.92 MHz

    # NFFT mapping for standard configurations
    NFFT_MAP = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 2048, 100: 2048}

    # Default windowing per MATLAB
    DEFAULT_WINDOWING = {
        'Normal': {6: 4, 15: 6, 25: 4, 50: 6, 75: 8, 100: 8},
        'Extended': {6: 4, 15: 6, 25: 4, 50: 6, 75: 8, 100: 8}
    }

    def __init__(self):
        """Initialize modulator with FFT plans"""
        self._ifft_plans = {}

    def _get_ifft_plan(self, nfft: int):
        """Get or create cached IFFT plan"""
        if nfft not in self._ifft_plans:
            freq_array = pyfftw.empty_aligned(nfft, dtype='complex128')
            time_array = pyfftw.empty_aligned(nfft, dtype='complex128')

            ifft_plan = pyfftw.FFTW(
                freq_array, time_array,
                direction='FFTW_BACKWARD',
                flags=('FFTW_MEASURE',),
                threads=pyfftw.config.NUM_THREADS,
                normalise_idft=True  # Match MATLAB ifft normalization
            )

            self._ifft_plans[nfft] = {
                'plan': ifft_plan,
                'freq_array': freq_array,
                'time_array': time_array
            }

        return self._ifft_plans[nfft]

    def _determine_nfft(self, nsc: int, ue: Dict) -> int:
        """
        Determine NFFT size matching MATLAB logic

        From MATLAB lteSCFDMAModulate:
        - For nSC power of 2 and > 64: nrb = fix(0.85*nSC/12)
        - Otherwise: nrb = fix(nSC/12)
        - Then lookup or calculate NFFT for 85% occupancy
        """
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
            # Smallest power of 2 >= 12*nrb/0.85
            min_fft = 12 * nrb / 0.85
            return int(2 ** np.ceil(np.log2(min_fft)))

    def _get_cyclic_prefix_lengths(self, nfft: int, cp_type: str,
                                   nb_spacing: Optional[str] = None) -> np.ndarray:
        """
        Get CP lengths matching MATLAB exactly

        MATLAB returns CP for TWO slots (14 or 12 symbols total)
        """
        if nb_spacing == '3.75kHz':
            # NB-IoT 3.75kHz: uniform 16 samples for 14 symbols
            return np.array([16] * 14, dtype=np.int32)

        if nb_spacing == '15kHz' or nfft == 128:
            # NB-IoT 15kHz or single narrowband
            if cp_type == 'Normal':
                return np.array([10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9], dtype=np.int32)
            else:
                return np.array([32] * 12, dtype=np.int32)

        # Standard LTE: scale from base NFFT=2048
        scale_factor = nfft / 2048.0

        if cp_type == 'Normal':
            # TWO slots: [160, 144*6] repeated twice
            base_cp = np.array([160, 144, 144, 144, 144, 144, 144,
                               160, 144, 144, 144, 144, 144, 144], dtype=np.float64)
        else:  # Extended
            # TWO slots: 12 symbols total
            base_cp = np.array([512] * 12, dtype=np.float64)

        return np.round(base_cp * scale_factor).astype(np.int32)

    def _get_default_windowing(self, nrb: int, cp_type: str,
                               nb_spacing: Optional[str] = None) -> int:
        """Get default windowing matching MATLAB"""
        if nb_spacing == '15kHz':
            return 6
        elif nb_spacing == '3.75kHz':
            return 4

        if nrb in self.DEFAULT_WINDOWING[cp_type]:
            return self.DEFAULT_WINDOWING[cp_type][nrb]

        return 0

    def _raised_cosine_window(self, symbol_length: int, N: int) -> np.ndarray:
        """
        Create raised-cosine window matching MATLAB

        MATLAB: lte.internal.raisedCosineWindow(nFFT+cpLength, N)
        """
        if N == 0:
            return np.ones(symbol_length)

        window = np.ones(symbol_length)

        # Rising edge at start
        for n in range(N):
            window[n] = 0.5 * (1 - np.cos(np.pi * n / N))

        # Falling edge at end
        for n in range(N):
            window[symbol_length - N + n] = 0.5 * (1 + np.cos(np.pi * n / N))

        return window

    def modulate(self, ue: Dict, grid: np.ndarray,
                chs: Optional[Dict] = None,
                windowing: Optional[int] = None) -> Tuple[np.ndarray, SCFDMAInfo]:
        """
        SC-FDMA modulation matching MATLAB lteSCFDMAModulate

        MATLAB algorithm:
        1. Map grid to IFFT input: ifftin(firstSC+(0:nSC-1),:) = grid(:,i,:)
        2. Perform IFFT: iffout = ifft(fftshift(ifftin,1))
        3. Add CP with windowing: [iffout(end-(cpLength+N)+1:end); iffout]
        4. Half-SC shift: * exp(1j*pi*((-cpLength-N):(nFFT-1))/nFFT)
        5. Apply raised-cosine window
        6. Overlap-add with "head" chopping

        Args:
            ue: UE configuration dict
            grid: Resource grid (M x N x P)
            chs: Optional channel config
            windowing: Optional windowing override

        Returns:
            waveform: Time-domain waveform (T x P)
            info: Modulation information
        """
        # Validate input
        if grid.ndim not in [2, 3]:
            raise ValueError("Grid must be 2D or 3D")

        nSC = grid.shape[0]
        nSymbols = grid.shape[1]
        nAnts = grid.shape[2] if grid.ndim == 3 else 1

        if grid.ndim == 2:
            grid = grid.reshape(nSC, nSymbols, 1)

        # Determine NFFT
        nFFT = self._determine_nfft(nSC, ue)

        # Get CP lengths (for two slots)
        cp_type = ue.get('CyclicPrefixUL', 'Normal')
        nb_spacing = ue.get('NBULSubcarrierSpacing')
        cpLengths = self._get_cyclic_prefix_lengths(nFFT, cp_type, nb_spacing)

        symbols_per_slot = len(cpLengths) // 2

        # Validate grid has whole number of slots
        if nSymbols % symbols_per_slot != 0:
            raise ValueError(f"Grid must have whole number of slots "
                           f"({symbols_per_slot} symbols per slot)")

        nSlots = nSymbols // symbols_per_slot

        # Determine windowing
        if windowing is not None:
            N = windowing
        elif 'Windowing' in ue and ue['Windowing'] is not None:
            N = ue['Windowing']
        else:
            nrb = ue.get('NULRB', nSC // 12)
            N = self._get_default_windowing(nrb, cp_type, nb_spacing)

        # Validate windowing
        if N > (nFFT - cpLengths[0]):
            raise ValueError(f"Windowing ({N}) must be <= {nFFT - cpLengths[0]}")

        # Calculate sampling rate
        if nb_spacing:
            sampling_rate = self.NB_IOT_SAMPLING_RATE
        else:
            sampling_rate = (self.BASE_SAMPLING_RATE / self.N_FFT_BASE) * nFFT

        # Gap samples for NB-IoT
        if nb_spacing == '3.75kHz':
            gapSamples = 144
        else:
            gapSamples = 0

        # Create info structure
        info = SCFDMAInfo(
            SamplingRate=sampling_rate,
            Nfft=nFFT,
            Windowing=N,
            CyclicPrefixLengths=cpLengths,
            NBULGapSamples=gapSamples
        )

        # Calculate total waveform length
        samplesPerSlot = np.sum(cpLengths[:symbols_per_slot]) + nFFT * symbols_per_slot + gapSamples
        total_samples = int(nSlots * samplesPerSlot)
        waveform = np.zeros((total_samples, nAnts), dtype=np.complex128)

        # Pre-calculate windows (one per CP length)
        # Window length = nFFT + cpLength + N (total extended symbol length)
        window0 = self._raised_cosine_window(nFFT + cpLengths[0] + N, N)
        window1 = self._raised_cosine_window(nFFT + cpLengths[1] + N, N)

        # MATLAB: firstSC = (nFFT/2) - (nSC/2) + 1 (MATLAB 1-indexed)
        # Python (0-indexed): firstSC = nFFT//2 - nSC//2
        firstSC = (nFFT // 2) - (nSC // 2)

        # Get IFFT plan
        ifft_dict = self._get_ifft_plan(nFFT)
        ifft_plan = ifft_dict['plan']
        freq_array = ifft_dict['freq_array']
        time_array = ifft_dict['time_array']

        # Process each antenna
        for ant in range(nAnts):
            pos = 0
            head = None

            # For each symbol
            for i in range(nSymbols):
                # SC-FDMA: DFT spreading (M-point FFT)
                dftOut = np.fft.fft(grid[:, i, ant], nSC)

                # Map to IFFT input - CONTIGUOUS block
                freq_array[:] = 0
                freq_array[firstSC:firstSC+nSC] = dftOut

                # MATLAB: iffout = ifft(fftshift(ifftin,1))
                freq_array[:] = np.fft.fftshift(freq_array)
                ifft_plan()
                iffout = time_array.copy()

                # Get CP length for this symbol
                cpLength = cpLengths[i % len(cpLengths)]

                # MATLAB: extended = [iffout(end-(cpLength+N)+1:end,:); iffout];
                # Python: extended = [iffout[-(cpLength+N):], iffout]
                extended = np.concatenate([iffout[-(cpLength+N):], iffout])

                # MATLAB: extended = extended .* exp(1i*pi*((-cpLength-N):(nFFT-1))/nFFT).';
                phase_indices = np.arange(-cpLength-N, nFFT)
                phase_shift = np.exp(1j * np.pi * phase_indices / nFFT)
                extended = extended * phase_shift

                # Apply zero threshold
                extended = apply_zero_threshold(extended)

                # Apply window
                if i % symbols_per_slot == 0:
                    windowed = extended * window0
                else:
                    windowed = extended * window1

                # Overlap-add with "head" chopping (MATLAB logic)
                if N > 0:
                    if i == 0:
                        # First symbol: chop head and save it
                        head = windowed[:N].copy()
                        L = cpLength + nFFT
                        waveform[pos:pos+L, ant] = windowed[N:N+L]
                    else:
                        # Subsequent symbols: overlap then add
                        L = cpLength + nFFT + N

                        # Add gap samples at end of each slot
                        if (i + 1) % symbols_per_slot == 0:
                            L = cpLength + nFFT + N + gapSamples
                            windowed = np.concatenate([windowed, np.zeros(gapSamples)])

                        waveform[pos-N:pos-N+L, ant] += windowed
                else:
                    # No windowing: just copy samples
                    L = cpLength + nFFT

                    # Add gap samples at end of each slot
                    if (i + 1) % symbols_per_slot == 0:
                        waveform[pos:pos+L, ant] = windowed[:L]
                        waveform[pos+L:pos+L+gapSamples, ant] = 0
                    else:
                        waveform[pos:pos+L, ant] = windowed[:L]

                # Update position
                if (i + 1) % symbols_per_slot == 0:
                    pos += cpLength + nFFT + gapSamples
                else:
                    pos += cpLength + nFFT

            # MATLAB: Finally overlap "head" with end of signal
            if head is not None and N > 0:
                waveform[-N:, ant] += head

        return waveform, info


def lteSCFDMAModulate(ue: Dict, *args) -> Tuple[np.ndarray, SCFDMAInfo]:
    """
    Main function interface matching MATLAB lteSCFDMAModulate

    Syntax:
        [waveform, info] = lteSCFDMAModulate(ue, grid)
        [waveform, info] = lteSCFDMAModulate(ue, grid, windowing)
        [waveform, info] = lteSCFDMAModulate(ue, chs, grid)
        [waveform, info] = lteSCFDMAModulate(ue, chs, grid, windowing)
    """
    modulator = LTESCFDMAModulator()

    if len(args) == 1:
        # lteSCFDMAModulate(ue, grid)
        grid = args[0]
        return modulator.modulate(ue, grid)

    elif len(args) == 2:
        if isinstance(args[0], dict):
            # lteSCFDMAModulate(ue, chs, grid)
            chs = args[0]
            grid = args[1]
            return modulator.modulate(ue, grid, chs=chs)
        else:
            # lteSCFDMAModulate(ue, grid, windowing)
            grid = args[0]
            windowing = args[1]
            return modulator.modulate(ue, grid, windowing=windowing)

    elif len(args) == 3:
        # lteSCFDMAModulate(ue, chs, grid, windowing)
        chs = args[0]
        grid = args[1]
        windowing = args[2]
        return modulator.modulate(ue, grid, chs=chs, windowing=windowing)

    else:
        raise ValueError("Invalid number of arguments")


if __name__ == '__main__':
    print("SC-FDMA Modulator - MATLAB-Exact Implementation")
    print("="*60)

    # Test case
    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal'}
    grid = np.random.randn(72, 14, 1) + 1j * np.random.randn(72, 14, 1)

    waveform, info = lteSCFDMAModulate(ue, grid)

    print(f"Input grid:  {grid.shape}")
    print(f"Waveform:    {waveform.shape}")
    print(f"NFFT:        {info.Nfft}")
    print(f"Windowing:   {info.Windowing}")
    print(f"Sampling:    {info.SamplingRate/1e6:.2f} MHz")
    print("✓ Test complete!")
