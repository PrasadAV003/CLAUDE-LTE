# ----------------------------------------------------------------------
#  LTE Symbol Modulation and Demodulation - CPU Version
#  MATLAB-Compatible Implementation per 3GPP TS 36.211
# ----------------------------------------------------------------------
"""
LTE Symbol Modulation/Demodulation Module (CPU Version)

This module implements 3GPP TS 36.211 compliant modulation schemes:
- BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM

Functions:
    lteSymbolModulate: Modulate bits to complex symbols
    lteSymbolDemodulate: Demodulate symbols to bits (hard/soft decision)

References:
    [1] 3GPP TS 36.211, Section 7.1
    [2] F. Tosato and P. Bisaglia, "Simplified soft-output demapper for binary
        interleaved COFDM with application to HIPERLAN/2", IEEE ICC 2002.
"""

import numpy as np
from typing import Union, List, Optional


class LTEModulator:
    """MATLAB Compatible LTE Modulation/Demodulation System

    Implements 3GPP TS 36.211 compliant modulation schemes with
    proper constellation normalization.

    Attributes:
        modulation_schemes: Dictionary mapping scheme names to bits per symbol
        norm_factors: Normalization factors for each modulation scheme
        constellations: Pre-computed constellation points for each scheme
    """

    def __init__(self):
        self.modulation_schemes = {
            'BPSK': 1,
            'QPSK': 2,
            '16QAM': 4,
            '64QAM': 6,
            '256QAM': 8,
            '1024QAM': 10
        }
        self.norm_factors = {
            'BPSK': 1.0 / np.sqrt(2),
            'QPSK': 1.0 / np.sqrt(2),
            '16QAM': 1.0 / np.sqrt(10),
            '64QAM': 1.0 / np.sqrt(42),
            '256QAM': 1.0 / np.sqrt(170),
            '1024QAM': 1.0 / np.sqrt(682)
        }
        self.constellations = {
            'BPSK': self._table_bpsk(),
            'QPSK': self._table_qpsk(),
            '16QAM': self._table_16qam(),
            '64QAM': self._table_64qam(),
            '256QAM': self._table_256qam(),
            '1024QAM': self._table_1024qam()
        }

    def _table_bpsk(self) -> np.ndarray:
        """3GPP TS 36.211 Table 7.1.1-1: BPSK modulation mapping"""
        norm = 1.0 / np.sqrt(2)
        constellation = np.zeros(2, dtype=np.complex128)
        constellation[0] = (1 + 1j) * norm   # b=0 → +1/√2 + j*1/√2
        constellation[1] = (-1 - 1j) * norm  # b=1 → -1/√2 - j*1/√2
        return constellation

    def _table_qpsk(self) -> np.ndarray:
        """3GPP TS 36.211 Table 7.1.2-1: QPSK"""
        norm = self.norm_factors['QPSK']
        table = [
            (0, 0, 1, 1), (0, 1, 1, -1),
            (1, 0, -1, 1), (1, 1, -1, -1)
        ]
        constellation = np.zeros(4, dtype=np.complex128)
        for b0, b1, I, Q in table:
            idx = (b0 << 1) | b1
            constellation[idx] = (I + 1j * Q) * norm
        return constellation

    def _table_16qam(self) -> np.ndarray:
        """3GPP TS 36.211 Table 7.1.3-1: 16QAM"""
        norm = self.norm_factors['16QAM']
        table = [
            (0,0,0,0, 1, 1), (0,0,0,1, 1, 3), (0,0,1,0, 3, 1), (0,0,1,1, 3, 3),
            (0,1,0,0, 1,-1), (0,1,0,1, 1,-3), (0,1,1,0, 3,-1), (0,1,1,1, 3,-3),
            (1,0,0,0,-1, 1), (1,0,0,1,-1, 3), (1,0,1,0,-3, 1), (1,0,1,1,-3, 3),
            (1,1,0,0,-1,-1), (1,1,0,1,-1,-3), (1,1,1,0,-3,-1), (1,1,1,1,-3,-3),
        ]
        constellation = np.zeros(16, dtype=np.complex128)
        for b0, b1, b2, b3, I, Q in table:
            idx = (b0 << 3) | (b1 << 2) | (b2 << 1) | b3
            constellation[idx] = (I + 1j * Q) * norm
        return constellation

    def _table_64qam(self) -> np.ndarray:
        """3GPP TS 36.211 Table 7.1.4-1: 64QAM"""
        norm = self.norm_factors['64QAM']
        table = [
            (0,0,0,0,0,0, 3, 3), (0,0,0,0,0,1, 3, 1), (0,0,0,0,1,0, 1, 3), (0,0,0,0,1,1, 1, 1),
            (0,0,0,1,0,0, 3, 5), (0,0,0,1,0,1, 3, 7), (0,0,0,1,1,0, 1, 5), (0,0,0,1,1,1, 1, 7),
            (0,0,1,0,0,0, 5, 3), (0,0,1,0,0,1, 5, 1), (0,0,1,0,1,0, 7, 3), (0,0,1,0,1,1, 7, 1),
            (0,0,1,1,0,0, 5, 5), (0,0,1,1,0,1, 5, 7), (0,0,1,1,1,0, 7, 5), (0,0,1,1,1,1, 7, 7),
            (0,1,0,0,0,0, 3,-3), (0,1,0,0,0,1, 3,-1), (0,1,0,0,1,0, 1,-3), (0,1,0,0,1,1, 1,-1),
            (0,1,0,1,0,0, 3,-5), (0,1,0,1,0,1, 3,-7), (0,1,0,1,1,0, 1,-5), (0,1,0,1,1,1, 1,-7),
            (0,1,1,0,0,0, 5,-3), (0,1,1,0,0,1, 5,-1), (0,1,1,0,1,0, 7,-3), (0,1,1,0,1,1, 7,-1),
            (0,1,1,1,0,0, 5,-5), (0,1,1,1,0,1, 5,-7), (0,1,1,1,1,0, 7,-5), (0,1,1,1,1,1, 7,-7),
            (1,0,0,0,0,0,-3, 3), (1,0,0,0,0,1,-3, 1), (1,0,0,0,1,0,-1, 3), (1,0,0,0,1,1,-1, 1),
            (1,0,0,1,0,0,-3, 5), (1,0,0,1,0,1,-3, 7), (1,0,0,1,1,0,-1, 5), (1,0,0,1,1,1,-1, 7),
            (1,0,1,0,0,0,-5, 3), (1,0,1,0,0,1,-5, 1), (1,0,1,0,1,0,-7, 3), (1,0,1,0,1,1,-7, 1),
            (1,0,1,1,0,0,-5, 5), (1,0,1,1,0,1,-5, 7), (1,0,1,1,1,0,-7, 5), (1,0,1,1,1,1,-7, 7),
            (1,1,0,0,0,0,-3,-3), (1,1,0,0,0,1,-3,-1), (1,1,0,0,1,0,-1,-3), (1,1,0,0,1,1,-1,-1),
            (1,1,0,1,0,0,-3,-5), (1,1,0,1,0,1,-3,-7), (1,1,0,1,1,0,-1,-5), (1,1,0,1,1,1,-1,-7),
            (1,1,1,0,0,0,-5,-3), (1,1,1,0,0,1,-5,-1), (1,1,1,0,1,0,-7,-3), (1,1,1,0,1,1,-7,-1),
            (1,1,1,1,0,0,-5,-5), (1,1,1,1,0,1,-5,-7), (1,1,1,1,1,0,-7,-5), (1,1,1,1,1,1,-7,-7),
        ]
        constellation = np.zeros(64, dtype=np.complex128)
        for b0, b1, b2, b3, b4, b5, I, Q in table:
            idx = (b0 << 5) | (b1 << 4) | (b2 << 3) | (b3 << 2) | (b4 << 1) | b5
            constellation[idx] = (I + 1j * Q) * norm
        return constellation

    def _table_256qam(self) -> np.ndarray:
        """3GPP TS 36.211 Table 7.1.5-1: 256QAM"""
        norm = self.norm_factors['256QAM']
        table_IQ = [
            (5,5), (5,7), (7,5), (7,7), (5,3), (5,1), (7,3), (7,1),
            (3,5), (3,7), (1,5), (1,7), (3,3), (3,1), (1,3), (1,1),
            (5,11), (5,9), (7,11), (7,9), (5,13), (5,15), (7,13), (7,15),
            (3,11), (3,9), (1,11), (1,9), (3,13), (3,15), (1,13), (1,15),
            (11,5), (11,7), (9,5), (9,7), (11,3), (11,1), (9,3), (9,1),
            (13,5), (13,7), (15,5), (15,7), (13,3), (13,1), (15,3), (15,1),
            (11,11), (11,9), (9,11), (9,9), (11,13), (11,15), (9,13), (9,15),
            (13,11), (13,9), (15,11), (15,9), (13,13), (13,15), (15,13), (15,15),
            (5,-5), (5,-7), (7,-5), (7,-7), (5,-3), (5,-1), (7,-3), (7,-1),
            (3,-5), (3,-7), (1,-5), (1,-7), (3,-3), (3,-1), (1,-3), (1,-1),
            (5,-11), (5,-9), (7,-11), (7,-9), (5,-13), (5,-15), (7,-13), (7,-15),
            (3,-11), (3,-9), (1,-11), (1,-9), (3,-13), (3,-15), (1,-13), (1,-15),
            (11,-5), (11,-7), (9,-5), (9,-7), (11,-3), (11,-1), (9,-3), (9,-1),
            (13,-5), (13,-7), (15,-5), (15,-7), (13,-3), (13,-1), (15,-3), (15,-1),
            (11,-11), (11,-9), (9,-11), (9,-9), (11,-13), (11,-15), (9,-13), (9,-15),
            (13,-11), (13,-9), (15,-11), (15,-9), (13,-13), (13,-15), (15,-13), (15,-15),
            (-5,5), (-5,7), (-7,5), (-7,7), (-5,3), (-5,1), (-7,3), (-7,1),
            (-3,5), (-3,7), (-1,5), (-1,7), (-3,3), (-3,1), (-1,3), (-1,1),
            (-5,11), (-5,9), (-7,11), (-7,9), (-5,13), (-5,15), (-7,13), (-7,15),
            (-3,11), (-3,9), (-1,11), (-1,9), (-3,13), (-3,15), (-1,13), (-1,15),
            (-11,5), (-11,7), (-9,5), (-9,7), (-11,3), (-11,1), (-9,3), (-9,1),
            (-13,5), (-13,7), (-15,5), (-15,7), (-13,3), (-13,1), (-15,3), (-15,1),
            (-11,11), (-11,9), (-9,11), (-9,9), (-11,13), (-11,15), (-9,13), (-9,15),
            (-13,11), (-13,9), (-15,11), (-15,9), (-13,13), (-13,15), (-15,13), (-15,15),
            (-5,-5), (-5,-7), (-7,-5), (-7,-7), (-5,-3), (-5,-1), (-7,-3), (-7,-1),
            (-3,-5), (-3,-7), (-1,-5), (-1,-7), (-3,-3), (-3,-1), (-1,-3), (-1,-1),
            (-5,-11), (-5,-9), (-7,-11), (-7,-9), (-5,-13), (-5,-15), (-7,-13), (-7,-15),
            (-3,-11), (-3,-9), (-1,-11), (-1,-9), (-3,-13), (-3,-15), (-1,-13), (-1,-15),
            (-11,-5), (-11,-7), (-9,-5), (-9,-7), (-11,-3), (-11,-1), (-9,-3), (-9,-1),
            (-13,-5), (-13,-7), (-15,-5), (-15,-7), (-13,-3), (-13,-1), (-15,-3), (-15,-1),
            (-11,-11), (-11,-9), (-9,-11), (-9,-9), (-11,-13), (-11,-15), (-9,-13), (-9,-15),
            (-13,-11), (-13,-9), (-15,-11), (-15,-9), (-13,-13), (-13,-15), (-15,-13), (-15,-15),
        ]
        constellation = np.zeros(256, dtype=np.complex128)
        for idx, (I, Q) in enumerate(table_IQ):
            constellation[idx] = (I + 1j * Q) * norm
        return constellation

    def _table_1024qam(self) -> np.ndarray:
        """1024-QAM according to TS 36.211 Table 7.1.6-1"""
        norm = self.norm_factors['1024QAM']
        const = np.zeros(1024, dtype=np.complex128)

        for idx in range(1024):
            # 10 bits → b0 (MSB) … b9 (LSB)
            b = np.array([(idx >> (9 - i)) & 1 for i in range(10)], dtype=int)

            # ---- In-phase (I) ----
            I = (1 - 2*b[0]) * (
                    16 - (1 - 2*b[2]) * (
                    8  - (1 - 2*b[4]) * (
                    4  - (1 - 2*b[6]) * (
                    2  - (1 - 2*b[8])))))

            # ---- Quadrature (Q) ----
            Q = (1 - 2*b[1]) * (
                    16 - (1 - 2*b[3]) * (
                    8  - (1 - 2*b[5]) * (
                    4  - (1 - 2*b[7]) * (
                    2  - (1 - 2*b[9])))))

            const[idx] = (I + 1j*Q) * norm

        return const


# Global modulator instance for convenience
_modulator = LTEModulator()


def lteSymbolModulate(in_bits: Union[List, np.ndarray], mod: str) -> np.ndarray:
    """
    lteSymbolModulate - Symbol modulation per 3GPP TS 36.211

    OUT = lteSymbolModulate(IN, MOD) maps the bit values in vector IN to
    complex modulation symbols with the modulation scheme specified in MOD.

    Parameters
    ----------
    in_bits : array_like
        Input bits as column vector where each bit is 0 or 1.
        Can be list or numpy array.
        Length must be multiple of:
        - 1 for BPSK
        - 2 for QPSK
        - 4 for 16QAM
        - 6 for 64QAM
        - 8 for 256QAM
        - 10 for 1024QAM
    mod : str
        Modulation scheme: 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM', '1024QAM'

    Returns
    -------
    out : ndarray
        Complex modulated symbols as column vector (complex128)

    Examples
    --------
    >>> sym = lteSymbolModulate([0, 1, 1, 0], 'QPSK')
    >>> # Returns: [0.7071-0.7071j; -0.7071+0.7071j]

    Notes
    -----
    Output symbols use constellation power normalization per TS 36.211:
    - BPSK/QPSK: 1/sqrt(2)
    - 16QAM: 1/sqrt(10)
    - 64QAM: 1/sqrt(42)
    - 256QAM: 1/sqrt(170)
    - 1024QAM: 1/sqrt(682)

    References
    ----------
    [1] 3GPP TS 36.211, Section 7.1
    """
    # Input validation and conversion
    if isinstance(in_bits, list):
        in_bits = np.array(in_bits, dtype=np.float64)
    elif not isinstance(in_bits, np.ndarray):
        in_bits = np.array(in_bits, dtype=np.float64)
    else:
        in_bits = np.asarray(in_bits, dtype=np.float64)

    # Handle empty input
    if len(in_bits) == 0:
        return np.zeros((0, 1), dtype=np.complex128)

    # Convert to column vector if needed
    if in_bits.ndim == 1:
        in_bits = in_bits.reshape(-1, 1)
    elif in_bits.ndim == 2 and in_bits.shape[1] != 1:
        raise ValueError("The input must be a vector and not a matrix.")

    # Validate bits are 0 or 1
    if not np.all((in_bits == 0) | (in_bits == 1)):
        raise ValueError("Input bits must be 0 or 1")

    # Validate modulation scheme
    valid_mods = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']
    if mod not in valid_mods:
        raise ValueError(f"Modulation ({mod}) must be one of ({', '.join(valid_mods)}).")

    bits_per_symbol = _modulator.modulation_schemes[mod]
    constellation = _modulator.constellations[mod]

    n_bits = len(in_bits)
    if n_bits % bits_per_symbol != 0:
        raise ValueError(
            f"Input length ({n_bits}) must be a multiple of the number of bits per symbol ({bits_per_symbol})."
        )

    n_symbols = n_bits // bits_per_symbol
    out = np.zeros((n_symbols, 1), dtype=np.complex128)
    in_bits_flat = in_bits.flatten()

    for i in range(n_symbols):
        start_idx = i * bits_per_symbol
        end_idx = start_idx + bits_per_symbol
        symbol_bits = in_bits_flat[start_idx:end_idx]

        idx = 0
        for j, bit in enumerate(symbol_bits):
            idx |= (int(bit) << (bits_per_symbol - 1 - j))

        out[i, 0] = constellation[idx]

    return out


def lteSymbolDemodulate(in_symbols: Union[List, np.ndarray], mod: str,
                        dec: str = 'Soft') -> np.ndarray:
    """
    lteSymbolDemodulate - Demodulation and symbol to bit conversion

    OUT = lteSymbolDemodulate(IN, MOD) returns a column vector containing bits
    resulting from soft constellation demodulation of complex values in vector IN.

    OUT = lteSymbolDemodulate(IN, MOD, DEC) allows the decision mode DEC to be
    specified, one of ('Hard', 'Soft'). Default is 'Soft'.

    Parameters
    ----------
    in_symbols : array_like
        Column vector of complex numeric values (symbols to demodulate)
        Can be list or numpy array
    mod : str
        Modulation format: 'BPSK', 'QPSK', '16QAM', '64QAM', '256QAM', '1024QAM'
    dec : str, optional
        Decision mode: 'Hard' or 'Soft' (default: 'Soft')

    Returns
    -------
    out : ndarray
        Column vector of demodulated bits (double precision)
        - Hard decision: bits as 0 or 1
        - Soft decision: LLR values (sign indicates bit, magnitude indicates confidence)

    Examples
    --------
    >>> out = lteSymbolDemodulate([0.7 - 0.7j, -0.7 + 0.7j], 'QPSK', 'Hard')
    >>> out = lteSymbolDemodulate(symbols, '16QAM')  # Soft decision by default

    Notes
    -----
    Demodulation assumes constellation power normalization per TS 36.211:
    - BPSK/QPSK: 1/sqrt(2)
    - 16QAM: 1/sqrt(10)
    - 64QAM: 1/sqrt(42)
    - 256QAM: 1/sqrt(170)
    - 1024QAM: 1/sqrt(682)

    For soft decision, LLR scaling magnitudes:
    - BPSK: [1]
    - QPSK: [1/sqrt(2)]
    - 16QAM: [1, 3]/sqrt(10)
    - 64QAM: [1, 3, 5, 7]/sqrt(42)
    - 256QAM: [1, 3, 5, 7, 9, 11, 13, 15]/sqrt(170)
    - 1024QAM: [1:2:31]/sqrt(682)

    References
    ----------
    [1] F. Tosato and P. Bisaglia, "Simplified soft-output demapper for binary
        interleaved COFDM with application to HIPERLAN/2", IEEE ICC 2002.
    [2] 3GPP TS 36.211, Section 7.1
    """
    # Input validation and conversion
    if isinstance(in_symbols, list):
        in_symbols = np.array(in_symbols, dtype=np.complex128)
    elif not isinstance(in_symbols, np.ndarray):
        in_symbols = np.array(in_symbols, dtype=np.complex128)

    # Ensure complex128 dtype for compatibility
    if in_symbols.dtype != np.complex128:
        in_symbols = in_symbols.astype(np.complex128)

    # Ensure column vector (flatten to 1D, will reshape output to column)
    if in_symbols.ndim == 2:
        in_symbols = in_symbols.flatten()
    elif in_symbols.ndim > 2:
        raise ValueError("Input must be a vector (1D or column vector)")

    # Validate modulation scheme
    valid_mods = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']
    if mod not in valid_mods:
        raise ValueError(f"Modulation must be one of {valid_mods}, got '{mod}'")

    # Validate decision mode
    if dec not in ['Hard', 'Soft']:
        raise ValueError(f"Decision mode must be 'Hard' or 'Soft', got '{dec}'")

    # Get modulation parameters
    M = {'BPSK':2, 'QPSK':4, '16QAM':16, '64QAM':64,
         '256QAM':256, '1024QAM':1024}[mod]
    bps = int(np.log2(M))

    const = _modulator.constellations[mod]

    # Hard decision demodulation
    if dec == 'Hard':
        # Vectorized hard decision using minimum Euclidean distance
        distances = np.abs(in_symbols[:, None] - const[None, :]) ** 2
        indices = np.argmin(distances, axis=1)

        # Convert indices to bits (column vector)
        out = np.zeros((len(in_symbols) * bps, 1), dtype=np.float64)
        for b in range(bps):
            bit_vals = (indices >> (bps - 1 - b)) & 1
            out[b::bps, 0] = bit_vals.astype(np.float64)

        return out

    # Soft decision demodulation
    scale = {
        'BPSK'   : np.array([1.0]),
        'QPSK'   : np.array([1.0]) / np.sqrt(2),
        '16QAM'  : np.array([1.0, 3.0]) / np.sqrt(10),
        '64QAM'  : np.array([1.0, 3.0, 5.0, 7.0]) / np.sqrt(42),
        '256QAM' : np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]) / np.sqrt(170),
        '1024QAM': np.arange(1.0, 32.0, 2.0) / np.sqrt(682)
    }[mod]

    out = np.zeros((len(in_symbols) * bps, 1), dtype=np.float64)

    # BPSK: Rotate by -45° and use real part (MATLAB compatible)
    if mod == 'BPSK':
        # Rotate all symbols by -45 degrees: s × (1-j)/√2
        # This aligns BPSK constellation to real axis for demodulation
        s_rotated = in_symbols * (1 - 1j) / np.sqrt(2)

        # LLR = negative of real part (scaled)
        # Sign: -ve for bit=0 (+ve constellation point), +ve for bit=1 (-ve point)
        llr = -s_rotated.real * scale[0]
        out[:, 0] = llr

        return out

    # For QPSK and QAM modulations - use I/Q separation
    Ilev = np.sort(np.unique(const.real))
    Qlev = np.sort(np.unique(const.imag))

    rI = in_symbols.real
    rQ = in_symbols.imag

    # Process each bit position
    for b in range(bps):
        # Alternate between I (even bits) and Q (odd bits)
        r = rI if b % 2 == 0 else rQ
        lev = Ilev if b % 2 == 0 else Qlev

        # Initialize minimum distances
        d0_min = np.full(len(in_symbols), np.inf, dtype=np.float64)
        d1_min = np.full(len(in_symbols), np.inf, dtype=np.float64)
        c0_level = np.zeros(len(in_symbols), dtype=np.float64)
        c1_level = np.zeros(len(in_symbols), dtype=np.float64)

        # Find minimum distances for bit=0 and bit=1 (vectorized over constellation)
        for idx in range(M):
            bit_val = (idx >> (bps - 1 - b)) & 1
            lev_val = const[idx].real if b % 2 == 0 else const[idx].imag
            dist = np.abs(r - lev_val)

            if bit_val == 0:
                mask = dist < d0_min
                d0_min = np.where(mask, dist, d0_min)
                c0_level = np.where(mask, lev_val, c0_level)
            else:
                mask = dist < d1_min
                d1_min = np.where(mask, dist, d1_min)
                c1_level = np.where(mask, lev_val, c1_level)

        # Calculate scale index based on level positions
        idx0_arr = np.zeros(len(in_symbols), dtype=np.int32)
        idx1_arr = np.zeros(len(in_symbols), dtype=np.int32)

        # Find which level index c0 and c1 correspond to
        for k in range(len(lev)):
            mask_c0 = np.abs(c0_level - lev[k]) < 1e-10
            mask_c1 = np.abs(c1_level - lev[k]) < 1e-10

            idx0_arr = np.where(mask_c0, k, idx0_arr)
            idx1_arr = np.where(mask_c1, k, idx1_arr)

        # Calculate scale index (distance between levels)
        scale_idx = np.abs(idx1_arr - idx0_arr) - 1
        scale_idx = np.clip(scale_idx, 0, len(scale) - 1)

        # Calculate LLR with correct sign
        # LLR = d(bit=0) - d(bit=1)
        # Positive LLR → more likely bit=1
        # Negative LLR → more likely bit=0
        raw_llr = d0_min - d1_min
        llr_sign = np.sign(raw_llr)

        # Handle zero case (exactly on decision boundary)
        llr_sign = np.where(np.abs(raw_llr) < 1e-12, 0.0, llr_sign)

        # Apply scaling (piecewise linear approximation to true LLR)
        scaled_llr = scale[scale_idx] * llr_sign

        # Store results in output column vector
        out[b::bps, 0] = scaled_llr

    return out


# ----------------------------------------------------------------------
#  Channel Functions
# ----------------------------------------------------------------------

def add_awgn_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add AWGN noise to signal for given SNR in dB

    Parameters
    ----------
    signal : ndarray
        Input signal (complex or real)
    snr_db : float
        Signal-to-noise ratio in dB

    Returns
    -------
    ndarray
        Noisy signal
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise_real = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    noise_imag = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    noise = noise_real + 1j * noise_imag

    return signal + noise


def apply_rayleigh_fading(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Apply Rayleigh fading channel with AWGN

    Parameters
    ----------
    signal : ndarray
        Input signal (complex)
    snr_db : float
        Signal-to-noise ratio in dB

    Returns
    -------
    ndarray
        Faded and noisy signal (with ZF equalization)
    """
    # Generate Rayleigh fading coefficients
    h_real = np.random.randn(len(signal)) / np.sqrt(2)
    h_imag = np.random.randn(len(signal)) / np.sqrt(2)
    h = h_real + 1j * h_imag

    # Apply fading
    faded_signal = signal * h

    # Add AWGN
    faded_power = np.mean(np.abs(faded_signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = faded_power / snr_linear

    noise_real = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    noise_imag = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    noise = noise_real + 1j * noise_imag

    # Zero-forcing equalization
    received_signal = faded_signal + noise
    received_signal = received_signal / (h + 1e-10)

    return received_signal


def calculate_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> tuple:
    """
    Calculate Bit Error Rate

    Parameters
    ----------
    bits_tx : ndarray
        Transmitted bits
    bits_rx : ndarray
        Received bits

    Returns
    -------
    tuple
        (BER, num_errors, total_bits)
    """
    min_len = min(len(bits_tx), len(bits_rx))
    bits_tx = bits_tx[:min_len].flatten()
    bits_rx = bits_rx[:min_len].flatten()

    errors = np.sum(bits_tx != bits_rx)
    ber = errors / min_len

    return ber, int(errors), min_len


def calculate_bler(bits_tx: np.ndarray, bits_rx: np.ndarray,
                   block_size: int = 1000) -> tuple:
    """
    Calculate Block Error Rate

    Parameters
    ----------
    bits_tx : ndarray
        Transmitted bits
    bits_rx : ndarray
        Received bits
    block_size : int
        Size of each block in bits

    Returns
    -------
    tuple
        (BLER, block_errors, total_blocks)
    """
    min_len = min(len(bits_tx), len(bits_rx))
    bits_tx = bits_tx[:min_len].flatten()
    bits_rx = bits_rx[:min_len].flatten()

    n_blocks = min_len // block_size

    if n_blocks == 0:
        return 0.0, 0, 0

    # Reshape into blocks
    bits_tx_blocks = bits_tx[:n_blocks * block_size].reshape(n_blocks, block_size)
    bits_rx_blocks = bits_rx[:n_blocks * block_size].reshape(n_blocks, block_size)

    # Check if blocks are equal (vectorized)
    block_errors = np.sum(np.any(bits_tx_blocks != bits_rx_blocks, axis=1))

    bler = block_errors / n_blocks

    return float(bler), int(block_errors), n_blocks


# ----------------------------------------------------------------------
#  Test Function
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("LTE Modulation/Demodulation Module - CPU Version")
    print("=" * 60)

    # Test each modulation scheme
    schemes = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']
    bps_list = [1, 2, 4, 6, 8, 10]

    for mod, bps in zip(schemes, bps_list):
        # Generate random bits
        n_bits = bps * 100
        bits = np.random.randint(0, 2, n_bits).astype(np.float64)

        # Modulate
        symbols = lteSymbolModulate(bits, mod)

        # Demodulate (hard decision)
        demod_bits = lteSymbolDemodulate(symbols, mod, 'Hard')

        # Check BER
        ber, errors, total = calculate_ber(bits, demod_bits.flatten())

        print(f"{mod:>8}: {n_bits} bits → {len(symbols)} symbols → BER = {ber:.6f} ({errors} errors)")

    print("\n" + "=" * 60)
    print("AWGN Channel Test (QPSK at 10 dB SNR)")

    # Generate bits
    bits = np.random.randint(0, 2, 2000).astype(np.float64)

    # Modulate
    symbols = lteSymbolModulate(bits, 'QPSK')

    # Add AWGN noise
    noisy_symbols = add_awgn_noise(symbols.flatten(), 10.0)

    # Demodulate
    demod_bits = lteSymbolDemodulate(noisy_symbols, 'QPSK', 'Hard')

    # Check BER
    ber, errors, total = calculate_ber(bits, demod_bits.flatten())
    print(f"BER = {ber:.6f} ({errors} errors out of {total} bits)")

    print("\nAll tests completed!")
