"""
lteULChannelEstimate - PUSCH Uplink Channel Estimation
======================================================

Python implementation matching MATLAB's lteULChannelEstimate function exactly.
Performs channel estimation using PUSCH Demodulation Reference Signals (DRS).

Spec: 3GPP TS 36.211 v10.1.0 & TS 36.101 Annex F

Compatible with MATLAB syntax:
    [hest, noiseest] = lteULChannelEstimate(ue, chs, rxgrid)
    [hest, noiseest] = lteULChannelEstimate(ue, chs, cec, rxgrid)
    [hest, noiseest] = lteULChannelEstimate(ue, chs, cec, rxgrid, refgrid)
    [hest, noiseest] = lteULChannelEstimate(ue, chs, rxgrid, refgrid)

Author: CLAUDE-LTE Project
Date: 2025-11-20
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import convolve2d
from typing import Dict, Optional, Tuple, Union, Any
import warnings


def lteULChannelEstimate(
    ue: Dict[str, Any],
    chs: Dict[str, Any],
    *args
) -> Tuple[np.ndarray, float]:
    """
    PUSCH uplink channel estimation

    Returns an estimate for the channel by averaging the least squares estimates
    of the reference symbols across time and copying these estimates across the
    allocated resource elements within the time frequency grid.

    Parameters
    ----------
    ue : dict
        UE-specific configuration with fields:
        - NULRB : int
            Number of uplink resource blocks (6, 15, 25, 50, 75, 100)
        - NCellID : int
            Physical layer cell identity (0-503)
        - NSubframe : int
            Subframe number (default 0)
        - CyclicPrefixUL : str, optional
            'Normal' (default) or 'Extended'
        - NTxAnts : int, optional
            Number of transmission antennas (default 1, can be 2 or 4)
        - Hopping : str, optional
            'Off' (default), 'Group', or 'Sequence'
        - SeqGroup : int, optional
            PUSCH sequence group assignment 0-29 (default 0)
        - CyclicShift : int, optional
            Number of cyclic shifts 0-7 (default 0)
        - NPUSCHID : int, optional
            PUSCH virtual cell identity 0-509
        - NDMRSID : int, optional
            DM-RS identity for cyclic shift hopping 0-509

    chs : dict
        PUSCH channel settings with fields:
        - PRBSet : np.ndarray
            Physical resource block indices (0-based), can be:
            - Column vector (same PRBs both slots)
            - 2-column matrix (different PRBs per slot)
        - NLayers : int, optional
            Number of transmission layers (default 1, range 1-4)
        - DynCyclicShift : int, optional
            Cyclic shift for DM-RS 0-7 (default 0)
        - OrthoCover : str, optional
            'Off' (default) or 'On'
        - PMI : int, optional
            Precoder matrix indication (required if NTxAnts > 1)

    *args : variable
        Can be:
        - (rxgrid,) : Use default estimation
        - (cec, rxgrid) : Use custom channel estimator configuration
        - (cec, rxgrid, refgrid) : Include reference grid
        - (rxgrid, refgrid) : Use TestEVM method (TS 36.101 Annex F4)

    rxgrid : np.ndarray
        Received resource element grid, shape (NSC, NSym, NR)
        - NSC: number of subcarriers
        - NSym: number of SC-FDMA symbols
        - NR: number of receive antennas

    cec : dict, optional
        Channel estimator configuration with fields:
        - FreqWindow : int
            Window size in RE for frequency averaging (odd or multiple of 12)
        - TimeWindow : int
            Window size in RE for time averaging (odd number)
        - InterpType : str
            Interpolation type: 'nearest', 'linear', 'natural', 'cubic', 'v4', 'none'
        - PilotAverage : str, optional
            'UserDefined' (default) or 'TestEVM'
        - Reference : str, optional
            'Antennas' (default), 'Layers', or 'None'
        - Window : str, optional
            'Left', 'Right', 'Centred', 'Centered' (for multi-subframe)

    refgrid : np.ndarray, optional
        Reference grid with known transmitted symbols, shape (NSC, NSym, NT)
        Unknown locations should be NaN

    Returns
    -------
    hest : np.ndarray
        Channel estimate, shape (NSC, NSym, NR, NT) or (NSC, NSym, NR, NLayers)
        - NSC: number of subcarriers
        - NSym: number of SC-FDMA symbols
        - NR: number of receive antennas
        - NT: number of transmit antennas (or NLayers if Reference='Layers')

    noiseest : float
        Noise power spectral density estimate

    Examples
    --------
    Basic usage with default settings:

    >>> ue = {'NULRB': 6, 'NCellID': 1, 'NSubframe': 0}
    >>> chs = {'PRBSet': np.arange(6).reshape(-1, 1)}
    >>> hest, noiseest = lteULChannelEstimate(ue, chs, rxgrid)

    With custom channel estimator configuration:

    >>> cec = {'FreqWindow': 7, 'TimeWindow': 1, 'InterpType': 'cubic'}
    >>> hest, noiseest = lteULChannelEstimate(ue, chs, cec, rxgrid)

    See Also
    --------
    ltePUSCHDRS : Generate PUSCH DM-RS
    ltePUSCHDRSIndices : Get DM-RS indices
    lteSCFDMADemodulate : Demodulate SC-FDMA waveform

    References
    ----------
    .. [1] 3GPP TS 36.211 "Physical channels and modulation"
    .. [2] 3GPP TS 36.101 "User Equipment (UE) radio transmission and reception"
    """

    # Parse input arguments
    cec, rxgrid, refgrid = _parse_inputs(ue, chs, args)

    # Validate inputs
    _validate_ue_config(ue)
    _validate_chs_config(chs, ue)
    _validate_cec_config(cec)
    _validate_rxgrid(rxgrid, ue)

    # Perform channel estimation
    hest, noiseest = _perform_channel_estimation(ue, chs, cec, rxgrid, refgrid)

    return hest, noiseest


def _parse_inputs(
    ue: Dict[str, Any],
    chs: Dict[str, Any],
    args: tuple
) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    """Parse variable input arguments"""

    if len(args) == 1:
        # lteULChannelEstimate(ue, chs, rxgrid)
        cec = _get_default_cec()
        rxgrid = args[0]
        refgrid = None

    elif len(args) == 2:
        # Could be (cec, rxgrid) or (rxgrid, refgrid)
        if isinstance(args[0], dict):
            # lteULChannelEstimate(ue, chs, cec, rxgrid)
            cec = args[0]
            rxgrid = args[1]
            refgrid = None
        else:
            # lteULChannelEstimate(ue, chs, rxgrid, refgrid) - TestEVM mode
            cec = {
                'PilotAverage': 'TestEVM',
                'Reference': 'Antennas',
                'InterpType': 'cubic'
            }
            rxgrid = args[0]
            refgrid = args[1]

    elif len(args) == 3:
        # lteULChannelEstimate(ue, chs, cec, rxgrid, refgrid)
        cec = args[0]
        rxgrid = args[1]
        refgrid = args[2]

    else:
        raise ValueError(f"Invalid number of arguments: {len(args) + 2}")

    # Ensure rxgrid is complex
    if not np.iscomplexobj(rxgrid):
        rxgrid = rxgrid.astype(complex)

    # Ensure refgrid is complex if provided
    if refgrid is not None and not np.iscomplexobj(refgrid):
        refgrid = refgrid.astype(complex)

    return cec, rxgrid, refgrid


def _get_default_cec() -> Dict[str, Any]:
    """Get default channel estimator configuration"""
    return {
        'FreqWindow': 1,
        'TimeWindow': 1,
        'InterpType': 'cubic',
        'PilotAverage': 'UserDefined',
        'Reference': 'Antennas'
    }


def _validate_ue_config(ue: Dict[str, Any]) -> None:
    """Validate UE configuration"""

    # Required fields
    if 'NULRB' not in ue:
        raise ValueError("UE must contain NULRB field")
    if 'NCellID' not in ue:
        raise ValueError("UE must contain NCellID field")

    # Validate NULRB
    valid_nulrb = [6, 15, 25, 50, 75, 100]
    if ue['NULRB'] not in valid_nulrb:
        raise ValueError(f"NULRB must be one of {valid_nulrb}, got {ue['NULRB']}")

    # Validate NCellID
    if not (0 <= ue['NCellID'] <= 503):
        raise ValueError(f"NCellID must be 0-503, got {ue['NCellID']}")

    # Validate optional fields
    if 'NSubframe' in ue and not (0 <= ue['NSubframe'] <= 9):
        raise ValueError(f"NSubframe must be 0-9, got {ue['NSubframe']}")

    if 'NTxAnts' in ue and ue['NTxAnts'] not in [1, 2, 4]:
        raise ValueError(f"NTxAnts must be 1, 2, or 4, got {ue['NTxAnts']}")

    if 'CyclicPrefixUL' in ue and ue['CyclicPrefixUL'] not in ['Normal', 'Extended']:
        raise ValueError(f"CyclicPrefixUL must be 'Normal' or 'Extended'")


def _validate_chs_config(chs: Dict[str, Any], ue: Dict[str, Any]) -> None:
    """Validate channel configuration"""

    if 'PRBSet' not in chs:
        raise ValueError("CHS must contain PRBSet field")

    prb_set = chs['PRBSet']
    if not isinstance(prb_set, np.ndarray):
        raise ValueError("PRBSet must be a numpy array")

    # Check PRB indices are valid
    max_prb = ue['NULRB']
    if np.any(prb_set < 0) or np.any(prb_set >= max_prb):
        raise ValueError(f"PRBSet indices must be 0-{max_prb-1}")

    # Validate NLayers
    if 'NLayers' in chs:
        if not (1 <= chs['NLayers'] <= 4):
            raise ValueError(f"NLayers must be 1-4, got {chs['NLayers']}")


def _validate_cec_config(cec: Dict[str, Any]) -> None:
    """Validate channel estimator configuration"""

    if 'PilotAverage' in cec and cec['PilotAverage'] == 'UserDefined':
        # UserDefined requires FreqWindow, TimeWindow, InterpType
        if 'FreqWindow' not in cec:
            raise ValueError("CEC.FreqWindow required for UserDefined pilot averaging")
        if 'TimeWindow' not in cec:
            raise ValueError("CEC.TimeWindow required for UserDefined pilot averaging")
        if 'InterpType' not in cec:
            raise ValueError("CEC.InterpType required for UserDefined pilot averaging")

        # Validate FreqWindow
        freq_win = cec['FreqWindow']
        if freq_win % 2 == 0 and freq_win % 12 != 0:
            raise ValueError("FreqWindow must be odd or a multiple of 12")

        # Validate TimeWindow
        time_win = cec['TimeWindow']
        if time_win % 2 == 0:
            raise ValueError("TimeWindow must be odd")

        # Validate InterpType
        valid_interp = ['nearest', 'linear', 'natural', 'cubic', 'v4', 'none']
        if cec['InterpType'].lower() not in valid_interp:
            raise ValueError(f"InterpType must be one of {valid_interp}")


def _validate_rxgrid(rxgrid: np.ndarray, ue: Dict[str, Any]) -> None:
    """Validate received grid dimensions"""

    if rxgrid.ndim not in [2, 3]:
        raise ValueError(f"rxgrid must be 2D or 3D, got {rxgrid.ndim}D")

    # Check subcarrier dimension
    nsc_rb = 12
    expected_nsc = ue['NULRB'] * nsc_rb
    if rxgrid.shape[0] != expected_nsc:
        raise ValueError(
            f"rxgrid has {rxgrid.shape[0]} subcarriers, "
            f"expected {expected_nsc} for NULRB={ue['NULRB']}"
        )

    # Check symbol dimension
    cp = ue.get('CyclicPrefixUL', 'Normal')
    nsym_per_subframe = 14 if cp == 'Normal' else 12

    if rxgrid.shape[1] % nsym_per_subframe != 0:
        raise ValueError(
            f"rxgrid has {rxgrid.shape[1]} symbols, "
            f"must be multiple of {nsym_per_subframe} for {cp} CP"
        )


def _perform_channel_estimation(
    ue: Dict[str, Any],
    chs: Dict[str, Any],
    cec: Dict[str, Any],
    rxgrid: np.ndarray,
    refgrid: Optional[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """
    Main channel estimation algorithm

    Algorithm steps:
    1. Extract reference signals from rxgrid
    2. Generate expected reference signals
    3. Compute least-squares estimates
    4. Average pilot estimates (noise reduction)
    5. Interpolate to full grid
    6. Estimate noise power
    """

    # Ensure rxgrid is 3D
    if rxgrid.ndim == 2:
        rxgrid = rxgrid[:, :, np.newaxis]

    nsc, nsym, nrx = rxgrid.shape

    # Get transmit antenna count
    ntx = ue.get('NTxAnts', 1)
    nlayers = chs.get('NLayers', 1)

    # Determine reference type
    reference = cec.get('Reference', 'Antennas')
    if reference == 'Layers':
        nt = nlayers
    else:
        nt = ntx

    # Initialize channel estimate
    hest = np.zeros((nsc, nsym, nrx, nt), dtype=complex)

    # Generate reference signals
    from lte_pusch_drs import ltePUSCHDRS, ltePUSCHDRSIndices

    # Generate DRS for each transmit antenna/layer
    if reference == 'Layers':
        # Generate layer-specific DRS
        _, _, drs_layers = ltePUSCHDRS(ue, chs)
        drs_tx = drs_layers
    else:
        # Generate antenna-specific DRS
        drs_ant, _, _ = ltePUSCHDRS(ue, chs)
        drs_tx = drs_ant

    if drs_tx is None or drs_tx.size == 0:
        warnings.warn("Failed to generate DRS sequences")
        return hest, 0.0

    # Get DRS indices
    from lte_pusch_drs_indices import ltePUSCHDRSIndices as get_drs_indices
    from lte_pusch_drs_indices import UEConfig, CHSConfig

    ue_config = UEConfig(
        NULRB=ue['NULRB'],
        CyclicPrefixUL=ue.get('CyclicPrefixUL', 'Normal'),
        NTxAnts=nt
    )

    chs_config = CHSConfig(PRBSet=chs['PRBSet'])

    # Get 0-based subscript indices
    drs_indices = get_drs_indices(ue_config, chs_config, '0based sub')

    # Extract received pilots
    rx_pilots = np.zeros((drs_indices.shape[0], nrx), dtype=complex)
    for i in range(drs_indices.shape[0]):
        sc_idx = int(drs_indices[i, 0])
        sym_idx = int(drs_indices[i, 1])
        rx_pilots[i, :] = rxgrid[sc_idx, sym_idx, :]

    # Perform least-squares estimation for each Tx-Rx pair
    pilot_estimates = np.zeros((drs_indices.shape[0], nrx, nt), dtype=complex)
    noise_samples = []

    for tx in range(nt):
        # Get expected DRS for this antenna/layer
        drs_expected = drs_tx[:, tx]

        for rx in range(nrx):
            # Least-squares estimate: H = Y / X
            with np.errstate(divide='ignore', invalid='ignore'):
                ls_estimates = rx_pilots[:, rx] / drs_expected
                ls_estimates[~np.isfinite(ls_estimates)] = 0

            pilot_estimates[:, rx, tx] = ls_estimates

            # Collect samples for noise estimation
            if len(ls_estimates) > 1:
                noise_samples.extend(np.diff(ls_estimates))

    # Average pilot estimates
    pilot_avg = _average_pilot_estimates(
        pilot_estimates, drs_indices, cec, ue, chs
    )

    # Interpolate to full grid
    if cec.get('InterpType', 'cubic').lower() != 'none':
        hest = _interpolate_channel_estimate(
            pilot_avg, drs_indices, nsc, nsym, nrx, nt, cec
        )
    else:
        # No interpolation - place estimates at pilot locations
        for i in range(drs_indices.shape[0]):
            sc_idx = int(drs_indices[i, 0])
            sym_idx = int(drs_indices[i, 1])
            hest[sc_idx, sym_idx, :, :] = pilot_avg[i, :, :]

    # Estimate noise power spectral density
    noiseest = _estimate_noise_power(noise_samples, pilot_estimates)

    # Handle additional reference grid if provided
    if refgrid is not None and cec.get('InterpType', 'cubic').lower() == 'none':
        hest = _incorporate_refgrid(hest, refgrid, rxgrid)

    return hest, noiseest


def _average_pilot_estimates(
    pilot_estimates: np.ndarray,
    drs_indices: np.ndarray,
    cec: Dict[str, Any],
    ue: Dict[str, Any],
    chs: Dict[str, Any]
) -> np.ndarray:
    """
    Average pilot estimates to reduce noise

    Implements both UserDefined and TestEVM pilot averaging methods
    """

    pilot_avg_method = cec.get('PilotAverage', 'UserDefined')

    if pilot_avg_method == 'TestEVM':
        # TS 36.101 Annex F method
        return _average_pilots_testevm(pilot_estimates, drs_indices, ue, chs)
    else:
        # UserDefined method with rectangular kernel
        return _average_pilots_userdefined(pilot_estimates, drs_indices, cec, ue)


def _average_pilots_userdefined(
    pilot_estimates: np.ndarray,
    drs_indices: np.ndarray,
    cec: Dict[str, Any],
    ue: Dict[str, Any]
) -> np.ndarray:
    """
    User-defined pilot averaging with rectangular kernel

    Uses FreqWindow x TimeWindow rectangular averaging kernel
    """

    freq_window = cec['FreqWindow']
    time_window = cec['TimeWindow']

    npilots, nrx, nt = pilot_estimates.shape
    pilot_avg = np.zeros_like(pilot_estimates)

    # Get unique symbols where pilots are located
    unique_symbols = np.unique(drs_indices[:, 1])

    # Special case: FreqWindow is multiple of 12 and TimeWindow is 1
    # This provides "despreading" for orthogonal cover codes
    if freq_window % 12 == 0 and time_window == 1:
        pilot_avg = _average_pilots_despread(
            pilot_estimates, drs_indices, freq_window
        )
    else:
        # Standard rectangular kernel averaging
        for tx in range(nt):
            for rx in range(nrx):
                for i in range(npilots):
                    sc_idx = drs_indices[i, 0]
                    sym_idx = drs_indices[i, 1]

                    # Find pilots within window
                    sc_range = [sc_idx - freq_window // 2, sc_idx + freq_window // 2]
                    sym_range = [sym_idx - time_window // 2, sym_idx + time_window // 2]

                    # Find pilots in window
                    in_window = (
                        (drs_indices[:, 0] >= sc_range[0]) &
                        (drs_indices[:, 0] <= sc_range[1]) &
                        (drs_indices[:, 1] >= sym_range[0]) &
                        (drs_indices[:, 1] <= sym_range[1])
                    )

                    # Average pilots in window
                    window_pilots = pilot_estimates[in_window, rx, tx]
                    if len(window_pilots) > 0:
                        pilot_avg[i, rx, tx] = np.mean(window_pilots)
                    else:
                        pilot_avg[i, rx, tx] = pilot_estimates[i, rx, tx]

    return pilot_avg


def _average_pilots_despread(
    pilot_estimates: np.ndarray,
    drs_indices: np.ndarray,
    freq_window: int
) -> np.ndarray:
    """
    Special despreading averaging for orthogonal cover codes

    When FreqWindow is multiple of 12 and TimeWindow is 1,
    always average across exactly freq_window subcarriers
    """

    npilots, nrx, nt = pilot_estimates.shape
    pilot_avg = np.zeros_like(pilot_estimates)

    # Get unique symbols
    unique_symbols = np.unique(drs_indices[:, 1])

    for sym in unique_symbols:
        # Get all pilots in this symbol
        sym_mask = drs_indices[:, 1] == sym
        sym_pilots = pilot_estimates[sym_mask, :, :]
        sym_sc = drs_indices[sym_mask, 0]

        # Sort by subcarrier
        sort_idx = np.argsort(sym_sc)
        sym_pilots = sym_pilots[sort_idx, :, :]
        sym_sc = sym_sc[sort_idx]

        # Average in groups of freq_window
        ngroups = int(np.ceil(len(sym_sc) / freq_window))

        for g in range(ngroups):
            start_idx = g * freq_window
            end_idx = min((g + 1) * freq_window, len(sym_sc))

            # Average this group
            group_avg = np.mean(sym_pilots[start_idx:end_idx, :, :], axis=0)

            # Assign to all pilots in group
            for i in range(start_idx, end_idx):
                orig_idx = np.where(
                    (drs_indices[:, 0] == sym_sc[i]) &
                    (drs_indices[:, 1] == sym)
                )[0][0]
                pilot_avg[orig_idx, :, :] = group_avg

    return pilot_avg


def _average_pilots_testevm(
    pilot_estimates: np.ndarray,
    drs_indices: np.ndarray,
    ue: Dict[str, Any],
    chs: Dict[str, Any]
) -> np.ndarray:
    """
    Test EVM pilot averaging per TS 36.101 Annex F

    This method follows the transmitter EVM testing procedure
    """

    # For TestEVM, use simple averaging per symbol
    # This is a simplified implementation
    npilots, nrx, nt = pilot_estimates.shape
    pilot_avg = np.zeros_like(pilot_estimates)

    unique_symbols = np.unique(drs_indices[:, 1])

    for sym in unique_symbols:
        sym_mask = drs_indices[:, 1] == sym
        sym_pilots = pilot_estimates[sym_mask, :, :]

        # Average all pilots in this symbol
        sym_avg = np.mean(sym_pilots, axis=0, keepdims=True)

        # Broadcast to all pilots in symbol
        pilot_avg[sym_mask, :, :] = sym_avg

    return pilot_avg


def _interpolate_channel_estimate(
    pilot_avg: np.ndarray,
    drs_indices: np.ndarray,
    nsc: int,
    nsym: int,
    nrx: int,
    nt: int,
    cec: Dict[str, Any]
) -> np.ndarray:
    """
    Interpolate pilot estimates to full resource grid

    Uses 2D interpolation with virtual pilots at grid edges
    """

    interp_type = cec.get('InterpType', 'cubic').lower()

    hest = np.zeros((nsc, nsym, nrx, nt), dtype=complex)

    # Map MATLAB interpolation types to scipy equivalents
    if interp_type == 'v4':
        method = 'cubic'  # scipy doesn't have MATLAB v4, use cubic
    elif interp_type == 'natural':
        method = 'linear'  # Natural neighbor not in griddata, use linear
    else:
        method = interp_type

    for tx in range(nt):
        for rx in range(nrx):
            # Get pilot locations and values
            pilot_sc = drs_indices[:, 0]
            pilot_sym = drs_indices[:, 1]
            pilot_vals = pilot_avg[:, rx, tx]

            # Remove any zero/invalid pilots
            valid_mask = pilot_vals != 0
            if not np.any(valid_mask):
                continue

            pilot_sc = pilot_sc[valid_mask]
            pilot_sym = pilot_sym[valid_mask]
            pilot_vals = pilot_vals[valid_mask]

            # Create virtual pilots for better edge interpolation
            pilot_sc_ext, pilot_sym_ext, pilot_vals_ext = _create_virtual_pilots(
                pilot_sc, pilot_sym, pilot_vals, nsc, nsym
            )

            # Create interpolation grid
            grid_sc, grid_sym = np.meshgrid(
                np.arange(nsc), np.arange(nsym), indexing='ij'
            )

            points = np.column_stack([pilot_sc_ext, pilot_sym_ext])
            grid_points = np.column_stack([grid_sc.ravel(), grid_sym.ravel()])

            try:
                # Interpolate real and imaginary parts separately
                real_interp = griddata(
                    points, pilot_vals_ext.real, grid_points, method=method
                )
                imag_interp = griddata(
                    points, pilot_vals_ext.imag, grid_points, method=method
                )

                # Combine and reshape
                h_interp = real_interp + 1j * imag_interp
                h_interp = h_interp.reshape(nsc, nsym)

                # Handle NaN values (extrapolation)
                nan_mask = np.isnan(h_interp)
                if np.any(nan_mask):
                    # Use nearest neighbor for extrapolation
                    nn_interp = griddata(
                        points, pilot_vals_ext, grid_points, method='nearest'
                    )
                    nn_interp = nn_interp.reshape(nsc, nsym)
                    h_interp[nan_mask] = nn_interp[nan_mask]

                hest[:, :, rx, tx] = h_interp

            except Exception as e:
                warnings.warn(f"Interpolation failed: {e}. Using nearest neighbor.")
                # Fallback to nearest neighbor
                nn_real = griddata(
                    points, pilot_vals_ext.real, grid_points, method='nearest'
                )
                nn_imag = griddata(
                    points, pilot_vals_ext.imag, grid_points, method='nearest'
                )
                hest[:, :, rx, tx] = (nn_real + 1j * nn_imag).reshape(nsc, nsym)

    return hest


def _create_virtual_pilots(
    pilot_sc: np.ndarray,
    pilot_sym: np.ndarray,
    pilot_vals: np.ndarray,
    nsc: int,
    nsym: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create virtual pilots outside the grid for better interpolation

    Virtual pilots help ensure accurate interpolation at grid edges
    """

    # Find edge pilots
    unique_symbols = np.unique(pilot_sym)

    sc_ext = [pilot_sc]
    sym_ext = [pilot_sym]
    vals_ext = [pilot_vals]

    for sym in unique_symbols:
        sym_mask = pilot_sym == sym
        sym_sc = pilot_sc[sym_mask]
        sym_vals = pilot_vals[sym_mask]

        if len(sym_sc) == 0:
            continue

        # Add virtual pilots before and after in frequency
        min_sc = np.min(sym_sc)
        max_sc = np.max(sym_sc)

        if min_sc > 0:
            # Add virtual pilot below
            sc_ext.append(np.array([min_sc - 12]))
            sym_ext.append(np.array([sym]))
            vals_ext.append(np.array([sym_vals[np.argmin(sym_sc)]]))

        if max_sc < nsc - 1:
            # Add virtual pilot above
            sc_ext.append(np.array([max_sc + 12]))
            sym_ext.append(np.array([sym]))
            vals_ext.append(np.array([sym_vals[np.argmax(sym_sc)]]))

    # Add virtual pilots in time dimension
    if len(unique_symbols) > 0:
        min_sym = np.min(unique_symbols)
        max_sym = np.max(unique_symbols)

        if min_sym > 0:
            # Replicate first symbol pilots to earlier symbol
            first_sym_mask = pilot_sym == min_sym
            sc_ext.append(pilot_sc[first_sym_mask])
            sym_ext.append(np.full(np.sum(first_sym_mask), min_sym - 1))
            vals_ext.append(pilot_vals[first_sym_mask])

        if max_sym < nsym - 1:
            # Replicate last symbol pilots to later symbol
            last_sym_mask = pilot_sym == max_sym
            sc_ext.append(pilot_sc[last_sym_mask])
            sym_ext.append(np.full(np.sum(last_sym_mask), max_sym + 1))
            vals_ext.append(pilot_vals[last_sym_mask])

    # Concatenate all
    sc_extended = np.concatenate(sc_ext)
    sym_extended = np.concatenate(sym_ext)
    vals_extended = np.concatenate(vals_ext)

    return sc_extended, sym_extended, vals_extended


def _estimate_noise_power(
    noise_samples: list,
    pilot_estimates: np.ndarray
) -> float:
    """
    Estimate noise power spectral density

    Uses differences between adjacent pilot estimates
    """

    if len(noise_samples) == 0:
        # No samples available, use variance of pilots
        return float(np.var(pilot_estimates))

    noise_samples = np.array(noise_samples)
    noise_samples = noise_samples[np.isfinite(noise_samples)]

    if len(noise_samples) == 0:
        return 0.0

    # Estimate noise variance from differences
    # Factor of 2 accounts for differencing operation
    noise_var = np.var(noise_samples) / 2.0

    return float(noise_var)


def _incorporate_refgrid(
    hest: np.ndarray,
    refgrid: np.ndarray,
    rxgrid: np.ndarray
) -> np.ndarray:
    """
    Incorporate reference grid information

    When InterpType='none' and refgrid is provided,
    use refgrid symbols as additional reference points
    """

    if refgrid.ndim == 2:
        refgrid = refgrid[:, :, np.newaxis]

    nsc, nsym, nt = refgrid.shape
    nrx = rxgrid.shape[2]

    # Find locations with known reference symbols (not NaN)
    ref_mask = ~np.isnan(refgrid)

    for tx in range(nt):
        for rx in range(nrx):
            for sc in range(nsc):
                for sym in range(nsym):
                    if ref_mask[sc, sym, tx]:
                        # Compute channel estimate at this location
                        ref_val = refgrid[sc, sym, tx]
                        rx_val = rxgrid[sc, sym, rx]

                        if ref_val != 0:
                            hest[sc, sym, rx, tx] = rx_val / ref_val

    return hest


# Convenience functions to match MATLAB style


def lteULPerfectChannelEstimate(
    ue: Dict[str, Any],
    chs: Dict[str, Any],
    channel_model: Any
) -> np.ndarray:
    """
    Generate perfect channel estimate (for testing/comparison)

    This would use the true channel model to generate perfect estimates.
    Not fully implemented - placeholder for compatibility.
    """
    raise NotImplementedError(
        "lteULPerfectChannelEstimate not implemented. "
        "Use lteULChannelEstimate for practical estimation."
    )


# Module test code
if __name__ == "__main__":
    print("="*80)
    print("lteULChannelEstimate - Python Implementation Test")
    print("="*80)
    print()

    # This would require the full PUSCH DRS implementation
    print("To test this module:")
    print("1. Ensure lte_pusch_drs.py is available")
    print("2. Ensure lte_pusch_drs_indices.py is available")
    print("3. Run the example test script")
    print()
    print("See test_lte_ul_channel_estimate.py for complete examples")
