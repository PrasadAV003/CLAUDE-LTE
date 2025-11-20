"""
LTE Uplink Resource Grid Utilities
===================================

Functions for creating and sizing uplink resource grids.
Compatible with MATLAB LTE Toolbox.

Author: CLAUDE-LTE Project
Date: 2025-11-20
"""

import numpy as np


def lteULResourceGridSize(ue, p=None):
    """
    Calculate the size of the uplink resource grid.

    Parameters
    ----------
    ue : dict
        UE-specific settings with fields:
        - NULRB: Number of uplink resource blocks (6-110)
        - CyclicPrefixUL: 'Normal' or 'Extended' (default: 'Normal')
        - NTxAnts: Number of transmission antennas (1, 2, or 4) (default: 1)
    p : int, optional
        Number of antenna planes (overrides NTxAnts if provided)

    Returns
    -------
    tuple
        (N, M, P) where:
        - N: Number of subcarriers (12 * NULRB)
        - M: Number of SC-FDMA symbols per subframe
        - P: Number of transmission antennas

    Examples
    --------
    >>> ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal', 'NTxAnts': 1}
    >>> lteULResourceGridSize(ue)
    (72, 14, 1)

    >>> ue = {'NULRB': 50}
    >>> lteULResourceGridSize(ue)
    (600, 14, 1)

    >>> ue = {'NULRB': 25, 'CyclicPrefixUL': 'Extended', 'NTxAnts': 2}
    >>> lteULResourceGridSize(ue)
    (300, 12, 2)
    """
    # Extract NULRB
    NULRB = ue.get('NULRB')
    if NULRB is None:
        raise ValueError("NULRB must be specified in ue structure")

    if not (6 <= NULRB <= 110):
        raise ValueError("NULRB must be between 6 and 110")

    # Number of subcarriers: N = 12 * NULRB
    N = 12 * NULRB

    # Get cyclic prefix type
    cyclic_prefix = ue.get('CyclicPrefixUL', 'Normal')

    # Number of SC-FDMA symbols per subframe
    # M = 14 for Normal CP, 12 for Extended CP
    if cyclic_prefix == 'Normal':
        M = 14
    elif cyclic_prefix == 'Extended':
        M = 12
    else:
        raise ValueError("CyclicPrefixUL must be 'Normal' or 'Extended'")

    # Number of transmission antennas
    if p is not None:
        P = p
    else:
        P = ue.get('NTxAnts', 1)

    if P not in [1, 2, 4]:
        raise ValueError("Number of transmission antennas must be 1, 2, or 4")

    return (N, M, P)


def lteULResourceGrid(ue, p=None):
    """
    Create an empty uplink subframe resource array.

    Parameters
    ----------
    ue : dict
        UE-specific settings
    p : int, optional
        Number of antenna planes (overrides NTxAnts if provided)

    Returns
    -------
    ndarray
        Empty resource grid of shape (N, M, P) with complex dtype

    Examples
    --------
    >>> ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal', 'NTxAnts': 1}
    >>> reGrid = lteULResourceGrid(ue)
    >>> reGrid.shape
    (72, 14, 1)

    >>> ue = {'NULRB': 50, 'NTxAnts': 2}
    >>> reGrid = lteULResourceGrid(ue)
    >>> reGrid.shape
    (600, 14, 2)
    """
    grid_size = lteULResourceGridSize(ue, p)
    reGrid = np.zeros(grid_size, dtype=complex)
    return reGrid


if __name__ == '__main__':
    # Quick test
    print("LTE Uplink Resource Grid Utilities")
    print("="*50)

    ue = {'NULRB': 6, 'CyclicPrefixUL': 'Normal', 'NTxAnts': 1}
    size = lteULResourceGridSize(ue)
    print(f"\nTest 1: {ue}")
    print(f"Size: {size}")

    ue = {'NULRB': 50, 'CyclicPrefixUL': 'Extended', 'NTxAnts': 2}
    grid = lteULResourceGrid(ue)
    print(f"\nTest 2: {ue}")
    print(f"Grid shape: {grid.shape}")

    print("\n✓ Tests complete!")
