"""
LTE Channel Coding Chain - Complete Decoding Implementation
Python equivalent of MATLAB LTE decoding functions

This module contains the complete decoding chain:
1. lteRateRecoverTurbo - Rate recovery with HARQ soft combining
2. lteTurboDecode - Max-Log-MAP turbo decoding
3. lteCodeBlockDesegment - Code block desegmentation with CRC checking
4. lteCRCDecode - CRC decoding and removal

Based on 3GPP TS 36.212 Sections 5.1.1, 5.1.2, 5.1.3, 5.1.4
"""

import numpy as np
from numba import jit
from typing import Union, List, Tuple, Optional


# ============================================================================
# CRC DECODE - Section 5.1.1
# ============================================================================

class LTE_CRC_Decoder:
    """LTE CRC Decoding"""

    def __init__(self):
        # CRC Generator Polynomials (MSB first)
        self.gCRC24A = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
        self.gCRC24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
        self.gCRC16 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.gCRC8 = [1, 1, 0, 0, 1, 1, 0, 1, 1]

    def crc_calculate(self, input_bits, generator_poly, L):
        """Calculate CRC parity bits"""
        input_bits = np.array(input_bits)
        input_for_crc = np.where(input_bits < 0, 0, input_bits).astype(int)

        poly = np.concatenate([input_for_crc, np.zeros(L, dtype=int)])

        for i in range(len(input_for_crc)):
            if poly[i] == 1:
                for j in range(len(generator_poly)):
                    poly[i + j] = (poly[i + j] + generator_poly[j]) % 2

        return poly[-L:].astype(int)


def lteCRCDecode(blkcrc, poly, mask=0):
    """
    CRC decoding and removal - MATLAB compatible

    Returns:
        blk: Data bits without CRC (int8 array)
        err: XOR difference between received and calculated CRC (uint32)
             err == 0: CRC passed
             err != 0: CRC failed or was masked
    """
    crc_lengths = {'8': 8, '16': 16, '24A': 24, '24B': 24}
    if poly not in crc_lengths:
        raise ValueError(f"Invalid CRC polynomial: {poly}")

    L = crc_lengths[poly]

    # Convert to numpy array and make hard decisions if needed
    blkcrc = np.array(blkcrc, dtype=float)

    # Handle soft values (LLRs)
    if np.any((blkcrc < 0) | (blkcrc > 1)):
        blkcrc_hard = np.where(blkcrc >= 0, 0, 1).astype(np.int8)
    else:
        blkcrc_hard = blkcrc.astype(np.int8)

    if len(blkcrc_hard) < L:
        raise ValueError(f"Input too short for {poly} CRC")

    # Split data and received CRC
    data_bits = blkcrc_hard[:-L]
    received_crc_bits = blkcrc_hard[-L:]

    # Calculate expected CRC
    decoder = LTE_CRC_Decoder()
    poly_map = {'8': decoder.gCRC8, '16': decoder.gCRC16,
                '24A': decoder.gCRC24A, '24B': decoder.gCRC24B}
    calculated_crc_bits = decoder.crc_calculate(data_bits, poly_map[poly], L)

    # Convert to integers for XOR
    received_crc_int = int(''.join(str(b) for b in received_crc_bits), 2)
    calculated_crc_int = int(''.join(str(b) for b in calculated_crc_bits), 2)

    # Compute XOR difference and apply mask
    crc_diff = received_crc_int ^ calculated_crc_int
    err = np.uint32(crc_diff ^ mask)

    return data_bits, err


# ============================================================================
# CODE BLOCK DESEGMENTATION - Section 5.1.2
# ============================================================================

def get_segmentation_params(blklen: int) -> dict:
    """Calculate code block segmentation parameters"""
    K_table = [
        40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
        168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272,
        280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384,
        392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496,
        504, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704,
        720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928,
        944, 960, 976, 992, 1008, 1024, 1056, 1088, 1120, 1152, 1184, 1216,
        1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600,
        1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984,
        2016, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688,
        2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456,
        3520, 3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096, 4160, 4224,
        4288, 4352, 4416, 4480, 4544, 4608, 4672, 4736, 4800, 4864, 4928, 4992,
        5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504, 5568, 5632, 5696, 5760,
        5824, 5888, 5952, 6016, 6080, 6144
    ]

    Z = 6144
    B = blklen

    if B <= Z:
        L = 0
        C = 1
        B_prime = B
        K_plus = min([k for k in K_table if k >= B_prime], default=6144)
        K_minus = 0
        C_plus = 1
        C_minus = 0
    else:
        L = 24
        C = int(np.ceil(B / (Z - L)))
        B_prime = B + C * L
        K_plus = min([k for k in K_table if C * k >= B_prime], default=6144)
        K_minus_candidates = [k for k in K_table if k < K_plus]
        K_minus = max(K_minus_candidates) if K_minus_candidates else 0

        if K_minus > 0:
            K_delta = K_plus - K_minus
            C_minus = int(np.floor((C * K_plus - B_prime) / K_delta))
            C_plus = C - C_minus
        else:
            C_plus = C
            C_minus = 0

    F = C_plus * K_plus + C_minus * K_minus - B_prime

    return {
        'C': C, 'K_plus': K_plus, 'K_minus': K_minus,
        'C_plus': C_plus, 'C_minus': C_minus, 'F': F, 'L': L
    }


def lteCodeBlockDesegment(cbs: Union[np.ndarray, List[np.ndarray]],
                          blklen: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Code block desegmentation with CRC checking

    Returns:
        blk: Desegmented output block (int8)
        err: CRC error indicators (int8), 0=pass, 1=fail
    """
    # Convert to list format
    if isinstance(cbs, np.ndarray):
        code_blocks = [cbs]
    else:
        code_blocks = cbs

    C = len(code_blocks)

    # Determine parameters
    if blklen is not None and blklen > 0:
        params = get_segmentation_params(blklen)
        F = params['F']
        L = params['L']
        K_plus = params['K_plus']
        K_minus = params['K_minus']
        C_minus = params['C_minus']
    else:
        F = 0
        L = 24 if C > 1 else 0
        if C > 1:
            sizes = [len(cb) for cb in code_blocks]
            K_plus = max(sizes)
            K_minus = min(sizes)
            C_minus = sum(1 for s in sizes if s == K_minus)
        else:
            K_plus = len(code_blocks[0])
            K_minus = 0
            C_minus = 0

    # Process code blocks
    output_bits = []
    crc_errors = []

    for r in range(C):
        cb = np.array(code_blocks[r], dtype=np.int8)

        # Determine size
        if C_minus > 0 and r < C_minus:
            K_r = K_minus
        else:
            K_r = K_plus

        if len(cb) != K_r:
            raise ValueError(f"Code block {r} has length {len(cb)}, expected {K_r}")

        # Remove filler bits
        start_idx = F if r == 0 else 0

        # Check and remove CRC if multiple blocks
        if C > 1:
            data_with_crc = cb[start_idx:K_r]
            decoded, err = lteCRCDecode(data_with_crc, '24B')
            crc_errors.append(0 if err == 0 else 1)
            output_bits.extend(decoded.tolist())
        else:
            data_bits = cb[start_idx:]
            output_bits.extend(data_bits.tolist())

    blk = np.array(output_bits, dtype=np.int8)

    if C > 1:
        err = np.array(crc_errors, dtype=np.int8)
    else:
        err = np.array([], dtype=np.int8)

    return blk, err


# ============================================================================
# TURBO DECODE - Section 5.1.3
# ============================================================================

# Trellis structure for LTE RSC encoder
NEXT_STATES = np.array([
    [0, 4], [0, 4], [5, 1], [5, 1],
    [6, 2], [6, 2], [3, 7], [3, 7]
], dtype=np.int32)

OUTPUTS = np.array([
    [0, 3], [1, 2], [1, 2], [0, 3],
    [0, 3], [1, 2], [1, 2], [0, 3]
], dtype=np.int32)


@jit(nopython=True)
def _siso_decode_maxlog_numba(sys_full: np.ndarray, par_full: np.ndarray,
                               apr_full: np.ndarray, K: int) -> np.ndarray:
    """Max-Log-MAP SISO decoder with Numba optimization"""
    N = K + 4

    # Initialize metrics
    alpha = np.full((N + 1, 8), -np.inf, dtype=np.float64)
    alpha[0, 0] = 0.0

    beta = np.full((N + 1, 8), -np.inf, dtype=np.float64)
    beta[N, 0] = 0.0

    # Precompute branch metrics
    gamma = np.zeros((N, 8, 2), dtype=np.float64)

    for t in range(N):
        sys_t = sys_full[t]
        par_t = par_full[t]
        apr_t = apr_full[t]

        for s in range(8):
            for input_bit in range(2):
                next_s = NEXT_STATES[s, input_bit]
                output_bits = OUTPUTS[s, input_bit]

                sys_out = (output_bits >> 1) & 1
                par_out = output_bits & 1

                sys_contrib = sys_t * (1.0 - 2.0 * sys_out)
                par_contrib = par_t * (1.0 - 2.0 * par_out)
                apr_contrib = apr_t * (1.0 - 2.0 * input_bit)

                gamma[t, s, input_bit] = sys_contrib + par_contrib + apr_contrib

    # Forward pass
    for t in range(N):
        for s in range(8):
            for input_bit in range(2):
                next_s = NEXT_STATES[s, input_bit]
                metric = alpha[t, s] + gamma[t, s, input_bit]
                alpha[t + 1, next_s] = max(alpha[t + 1, next_s], metric)

    # Backward pass
    for t in range(N - 1, -1, -1):
        for s in range(8):
            for input_bit in range(2):
                next_s = NEXT_STATES[s, input_bit]
                metric = beta[t + 1, next_s] + gamma[t, s, input_bit]
                beta[t, s] = max(beta[t, s], metric)

    # Compute LLRs
    llr_out = np.zeros(K, dtype=np.float64)

    for t in range(K):
        max_0 = -np.inf
        max_1 = -np.inf

        for s in range(8):
            next_s_0 = NEXT_STATES[s, 0]
            metric_0 = alpha[t, s] + gamma[t, s, 0] + beta[t + 1, next_s_0]
            max_0 = max(max_0, metric_0)

            next_s_1 = NEXT_STATES[s, 1]
            metric_1 = alpha[t, s] + gamma[t, s, 1] + beta[t + 1, next_s_1]
            max_1 = max(max_1, metric_1)

        total_llr = max_0 - max_1
        llr_out[t] = total_llr - apr_full[t]

    return llr_out


class QPPInterleaver:
    """QPP Interleaver for LTE Turbo Codes"""

    def __init__(self):
        self.params = {
            40: (3, 10), 48: (7, 12), 56: (19, 42), 64: (7, 16), 72: (7, 18),
            80: (11, 20), 88: (5, 22), 96: (11, 24), 104: (7, 26), 112: (41, 84),
            120: (103, 90), 128: (15, 32), 136: (9, 34), 144: (17, 108), 152: (9, 38),
            160: (21, 120), 168: (101, 84), 176: (21, 44), 184: (57, 46), 192: (23, 48),
            200: (13, 50), 208: (27, 52), 216: (11, 36), 224: (27, 56), 232: (85, 58),
            240: (29, 60), 248: (33, 62), 256: (15, 32), 264: (17, 198), 272: (33, 68),
            280: (103, 210), 288: (19, 36), 296: (19, 74), 304: (37, 76), 312: (19, 78),
            320: (21, 120), 328: (21, 82), 336: (115, 84), 344: (193, 86), 352: (21, 44),
            360: (133, 90), 368: (81, 46), 376: (45, 94), 384: (23, 48), 392: (243, 98),
            400: (151, 40), 408: (155, 102), 416: (25, 52), 424: (51, 106), 432: (47, 72),
            440: (91, 110), 448: (29, 168), 456: (29, 114), 464: (247, 58), 472: (29, 118),
            480: (89, 180), 488: (91, 122), 496: (157, 62), 504: (55, 84), 512: (31, 64),
            528: (17, 66), 544: (35, 68), 560: (227, 420), 576: (65, 96), 592: (19, 74),
            608: (37, 76), 624: (41, 234), 640: (39, 80), 656: (185, 82), 672: (43, 252),
            688: (21, 86), 704: (155, 44), 720: (79, 120), 736: (139, 92), 752: (23, 94),
            768: (217, 48), 784: (25, 98), 800: (17, 80), 816: (127, 102), 832: (25, 52),
            848: (239, 106), 864: (17, 48), 880: (137, 110), 896: (215, 112), 912: (29, 114),
            928: (15, 58), 944: (147, 118), 960: (29, 60), 976: (59, 122), 992: (65, 124),
            1008: (55, 84), 1024: (31, 64), 1056: (17, 66), 1088: (171, 204), 1120: (67, 140),
            1152: (35, 72), 1184: (19, 74), 1216: (39, 76), 1248: (19, 78), 1280: (199, 240),
            1312: (21, 82), 1344: (211, 252), 1376: (21, 86), 1408: (43, 88), 1440: (149, 60),
            1472: (45, 92), 1504: (49, 846), 1536: (71, 48), 1568: (13, 28), 1600: (17, 80),
            1632: (25, 102), 1664: (183, 104), 1696: (55, 954), 1728: (127, 96), 1760: (27, 110),
            1792: (29, 112), 1824: (29, 114), 1856: (57, 116), 1888: (45, 354), 1920: (31, 120),
            1952: (59, 610), 1984: (185, 124), 2016: (113, 420), 2048: (31, 64), 2112: (17, 66),
            2176: (171, 136), 2240: (209, 420), 2304: (253, 216), 2368: (367, 444), 2432: (265, 456),
            2496: (181, 468), 2560: (39, 80), 2624: (27, 164), 2688: (127, 504), 2752: (143, 172),
            2816: (43, 88), 2880: (29, 300), 2944: (45, 92), 3008: (157, 188), 3072: (47, 96),
            3136: (13, 28), 3200: (111, 240), 3264: (443, 204), 3328: (51, 104), 3392: (51, 212),
            3456: (451, 192), 3520: (257, 220), 3584: (57, 336), 3648: (313, 228), 3712: (271, 232),
            3776: (179, 236), 3840: (331, 120), 3904: (363, 244), 3968: (375, 248), 4032: (127, 168),
            4096: (31, 64), 4160: (33, 130), 4224: (43, 264), 4288: (33, 134), 4352: (477, 408),
            4416: (35, 138), 4480: (233, 280), 4544: (357, 142), 4608: (337, 480), 4672: (37, 146),
            4736: (71, 444), 4800: (71, 120), 4864: (37, 152), 4928: (39, 462), 4992: (127, 234),
            5056: (39, 158), 5120: (39, 80), 5184: (31, 96), 5248: (113, 902), 5312: (41, 166),
            5376: (251, 336), 5440: (43, 170), 5504: (21, 86), 5568: (43, 174), 5632: (45, 176),
            5696: (45, 178), 5760: (161, 120), 5824: (89, 182), 5888: (323, 184), 5952: (47, 186),
            6016: (23, 94), 6080: (47, 190), 6144: (263, 480)
        }

    def interleave(self, data: np.ndarray, K: int) -> np.ndarray:
        if K not in self.params:
            raise ValueError(f"Unsupported interleaver size K={K}")
        f1, f2 = self.params[K]
        output = np.zeros_like(data)
        for i in range(K):
            pi_i = (f1 * i + f2 * i * i) % K
            output[pi_i] = data[i]
        return output

    def deinterleave(self, data: np.ndarray, K: int) -> np.ndarray:
        if K not in self.params:
            raise ValueError(f"Unsupported interleaver size K={K}")
        f1, f2 = self.params[K]
        output = np.zeros_like(data)
        for i in range(K):
            pi_i = (f1 * i + f2 * i * i) % K
            output[i] = data[pi_i]
        return output


class LTE_TurboDecoder:
    """LTE Turbo Decoder with Max-Log-MAP"""

    def __init__(self):
        self.interleaver = QPPInterleaver()

    def decode(self, soft_input: np.ndarray, K: int, num_iterations: int = 5) -> np.ndarray:
        """Turbo decode soft input data"""
        N = K + 4

        # Split input: [S | P1 | P2]
        sys_llr = soft_input[0:N].copy()
        par1_llr = soft_input[N:2*N].copy()
        par2_llr = soft_input[2*N:3*N].copy()

        # Initialize a priori
        apr1 = np.zeros(N, dtype=np.float64)
        apr2 = np.zeros(N, dtype=np.float64)

        # Iterative decoding
        for iteration in range(num_iterations):
            # Decoder 1
            ext1 = _siso_decode_maxlog_numba(sys_llr, par1_llr, apr1, K)

            ext1_full = np.zeros(N, dtype=np.float64)
            ext1_full[:K] = ext1

            # Interleave for decoder 2
            apr2_interleaved = self.interleaver.interleave(ext1_full, K)

            sys_llr_interleaved = self.interleaver.interleave(sys_llr[:K], K)
            sys_llr_int_full = np.zeros(N, dtype=np.float64)
            sys_llr_int_full[:K] = sys_llr_interleaved
            sys_llr_int_full[K:] = sys_llr[K:]

            # Decoder 2
            ext2 = _siso_decode_maxlog_numba(sys_llr_int_full, par2_llr, apr2_interleaved, K)

            ext2_full = np.zeros(N, dtype=np.float64)
            ext2_full[:K] = ext2

            # Deinterleave
            apr1_deinterleaved = self.interleaver.deinterleave(ext2_full, K)
            apr1 = apr1_deinterleaved
            apr2 = apr2_interleaved

        # Final decision
        ext1_final = _siso_decode_maxlog_numba(sys_llr, par1_llr, apr1, K)

        decoded = np.zeros(K, dtype=np.int8)
        for i in range(K):
            total_llr = sys_llr[i] + ext1_final[i]
            decoded[i] = 0 if total_llr >= 0 else 1

        return decoded


def lteTurboDecode(in_data: Union[np.ndarray, List[np.ndarray]],
                   nturbodecits: int = 5) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Turbo decoding with Max-Log-MAP algorithm

    Parameters:
        in_data: Soft bit input (LLRs) in [S P1 P2] format
        nturbodecits: Number of iterations (1-30)

    Returns:
        Decoded bits (int8)
    """
    if nturbodecits < 1 or nturbodecits > 30:
        raise ValueError(f"Iterations must be 1-30, got {nturbodecits}")

    decoder = LTE_TurboDecoder()

    if isinstance(in_data, list):
        result = []
        for code_block in in_data:
            if len(code_block) == 0:
                result.append(np.array([], dtype=np.int8))
                continue

            soft_block = np.array(code_block, dtype=float)

            if len(soft_block) % 3 != 0:
                raise ValueError(f"Length must be multiple of 3")

            D = len(soft_block) // 3
            K = D - 4

            if K < 40 or K > 6144:
                raise ValueError(f"Invalid K={K}")

            decoded = decoder.decode(soft_block, K, nturbodecits)
            result.append(decoded)

        return result
    else:
        soft_data = np.array(in_data, dtype=float)

        if len(soft_data) == 0:
            return np.array([], dtype=np.int8)

        if len(soft_data) % 3 != 0:
            raise ValueError(f"Length must be multiple of 3")

        D = len(soft_data) // 3
        K = D - 4

        if K < 40 or K > 6144:
            raise ValueError(f"Invalid K={K}")

        decoded = decoder.decode(soft_data, K, nturbodecits)

        return decoded


# ============================================================================
# RATE RECOVERY - Section 5.1.4
# ============================================================================

class LTE_RateRecovery:
    """LTE Rate Recovery with HARQ Support"""

    def __init__(self):
        self.sub_block_interleaver_pattern = np.array([
            0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
            1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31
        ], dtype=int)

    def sub_block_deinterleaver(self, d: np.ndarray) -> np.ndarray:
        """Inverse of sub-block interleaver"""
        D = len(d)
        if D == 0:
            return d

        R_sb = 32
        C_sb = int(np.ceil(D / R_sb))
        N_dummy = R_sb * C_sb - D

        y = np.full(R_sb * C_sb, np.nan, dtype=float)
        y[N_dummy:] = d

        matrix = y.reshape(C_sb, R_sb).T

        inv_pattern = np.argsort(self.sub_block_interleaver_pattern)
        deinterleaved_matrix = matrix[inv_pattern, :]

        output = deinterleaved_matrix.T.flatten()
        output = output[~np.isnan(output)]

        return output

    def inverse_bit_selection(self, e_bits: np.ndarray, K_w: int, rv: int,
                              R_subblock: int) -> np.ndarray:
        """Inverse of bit selection"""
        w = np.zeros(K_w, dtype=float)

        # Circular buffer parameters
        N_cb = K_w
        k = 0
        j = 0

        # Starting point based on RV
        if rv == 0:
            k_0 = 0
        elif rv == 1:
            k_0 = int(np.floor(R_subblock * (3/4))) * 3
        elif rv == 2:
            k_0 = 0
        else:  # rv == 3
            k_0 = int(np.floor(R_subblock * (3/4))) * 3

        k = k_0

        # Fill circular buffer
        while j < len(e_bits):
            w[k % N_cb] = e_bits[j]
            k += 1
            j += 1

        return w

    def rate_recover_code_block(self, e_bits: np.ndarray, K: int, rv: int,
                                cbsbuffer: Optional[np.ndarray] = None) -> np.ndarray:
        """Rate recovery for single code block"""
        D = K + 4
        K_pi = 3 * D

        # Step 1: Determine R_subblock
        R_subblock = int(np.ceil(K_pi / 32))
        K_w = 3 * R_subblock * 32

        # Step 2: Inverse bit selection
        w_all = self.inverse_bit_selection(e_bits, K_w, rv, R_subblock)

        # Step 3: Split into three streams
        w0 = w_all[0::3]
        w1 = w_all[1::3]
        w2 = w_all[2::3]

        # Step 4: Sub-block deinterleaving
        v0 = self.sub_block_deinterleaver(w0)
        v1 = self.sub_block_deinterleaver(w1)
        v2 = self.sub_block_deinterleaver(w2)

        # Reconstruct output
        output = np.zeros(K_pi, dtype=float)
        output[0:D] = v0[0:D]
        output[D:2*D] = v1[0:D]
        output[2*D:3*D] = v2[0:D]

        # HARQ soft combining
        if cbsbuffer is not None and len(cbsbuffer) > 0:
            output = output + cbsbuffer.astype(float)

        return output


def lteRateRecoverTurbo(in_data: Union[np.ndarray, List[np.ndarray]],
                        trblklen: int,
                        rv: Union[int, List[int]] = 0,
                        cbsbuffers: Optional[Union[np.ndarray, List[np.ndarray]]] = None) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Rate recovery for turbo coded data with HARQ support

    Parameters:
        in_data: Rate matched bits (soft values/LLRs)
        trblklen: Transport block length
        rv: Redundancy version (0-3)
        cbsbuffers: Previous soft buffers for HARQ combining

    Returns:
        Rate recovered soft bits
    """
    recovery = LTE_RateRecovery()

    # Get segmentation parameters
    params = get_segmentation_params(trblklen)
    C = params['C']

    # Handle inputs
    if isinstance(in_data, list):
        code_blocks = in_data
    else:
        # Split into code blocks
        total_len = len(in_data)
        block_len = total_len // C
        code_blocks = [in_data[i*block_len:(i+1)*block_len] for i in range(C)]

    if isinstance(rv, int):
        rv_list = [rv] * C
    else:
        rv_list = rv

    if cbsbuffers is None:
        buffer_list = [None] * C
    elif isinstance(cbsbuffers, list):
        buffer_list = cbsbuffers
    else:
        buffer_list = [cbsbuffers]

    # Process each code block
    result = []
    for r in range(C):
        if C_minus := params['C_minus']:
            K = params['K_minus'] if r < C_minus else params['K_plus']
        else:
            K = params['K_plus']

        e_bits = np.array(code_blocks[r], dtype=float)
        recovered = recovery.rate_recover_code_block(e_bits, K, rv_list[r], buffer_list[r])
        result.append(recovered)

    if isinstance(in_data, list):
        return result
    else:
        return result[0] if C == 1 else result


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE Channel Coding - Complete Decoding Chain")
    print("="*70)
    print()

    # Example: Complete decode chain
    print("Example: Complete Decoding Chain")
    print("-" * 70)

    # Simulate received soft bits
    K = 1000
    print(f"Transport block: {K} bits")
    print()

    print("Decoding chain:")
    print("1. Rate recovery (with HARQ combining)")
    print("2. Turbo decoding (Max-Log-MAP)")
    print("3. Code block desegmentation (CRC checking)")
    print("4. CRC decoding (transport block CRC)")
    print()

    print("="*70)
    print("All decoding functions available in single module:")
    print("- lteRateRecoverTurbo")
    print("- lteTurboDecode")
    print("- lteCodeBlockDesegment")
    print("- lteCRCDecode")
    print("="*70)
