"""
LTE PUSCH Rate Matching Implementation - MATLAB-Compatible Version
Python code following MATLAB LTE Toolbox syntax and 3GPP TS 36.212 specification

COMPLETE IMPLEMENTATION:
- CRC calculation (CRC-8, CRC-16, CRC-24A, CRC-24B)
- Code block segmentation with filler bits
- Turbo encoding with QPP interleaver
- Rate matching with sub-block interleaver
- Code block concatenation

MATLAB COMPATIBILITY:
1. CRC calculation treats -1 (NULL/filler bits) as 0
2. Filler bits represented as -1 throughout (NULL marker)
3. Code block segmentation outputs -1 for filler positions
4. All functions match MATLAB LTE Toolbox behavior
"""

import numpy as np
from typing import List, Tuple, Dict

# ============================================================================
# CRC CALCULATION
# ============================================================================

class LTE_CRC:
    """
    LTE CRC Calculation
    Based on 3GPP TS 36.212 Section 5.1.1
    MATLAB-COMPATIBLE - Matches lteCRCEncode
    """

    def __init__(self):
        # CRC Generator Polynomials (MSB first, length includes x^n term)
        # gCRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
        self.gCRC24A = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]

        # gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
        self.gCRC24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        # gCRC16(D) = D^16 + D^12 + D^5 + 1
        self.gCRC16 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        # gCRC8(D) = D^8 + D^7 + D^4 + D^3 + D + 1
        self.gCRC8 = [1, 1, 0, 0, 1, 1, 0, 1, 1]

    def crc_calculate(self, input_bits, generator_poly, L):
        """
        Calculate CRC parity bits using polynomial division in GF(2)

        MATLAB-COMPATIBLE: Negative input bit values are interpreted as logical 0
        for the purposes of the CRC calculation (-1 is used to represent filler bits)
        """
        input_bits = np.array(input_bits)

        # MATLAB: "negative input bit values are interpreted as logical 0"
        # Convert -1 (NULL/filler) to 0 for CRC calculation
        input_for_crc = np.where(input_bits < 0, 0, input_bits).astype(int)

        poly = np.concatenate([input_for_crc, np.zeros(L, dtype=int)])

        for i in range(len(input_for_crc)):
            if poly[i] == 1:
                for j in range(len(generator_poly)):
                    poly[i + j] = (poly[i + j] + generator_poly[j]) % 2

        return poly[-L:].astype(int)

    def crc_attach(self, input_bits, crc_type='CRC24A', mask=0):
        """
        Attach CRC to input bit sequence

        MATLAB-COMPATIBLE: lteCRCEncode behavior
        - Supports CRC types: '8', '16', '24A', '24B'
        - Treats -1 (filler bits) as 0 for CRC calculation
        - Appends CRC after input (preserving -1 values in input)
        - Applies XOR mask MSB-first if provided
        """
        input_bits = np.array(input_bits)

        # Select CRC polynomial and length
        if crc_type == 'CRC24A' or crc_type == '24A':
            generator_poly = self.gCRC24A
            L = 24
        elif crc_type == 'CRC24B' or crc_type == '24B':
            generator_poly = self.gCRC24B
            L = 24
        elif crc_type == 'CRC16' or crc_type == '16':
            generator_poly = self.gCRC16
            L = 16
        elif crc_type == 'CRC8' or crc_type == '8':
            generator_poly = self.gCRC8
            L = 8
        else:
            raise ValueError(f"Unknown CRC type: {crc_type}")

        # Calculate CRC (handles -1 as 0 internally)
        parity_bits = self.crc_calculate(input_bits, generator_poly, L)

        # Apply mask if provided (MSB first)
        if mask != 0:
            mask_bits = np.array([int(b) for b in format(mask, f'0{L}b')], dtype=int)
            parity_bits = (parity_bits + mask_bits) % 2

        # Return input (with -1 preserved) + CRC
        return np.concatenate([input_bits, parity_bits])


# ============================================================================
# CODE BLOCK SEGMENTATION
# ============================================================================

class LTE_CodeBlockSegmentation:
    """
    LTE Code Block Segmentation
    Based on 3GPP TS 36.212 Section 5.1.2
    MATLAB-COMPATIBLE - Matches lteCodeBlockSegment
    """

    def __init__(self):
        # Maximum code block size
        self.Z = 6144

        # CRC24B Generator Polynomial (for code blocks when B > 6144)
        # gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
        self.gCRC24B = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]

        # Valid turbo interleaver block sizes from 3GPP TS 36.212 Table 5.1.3-3
        self.K_table = [
            40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
            200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336,
            344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480,
            488, 496, 504, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736,
            752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1024,
            1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504,
            1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984,
            2016, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880,
            2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776, 3840,
            3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544, 4608, 4672, 4736, 4800,
            4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504, 5568, 5632, 5696, 5760,
            5824, 5888, 5952, 6016, 6080, 6144
        ]

    def crc_calculate(self, input_bits, generator_poly, L):
        """
        Calculate CRC parity bits using polynomial division in GF(2)
        MATLAB-COMPATIBLE: Negative input bit values (-1) are interpreted as logical 0
        """
        input_bits = np.array(input_bits)

        # Convert -1 (NULL/filler) to 0 for CRC calculation
        input_for_crc = np.where(input_bits < 0, 0, input_bits).astype(int)

        # Polynomial division in GF(2)
        poly = np.concatenate([input_for_crc, np.zeros(L, dtype=int)])

        for i in range(len(input_for_crc)):
            if poly[i] == 1:
                for j in range(len(generator_poly)):
                    poly[i + j] = (poly[i + j] + generator_poly[j]) % 2

        return poly[-L:].astype(int)

    def segment(self, input_bits):
        """
        Code block segmentation following 3GPP TS 36.212 Section 5.1.2
        MATLAB lteCodeBlockSegment behavior
        """
        input_bits = np.array(input_bits, dtype=int)
        B = len(input_bits)

        # Determine if segmentation is needed
        if B <= self.Z:
            L = 0  # No CRC24B appended
            C = 1
            B_prime = B
        else:
            L = 24  # CRC24B will be appended to each block
            C = int(np.ceil(B / (self.Z - L)))
            B_prime = B + C * L

        # Find K+ and K- from table
        if C == 1:
            K_plus = min([k for k in self.K_table if k >= B], default=6144)
            K_minus = 0
            C_plus = 1
            C_minus = 0
        else:
            K_plus = min([k for k in self.K_table if C * k >= B_prime], default=6144)

            K_minus_candidates = [k for k in self.K_table if k < K_plus]
            if K_minus_candidates:
                K_minus = max(K_minus_candidates)
            else:
                K_minus = 0

            if K_minus > 0:
                delta_K = K_plus - K_minus
                C_minus = int(np.floor((C * K_plus - B_prime) / delta_K))
                C_plus = C - C_minus
            else:
                C_plus = C
                C_minus = 0

            if C_minus == 0:
                K_minus = 0

        # Calculate filler bits
        if C == 1:
            F = K_plus - B
        else:
            F = C_plus * K_plus + C_minus * K_minus - B_prime

        # Generate code blocks
        code_blocks = []
        bit_index = 0

        for r in range(C):
            if r < C_minus:
                K_r = K_minus
            else:
                K_r = K_plus

            code_block = np.zeros(K_r, dtype=int)

            # Filler bits only in first block
            if r == 0:
                filler_count = F
                data_start = F
                # MATLAB: "The <NULL> filler bits (represented by -1 at the output)"
                code_block[:F] = -1
            else:
                filler_count = 0
                data_start = 0

            data_length = K_r - L if L > 0 else K_r

            # Fill data bits
            for k in range(data_start, data_length):
                if bit_index < B:
                    code_block[k] = input_bits[bit_index]
                    bit_index += 1
                else:
                    code_block[k] = 0

            # Attach CRC24B if needed (only when B > 6144)
            if L >= 1:
                if r == 0 and F > 0:
                    data_for_crc = code_block[F:data_length]
                else:
                    data_for_crc = code_block[:data_length]

                # CRC calculation treats -1 as 0
                crc_bits = self.crc_calculate(data_for_crc, self.gCRC24B, L)
                code_block[data_length:] = crc_bits

            code_blocks.append(code_block)

        segmentation_info = {
            'B': B,
            'C': C,
            'L': L,
            'B_prime': B_prime,
            'K_plus': K_plus,
            'K_minus': K_minus,
            'C_plus': C_plus,
            'C_minus': C_minus,
            'F': F,
            'code_block_sizes': [len(cb) for cb in code_blocks]
        }

        return code_blocks, segmentation_info


# ============================================================================
# TURBO ENCODER
# ============================================================================

class LTE_TurboEncoder:
    """LTE Turbo Encoder with QPP Interleaver - MATLAB Compatible"""

    def __init__(self):
        self.interleaver_params = {
            40: (3, 10), 48: (7, 12), 56: (19, 42), 64: (7, 16), 72: (7, 18), 80: (11, 20),
            88: (5, 22), 96: (11, 24), 104: (7, 26), 112: (41, 84), 120: (103, 90), 128: (15, 32),
            136: (9, 34), 144: (17, 108), 152: (9, 38), 160: (21, 120), 168: (101, 84), 176: (21, 44),
            184: (57, 46), 192: (23, 48), 200: (13, 50), 208: (27, 52), 216: (11, 36), 224: (27, 56),
            232: (85, 58), 240: (29, 60), 248: (33, 62), 256: (15, 32), 264: (17, 198), 272: (33, 68),
            280: (103, 210), 288: (19, 36), 296: (19, 74), 304: (37, 76), 312: (19, 78), 320: (21, 120),
            328: (21, 82), 336: (115, 84), 344: (193, 86), 352: (21, 44), 360: (133, 90), 368: (81, 46),
            376: (45, 94), 384: (23, 48), 392: (243, 98), 400: (151, 40), 408: (155, 102), 416: (25, 52),
            424: (51, 106), 432: (47, 72), 440: (91, 110), 448: (29, 168), 456: (29, 114), 464: (247, 58),
            472: (29, 118), 480: (89, 180), 488: (91, 122), 496: (157, 62), 504: (55, 84), 512: (31, 64),
            528: (17, 66), 544: (35, 68), 560: (227, 420), 576: (65, 96), 592: (19, 74), 608: (37, 76),
            624: (41, 234), 640: (39, 80), 656: (185, 82), 672: (43, 252), 688: (21, 86), 704: (155, 44),
            720: (79, 120), 736: (139, 92), 752: (23, 94), 768: (217, 48), 784: (25, 98), 800: (17, 80),
            816: (127, 102), 832: (25, 52), 848: (239, 106), 864: (17, 48), 880: (137, 110), 896: (215, 112),
            912: (29, 114), 928: (15, 58), 944: (147, 118), 960: (29, 60), 976: (59, 122), 992: (65, 124),
            1008: (55, 84), 1024: (31, 64), 1056: (17, 66), 1088: (171, 204), 1120: (67, 140), 1152: (35, 72),
            1184: (19, 74), 1216: (39, 76), 1248: (19, 78), 1280: (199, 240), 1312: (21, 82), 1344: (211, 252),
            1376: (21, 86), 1408: (43, 88), 1440: (149, 60), 1472: (45, 92), 1504: (49, 846), 1536: (71, 48),
            1568: (13, 28), 1600: (17, 80), 1632: (25, 102), 1664: (183, 104), 1696: (55, 954), 1728: (127, 96),
            1760: (27, 110), 1792: (29, 112), 1824: (29, 114), 1856: (57, 116), 1888: (45, 354), 1920: (31, 120),
            1952: (59, 610), 1984: (185, 124), 2016: (113, 420), 2048: (31, 64), 2112: (17, 66), 2176: (171, 136),
            2240: (209, 420), 2304: (253, 216), 2368: (367, 444), 2432: (265, 456), 2496: (181, 468), 2560: (39, 80),
            2624: (27, 164), 2688: (127, 504), 2752: (143, 172), 2816: (43, 88), 2880: (29, 300), 2944: (45, 92),
            3008: (157, 188), 3072: (47, 96), 3136: (13, 28), 3200: (111, 240), 3264: (443, 204), 3328: (51, 104),
            3392: (51, 212), 3456: (451, 192), 3520: (257, 220), 3584: (57, 336), 3648: (313, 228), 3712: (271, 232),
            3776: (179, 236), 3840: (331, 120), 3904: (363, 244), 3968: (375, 248), 4032: (127, 168), 4096: (31, 64),
            4160: (33, 130), 4224: (43, 264), 4288: (33, 134), 4352: (477, 408), 4416: (35, 138), 4480: (233, 280),
            4544: (357, 142), 4608: (337, 480), 4672: (37, 146), 4736: (71, 444), 4800: (71, 120), 4864: (37, 152),
            4928: (39, 462), 4992: (127, 234), 5056: (39, 158), 5120: (39, 80), 5184: (31, 96), 5248: (113, 902),
            5312: (41, 166), 5376: (251, 336), 5440: (43, 170), 5504: (21, 86), 5568: (43, 174), 5632: (45, 176),
            5696: (45, 178), 5760: (161, 120), 5824: (89, 182), 5888: (323, 184), 5952: (47, 186), 6016: (23, 94),
            6080: (47, 190), 6144: (263, 480)
        }

    def qpp_interleaver(self, c: List[int], K: int) -> List[int]:
        """
        QPP interleaver - MATLAB compatible
        Interleaves the input sequence (may contain -1 for filler bits)
        """
        if K not in self.interleaver_params:
            raise ValueError(f"Interleaver params for K={K} not found")
        f1, f2 = self.interleaver_params[K]
        i = np.arange(K, dtype=np.int64)
        p = (f1 * i + f2 * (i * i)) % K

        # Use int16 to handle -1 (NULL) values
        c_arr = np.array(c, dtype=np.int16)
        return c_arr[p].tolist()

    def rsc_encode(self, u: List[int], filler_count: int = 0) -> Tuple[List[int], List[int], int]:
        """
        RSC encoder with filler bit handling - MATLAB compatible

        The encoder is initialized with all zeros (state = 0).
        For filler positions (k=0,...,filler_count-1):
            - Input: Treat -1 as 0 for encoding (state calculation)
            - Output: Set to NULL (-1) after encoding
        """
        sys_out, par_out = [], []
        state = 0  # Encoder initialized with all zeros

        for idx, bit in enumerate(u):
            # For encoding: treat -1 as 0, otherwise use the bit value
            encoding_bit = 0 if bit == -1 else int(bit)

            # RSC encoder logic
            s1 = state & 1
            s2 = (state >> 1) & 1
            s3 = (state >> 2) & 1
            r = encoding_bit ^ s2 ^ s3
            p = r ^ s1 ^ s3

            # For filler positions (first F bits): output NULL
            if idx < filler_count:
                sys_out.append(-1)  # x_k = NULL
                par_out.append(-1)  # z_k = NULL
            else:
                sys_out.append(encoding_bit)
                par_out.append(int(p))

            # State always updates (using encoding_bit = 0 for fillers)
            state = (s2 << 2) | (s1 << 1) | (r & 1)

        return sys_out, par_out, state

    def generate_tail(self, state: int) -> Tuple[List[int], List[int]]:
        """Generate tail bits"""
        tail_u, tail_p = [], []
        cur = state
        for _ in range(3):
            s1 = cur & 1
            s2 = (cur >> 1) & 1
            s3 = (cur >> 2) & 1
            u = s2 ^ s3
            r = u ^ s2 ^ s3
            p = r ^ s1 ^ s3
            tail_u.append(int(u))
            tail_p.append(int(p))
            cur = (s2 << 2) | (s1 << 1) | (r & 1)
        return tail_u, tail_p

    def turbo_encode(self, message: List[int], F: int = 0) -> Tuple[List[int], List[int], List[int]]:
        """
        Main turbo encoder with filler bit handling (3GPP TS 36.212)
        MATLAB-COMPATIBLE

        The turbo encoder is initialized with all zeros (state = 0).

        For filler bits (k=0,...,F-1):
            - Input: c_k = -1 (NULL marker)
            - Processing: Encoder treats -1 as 0 for state calculation
            - Output: x_k = <NULL> (-1), z^(1)_k = <NULL> (-1), z^(2)_k = <NULL> (-1)
            - State: Updates normally (as if input was 0)

        For data bits (k=F,...,K-1):
            - Normal turbo encoding
        """
        K = len(message)

        # First RSC encoder (initialized with state=0)
        # Processes -1 as 0, outputs -1 for first F positions
        sys1, par1, state1 = self.rsc_encode(message, filler_count=F)
        tail_u1, tail_p1 = self.generate_tail(state1)

        # Interleave the input (interleaves -1 values normally)
        c_prime = self.qpp_interleaver(message, K)

        # Second RSC encoder
        # Process c_prime and manually set outputs to -1 where input is -1
        sys2_temp, par2_temp, state2 = self.rsc_encode(c_prime, filler_count=0)

        # Set outputs to -1 where c_prime had -1
        par2 = []
        for idx in range(K):
            if c_prime[idx] == -1:
                par2.append(-1)  # z^(2)_k = NULL for filler positions
            else:
                par2.append(par2_temp[idx])

        tail_u2, tail_p2 = self.generate_tail(state2)

        # Build transmission sequence in interleaved format
        tx_seq = []
        for k in range(K):
            tx_seq.extend([sys1[k], par1[k], par2[k]])

        # Append tails (never NULL)
        for t in range(3):
            tx_seq.append(tail_u1[t])
            tx_seq.append(tail_p1[t])
        for t in range(3):
            tx_seq.append(tail_u2[t])
            tx_seq.append(tail_p2[t])

        # Split into d0, d1, d2 streams
        d0 = tx_seq[0::3]
        d1 = tx_seq[1::3]
        d2 = tx_seq[2::3]

        return d0, d1, d2


# ============================================================================
# RATE MATCHING WITH SUB-BLOCK INTERLEAVER
# ============================================================================

class LTE_RateMatching:
    """
    LTE Rate Matching for Turbo Coded Data
    Based on 3GPP TS 36.212 Section 5.1.4.1
    MATLAB-COMPATIBLE
    """

    def __init__(self):
        # Sub-block interleaver parameters
        self.C_subblock = 32
        # Column permutation pattern from Table 5.1.4-1
        self.P = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                          1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])

    def sub_block_interleaver(self, d: np.ndarray, stream_idx: int) -> Tuple[np.ndarray, int]:
        """
        Sub-block interleaver (3GPP TS 36.212 Section 5.1.4.1.1)
        """
        D = len(d)

        # Step (1): Number of columns is fixed at 32
        C_subblock = self.C_subblock

        # Step (2): Determine number of rows
        R_subblock = int(np.ceil(D / C_subblock))

        # Calculate total matrix size
        K_PI = R_subblock * C_subblock

        # Step (3): Pad dummy bits if needed
        N_D = K_PI - D

        # Create padded sequence y
        if N_D > 0:
            y = np.concatenate([np.full(N_D, -1, dtype=int), d])
        else:
            y = d.copy()

        # Write y into (R_subblock × C_subblock) matrix ROW BY ROW
        matrix = y.reshape(R_subblock, C_subblock)

        # For d^(0) and d^(1): Standard inter-column permutation
        if stream_idx in [0, 1]:
            # Step (4): Perform inter-column permutation
            matrix_permuted = matrix[:, self.P]

            # Step (5): Read column by column (column-major order)
            v = matrix_permuted.flatten('F')  # Fortran order

        # For d^(2): Special permutation formula
        else:
            # Apply special permutation formula
            v = np.zeros(K_PI, dtype=int)

            for k in range(K_PI):
                col_index = k // R_subblock
                row_index = k % R_subblock
                pi_k = (self.P[col_index] + C_subblock * row_index + 1) % K_PI
                v[k] = y[pi_k]

        return v, K_PI

    def create_circular_buffer(self, d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create circular buffer from three interleaved streams
        3GPP TS 36.212 Section 5.1.4.1.2

        Interlacing pattern:
        w_k = v_k^(0)           for k = 0,..., K_Π - 1
        w_{K_Π + 2k} = v_k^(1)  for k = 0,..., K_Π - 1
        w_{K_Π + 2k+1} = v_k^(2) for k = 0,..., K_Π - 1
        """
        # Apply sub-block interleaving to each stream
        v0, K_pi = self.sub_block_interleaver(d0, stream_idx=0)
        v1, _ = self.sub_block_interleaver(d1, stream_idx=1)
        v2, _ = self.sub_block_interleaver(d2, stream_idx=2)

        # Create circular buffer with interlacing pattern
        K_w = 3 * K_pi
        w = np.full(K_w, -1, dtype=int)

        # w_k = v_k^(0) for k = 0,..., K_Π - 1
        for k in range(K_pi):
            w[k] = v0[k]

        # w_{K_Π + 2k} = v_k^(1) for k = 0,..., K_Π - 1
        for k in range(K_pi):
            w[K_pi + 2*k] = v1[k]

        # w_{K_Π + 2k+1} = v_k^(2) for k = 0,..., K_Π - 1
        for k in range(K_pi):
            w[K_pi + 2*k + 1] = v2[k]

        return w, K_w

    def bit_selection_and_pruning(self, w: np.ndarray, K_w: int, E: int, rv: int, R_subblock: int) -> np.ndarray:
        """
        Bit selection and pruning from circular buffer
        3GPP TS 36.212 Section 5.1.4.1.2

        MATLAB-COMPATIBLE: Skips NULL bits (-1) during selection
        """
        # For UL-SCH: N_cb = K_w (no soft buffer limitation)
        N_cb = K_w

        # Calculate starting position based on redundancy version
        # k_0 = R_subblock × (2 × ⌈N_cb/(8×R_subblock)⌉ × rv + 2)
        k_0 = int(R_subblock * (2 * np.ceil(N_cb / (8 * R_subblock)) * rv + 2))

        # Bit selection (circular buffer readout, skip NULL bits)
        out = []
        j = 0
        k = 0

        while k < E:
            # Wrap around circular buffer
            w_idx = (k_0 + j) % N_cb
            bit = w[w_idx]

            # Skip NULL bits (represented as -1)
            if bit != -1:
                out.append(int(bit))
                k += 1

            j += 1

        return np.array(out, dtype=int)

    def rate_match(self, turbo_output: np.ndarray, E: int, rv: int = 0) -> np.ndarray:
        """
        Complete rate matching process - MATLAB compatible

        Handles -1 (NULL) for filler bit positions
        """
        # Verify input length is multiple of 3
        total_len = len(turbo_output)
        if total_len % 3 != 0:
            raise ValueError(f"Input length ({total_len}) must be a multiple of 3")

        # Calculate D: length of each stream (d0, d1, d2)
        D = total_len // 3

        # Extract the three streams
        d0 = turbo_output[0::3]
        d1 = turbo_output[1::3]
        d2 = turbo_output[2::3]

        # Verify extraction
        if len(d0) != D or len(d1) != D or len(d2) != D:
            raise ValueError(f"Stream extraction failed: d0={len(d0)}, d1={len(d1)}, d2={len(d2)}, expected D={D}")

        # Calculate R_subblock for bit selection
        R_subblock = int(np.ceil(D / self.C_subblock))

        # Create circular buffer (handles NULL values)
        w, K_w = self.create_circular_buffer(d0, d1, d2)

        # Bit selection and pruning (skips NULL values)
        rate_matched = self.bit_selection_and_pruning(w, K_w, E, rv, R_subblock)

        return rate_matched


# ============================================================================
# CODE BLOCK CONCATENATION
# ============================================================================

def code_block_concatenation(rate_matched_blocks: List[np.ndarray]) -> np.ndarray:
    """
    Code Block Concatenation (3GPP TS 36.212 Section 5.1.5)
    """
    f = []
    for r in range(len(rate_matched_blocks)):
        for j in range(len(rate_matched_blocks[r])):
            f.append(rate_matched_blocks[r][j])
    return np.array(f, dtype=int)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def binary_string_to_array(binary_string: str) -> np.ndarray:
    """Convert binary string to array"""
    return np.array([int(bit) for bit in binary_string], dtype=int)


def array_to_binary_string(bit_array) -> str:
    """Convert array to binary string (showing -1 as 'N')"""
    return ''.join(['N' if int(b) == -1 else str(int(b)) for b in bit_array])


def lteTurboEncode(blk):
    """
    MATLAB lteTurboEncode equivalent - Turbo encoding

    Syntax:
        out = lteTurboEncode(in)

    Parameters:
        blk: Input data vector or cell array (Python list) of vectors
             Only legal turbo interleaver block sizes are supported (40-6144)
             Filler bits supported through negative input values (-1)

    Returns:
        out: Turbo encoded bits as int8 vector or cell array of int8 vectors
             Output format: [S P1 P2] where:
               S = systematic bits (K+4)
               P1 = encoder 1 parity bits (K+4)
               P2 = encoder 2 parity bits (K+4)
             Total output length: 3*(K+4) bits

    MATLAB Documentation:
        "The encoder is a Parallel Concatenated Convolutional Code (PCCC) with
        two 8-state constituent encoders and a contention free interleaver. The
        coding rate of turbo encoder is 1/3 and the 3 encoded parity streams
        are concatenated block-wise to form the encoded output i.e. [S P1 P2]
        where S is the systematic bits, P1 is the encoder 1 bits and P2 is the
        encoder 2 bits. To support the correct processing of filler bits,
        negative input bit values are specially processed. They are treated as
        logical 0 at the input to both encoders but their negative values are
        passed directly through to the associated output positions in
        sub-blocks S and P1."

    Examples:
        >>> # Single vector input
        >>> out1 = lteTurboEncode(np.ones(40, dtype=int))
        >>> out1.shape
        (132,)
        >>> out1.dtype
        dtype('int8')

        >>> # Cell array input
        >>> out2 = lteTurboEncode([np.ones(40, dtype=int), np.ones(6144, dtype=int)])
        >>> len(out2)
        2
        >>> [len(cb) for cb in out2]
        [132, 18444]

        >>> # With filler bits (-1)
        >>> input_with_filler = np.array([-1, -1, 1, 0, 1, 1, 0, 1] + [1]*32, dtype=int)
        >>> out3 = lteTurboEncode(input_with_filler)
        >>> out3[0]  # Filler bit passed through in S
        -1
    """
    encoder = LTE_TurboEncoder()

    # Check if input is cell array (Python list)
    if isinstance(blk, list):
        # Process each vector in the cell array
        result = []
        for vec in blk:
            vec_array = np.array(vec, dtype=int)
            K = len(vec_array)

            # Count filler bits (negative values)
            F = np.sum(vec_array < 0)

            # Encode
            d0, d1, d2 = encoder.turbo_encode(vec_array.tolist(), F=F)

            # Concatenate [S P1 P2] and convert to int8
            output = np.concatenate([d0, d1, d2]).astype(np.int8)
            result.append(output)

        return result

    else:
        # Single vector input
        blk_array = np.array(blk, dtype=int)
        K = len(blk_array)

        # Count filler bits (negative values)
        F = np.sum(blk_array < 0)

        # Encode
        d0, d1, d2 = encoder.turbo_encode(blk_array.tolist(), F=F)

        # Concatenate [S P1 P2] and convert to int8
        output = np.concatenate([d0, d1, d2]).astype(np.int8)

        return output


def calculate_rate_matching_E(num_blocks: int, input_bits: int, N_PRB: int, Q_m: int,
                               n_layers: int = 1, N_symb_PUSCH: int = 12) -> List[int]:
    """
    Calculate rate matching output size E for each code block
    3GPP TS 36.212 Section 5.1.4.1.2
    """
    N_sc_RB = 12

    # Total number of bits available for transmission
    G = N_PRB * N_sc_RB * Q_m * n_layers * N_symb_PUSCH

    # G' = G / (N_L · Q_m)
    G_prime = G / (n_layers * Q_m)

    # γ = G' mod C
    gamma = int(G_prime) % num_blocks

    # Calculate E for each code block
    E_vector = []
    for r in range(num_blocks):
        if r <= (num_blocks - gamma - 1):
            E = int(n_layers * Q_m * np.floor(G_prime / num_blocks))
        else:
            E = int(n_layers * Q_m * np.ceil(G_prime / num_blocks))
        E_vector.append(E)

    return E_vector
