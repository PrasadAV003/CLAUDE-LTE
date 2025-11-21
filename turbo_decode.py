"""
LTE Turbo Decoding - MATLAB-Compatible Implementation
Python equivalent of MATLAB lteTurboDecode

MATLAB Compatibility:
- Parallel Concatenated Convolutional Code (PCCC) decoder
- Max-Log-MAP algorithm for constituent RSC decoders
- Input format: [S P1 P2] block-wise concatenation
- Configurable iteration cycles (default: 5, range: 1-30)
- Supports single vector or cell array input
- Returns int8 decoded bits

Based on 3GPP TS 36.212 Section 5.1.3.2
Implementation uses Max-Log-MAP SISO algorithm with Numba optimization
"""

import numpy as np
from numba import jit
from typing import Union, List, Tuple


# ============================================================================
# TRELLIS STRUCTURE FOR LTE RSC ENCODER
# ============================================================================

# Generator polynomials for LTE turbo code RSC encoder
# g0 = 0o13 = 1011 binary (feedback, taps at positions 0,1,3)
# g1 = 0o15 = 1101 binary (feedforward, taps at positions 0,2,3)
#
# Trellis structure for 8-state (constraint length 4) RSC encoder
# State encoding: [s2, s1, s0] where s0 is oldest
#
# For each state and input bit (0 or 1):
#   - Compute feedback: input ⊕ s0 ⊕ s1
#   - Next state: [feedback, s2, s1]
#   - Systematic output: feedback (which equals input ⊕ s0 ⊕ s1)
#   - Parity output: feedback ⊕ s0 ⊕ s2

# Precomputed next state table: nextStates[state][input]
# Verified to match turbo_encode.py RSC encoder exactly
NEXT_STATES = np.array([
    [0, 4],  # State 0: input 0 → state 0, input 1 → state 4
    [0, 4],  # State 1: input 0 → state 0, input 1 → state 4
    [5, 1],  # State 2: input 0 → state 5, input 1 → state 1
    [5, 1],  # State 3: input 0 → state 5, input 1 → state 1
    [6, 2],  # State 4: input 0 → state 6, input 1 → state 2
    [6, 2],  # State 5: input 0 → state 6, input 1 → state 2
    [3, 7],  # State 6: input 0 → state 3, input 1 → state 7
    [3, 7],  # State 7: input 0 → state 3, input 1 → state 7
], dtype=np.int32)

# Precomputed output table: outputs[state][input] = (systematic_bit << 1) | parity_bit
# Key: Systematic output = input bit (not feedback!)
# Verified to match turbo_encode.py RSC encoder exactly
OUTPUTS = np.array([
    [0, 3],  # State 0: input 0 → sys=0,par=0; input 1 → sys=1,par=1
    [1, 2],  # State 1: input 0 → sys=0,par=1; input 1 → sys=1,par=0
    [1, 2],  # State 2: input 0 → sys=0,par=1; input 1 → sys=1,par=0
    [0, 3],  # State 3: input 0 → sys=0,par=0; input 1 → sys=1,par=1
    [0, 3],  # State 4: input 0 → sys=0,par=0; input 1 → sys=1,par=1
    [1, 2],  # State 5: input 0 → sys=0,par=1; input 1 → sys=1,par=0
    [1, 2],  # State 6: input 0 → sys=0,par=1; input 1 → sys=1,par=0
    [0, 3],  # State 7: input 0 → sys=0,par=0; input 1 → sys=1,par=1
], dtype=np.int32)


# ============================================================================
# MAX-LOG-MAP SISO DECODER (NUMBA OPTIMIZED)
# ============================================================================

@jit(nopython=True)
def _siso_decode_maxlog_numba(sys_full: np.ndarray, par_full: np.ndarray,
                               apr_full: np.ndarray, K: int) -> np.ndarray:
    """
    Max-Log-MAP SISO decoder for RSC constituent code

    Implements the BCJR algorithm using max approximation (Max-Log-MAP).
    Computes soft output LLRs using forward (alpha) and backward (beta) metrics.

    Parameters:
        sys_full: Systematic bit LLRs (length K+4, includes tail)
        par_full: Parity bit LLRs (length K+4, includes tail)
        apr_full: A priori LLRs (length K+4)
        K: Information block size (without tail bits)

    Returns:
        Extrinsic information LLRs (length K, no tail)
    """
    N = K + 4  # Total length with tail bits

    # Initialize forward metrics (alpha)
    alpha = np.full((N + 1, 8), -np.inf, dtype=np.float64)
    alpha[0, 0] = 0.0  # Start at state 0

    # Initialize backward metrics (beta)
    beta = np.full((N + 1, 8), -np.inf, dtype=np.float64)
    beta[N, 0] = 0.0  # End at state 0 (trellis termination)

    # Precompute branch metrics (gamma)
    # gamma[t, s, input] = branch metric for transition at time t from state s with input bit
    gamma = np.zeros((N, 8, 2), dtype=np.float64)

    for t in range(N):
        sys_t = sys_full[t]
        par_t = par_full[t]
        apr_t = apr_full[t]

        for s in range(8):
            for input_bit in range(2):
                # Get next state and output
                next_s = NEXT_STATES[s, input_bit]
                output_bits = OUTPUTS[s, input_bit]

                # Extract systematic and parity output bits
                sys_out = (output_bits >> 1) & 1  # Systematic output
                par_out = output_bits & 1          # Parity output

                # Branch metric: correlation with received soft bits
                # LLR convention: positive = bit 0, negative = bit 1
                sys_contrib = sys_t * (1.0 - 2.0 * sys_out)
                par_contrib = par_t * (1.0 - 2.0 * par_out)
                apr_contrib = apr_t * (1.0 - 2.0 * input_bit)

                gamma[t, s, input_bit] = sys_contrib + par_contrib + apr_contrib

    # Forward pass (compute alpha)
    for t in range(N):
        for s in range(8):
            for input_bit in range(2):
                next_s = NEXT_STATES[s, input_bit]
                metric = alpha[t, s] + gamma[t, s, input_bit]
                alpha[t + 1, next_s] = max(alpha[t + 1, next_s], metric)

    # Backward pass (compute beta)
    for t in range(N - 1, -1, -1):
        for s in range(8):
            for input_bit in range(2):
                next_s = NEXT_STATES[s, input_bit]
                metric = beta[t + 1, next_s] + gamma[t, s, input_bit]
                beta[t, s] = max(beta[t, s], metric)

    # Compute LLRs (extrinsic information only, exclude a priori)
    llr_out = np.zeros(K, dtype=np.float64)

    for t in range(K):  # Only for information bits (not tail)
        max_0 = -np.inf
        max_1 = -np.inf

        for s in range(8):
            # Input bit = 0
            next_s_0 = NEXT_STATES[s, 0]
            metric_0 = alpha[t, s] + gamma[t, s, 0] + beta[t + 1, next_s_0]
            max_0 = max(max_0, metric_0)

            # Input bit = 1
            next_s_1 = NEXT_STATES[s, 1]
            metric_1 = alpha[t, s] + gamma[t, s, 1] + beta[t + 1, next_s_1]
            max_1 = max(max_1, metric_1)

        # LLR = log(P(bit=0)/P(bit=1))
        # Extrinsic = total LLR - a priori LLR
        total_llr = max_0 - max_1
        llr_out[t] = total_llr - apr_full[t]

    return llr_out


# ============================================================================
# QPP INTERLEAVER
# ============================================================================

class QPPInterleaver:
    """Quadratic Permutation Polynomial Interleaver for LTE Turbo Codes"""

    def __init__(self):
        # QPP Interleaver parameters from TS 36.212 Table 5.1.3-3
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
        """Interleave data using QPP: Π(i) = (f1*i + f2*i²) mod K"""
        if K not in self.params:
            raise ValueError(f"Unsupported interleaver size K={K}")

        f1, f2 = self.params[K]
        output = np.zeros_like(data)

        for i in range(K):
            pi_i = (f1 * i + f2 * i * i) % K
            output[pi_i] = data[i]

        return output

    def deinterleave(self, data: np.ndarray, K: int) -> np.ndarray:
        """Deinterleave data using inverse QPP"""
        if K not in self.params:
            raise ValueError(f"Unsupported interleaver size K={K}")

        f1, f2 = self.params[K]
        output = np.zeros_like(data)

        for i in range(K):
            pi_i = (f1 * i + f2 * i * i) % K
            output[i] = data[pi_i]

        return output


# ============================================================================
# TURBO DECODER
# ============================================================================

class LTE_TurboDecoder:
    """
    LTE Turbo Decoder - Max-Log-MAP Implementation

    Implements iterative turbo decoding for PCCC using Max-Log-MAP SISO
    algorithm for constituent RSC decoders.

    Based on 3GPP TS 36.212 Section 5.1.3.2
    """

    def __init__(self):
        self.interleaver = QPPInterleaver()

    def decode(self, soft_input: np.ndarray, K: int, num_iterations: int = 5) -> np.ndarray:
        """
        Turbo decode soft input data

        Parameters:
            soft_input: Soft input LLRs in [S P1 P2] format (length 3*(K+4))
            K: Information block size (before tail bits)
            num_iterations: Number of decoding iterations (1-30)

        Returns:
            Decoded hard bits (int8, length K)
        """
        N = K + 4  # Total length with tail bits

        # Split input into three streams: [S | P1 | P2]
        # Format: Block-wise concatenation, NOT interleaved
        # S = soft_input[0:N]
        # P1 = soft_input[N:2*N]
        # P2 = soft_input[2*N:3*N]
        sys_llr = soft_input[0:N].copy()
        par1_llr = soft_input[N:2*N].copy()
        par2_llr = soft_input[2*N:3*N].copy()

        # Initialize a priori information
        apr1 = np.zeros(N, dtype=np.float64)
        apr2 = np.zeros(N, dtype=np.float64)

        # Iterative decoding
        for iteration in range(num_iterations):
            # Decoder 1: Process systematic + parity1
            ext1 = _siso_decode_maxlog_numba(sys_llr, par1_llr, apr1, K)

            # Extend to full length for interleaving (pad tail bits with zeros)
            ext1_full = np.zeros(N, dtype=np.float64)
            ext1_full[:K] = ext1

            # Interleave extrinsic information for decoder 2
            apr2_interleaved = self.interleaver.interleave(ext1_full, K)

            # Interleave systematic information
            sys_llr_interleaved = self.interleaver.interleave(sys_llr[:K], K)
            sys_llr_int_full = np.zeros(N, dtype=np.float64)
            sys_llr_int_full[:K] = sys_llr_interleaved
            # Add tail bits (last 4 systematic bits, not interleaved)
            sys_llr_int_full[K:] = sys_llr[K:]

            # Decoder 2: Process interleaved systematic + parity2
            ext2 = _siso_decode_maxlog_numba(sys_llr_int_full, par2_llr, apr2_interleaved, K)

            # Extend to full length
            ext2_full = np.zeros(N, dtype=np.float64)
            ext2_full[:K] = ext2

            # Deinterleave extrinsic information for decoder 1
            apr1_deinterleaved = self.interleaver.deinterleave(ext2_full, K)
            apr1 = apr1_deinterleaved
            apr2 = apr2_interleaved

        # Final decision: run one more SISO decode to get final LLRs
        ext1_final = _siso_decode_maxlog_numba(sys_llr, par1_llr, apr1, K)

        # Make hard decisions
        # Total LLR = systematic + extrinsic from decoder 1
        # (a priori is already included in ext1_final computation, so don't add again)
        decoded = np.zeros(K, dtype=np.int8)
        for i in range(K):
            # Combine systematic LLR + extrinsic LLR (which includes effect of both decoders)
            total_llr = sys_llr[i] + ext1_final[i]
            decoded[i] = 0 if total_llr >= 0 else 1

        return decoded


# ============================================================================
# MATLAB-COMPATIBLE WRAPPER FUNCTION
# ============================================================================

def lteTurboDecode(in_data: Union[np.ndarray, List[np.ndarray]],
                   nturbodecits: int = 5) -> Union[np.ndarray, List[np.ndarray]]:
    """
    MATLAB lteTurboDecode equivalent - Turbo decoding

    Implements Max-Log-MAP algorithm for PCCC turbo decoding.

    Syntax:
        out = lteTurboDecode(in)
        out = lteTurboDecode(in, nturbodecits)

    Parameters:
        in_data: Soft bit input data - vector or cell array (list) of vectors
                 Expected to be PCCC encoded in [S P1 P2] format
                 Soft values (LLRs - Log-Likelihood Ratios)
                 Positive LLR = bit 0, Negative LLR = bit 1
        nturbodecits: Number of turbo decoding iteration cycles (1-30)
                     Optional, default: 5

    Returns:
        Decoded bits as int8 column vector or cell array of int8 vectors

    Example:
        txBits = randi([0 1], 6144, 1);
        codedData = lteTurboEncode(txBits);
        txSymbols = lteSymbolModulate(codedData, 'QPSK');
        noise = 0.5*complex(randn(size(txSymbols)), randn(size(txSymbols)));
        rxSymbols = txSymbols + noise;
        softBits = lteSymbolDemodulate(rxSymbols, 'QPSK', 'Soft');
        rxBits = lteTurboDecode(softBits);
        numberErrors = sum(rxBits ~= int8(txBits))

    Note:
        This implementation uses Max-Log-MAP algorithm (sub-log-MAP) which
        provides near-optimal performance with reduced complexity.
    """
    # Validate iterations
    if nturbodecits < 1 or nturbodecits > 30:
        raise ValueError(f"Number of iterations must be between 1 and 30, got {nturbodecits}")

    # Create decoder instance
    decoder = LTE_TurboDecoder()

    # Check if input is cell array (list)
    if isinstance(in_data, list):
        # Process each code block separately
        result = []

        for code_block in in_data:
            if len(code_block) == 0:
                result.append(np.array([], dtype=np.int8))
                continue

            # Convert to float for soft values
            soft_block = np.array(code_block, dtype=np.float64)

            # Determine K from length
            total_len = len(soft_block)
            if total_len % 3 != 0:
                raise ValueError(f"Input length ({total_len}) must be multiple of 3")

            D = total_len // 3
            K = D - 4  # Remove tail bits

            if K < 40 or K > 6144:
                raise ValueError(f"Invalid block size K={K}, must be in range [40, 6144]")

            # Decode
            decoded = decoder.decode(soft_block, K, nturbodecits)
            result.append(decoded)

        return result

    else:
        # Single vector input
        soft_data = np.array(in_data, dtype=np.float64)

        if len(soft_data) == 0:
            return np.array([], dtype=np.int8)

        # Determine K from length
        total_len = len(soft_data)
        if total_len % 3 != 0:
            raise ValueError(f"Input length ({total_len}) must be multiple of 3")

        D = total_len // 3
        K = D - 4  # Remove tail bits

        if K < 40 or K > 6144:
            raise ValueError(f"Invalid block size K={K}, must be in range [40, 6144]")

        # Decode
        decoded = decoder.decode(soft_data, K, nturbodecits)

        return decoded


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LTE Turbo Decode - Max-Log-MAP Implementation")
    print("="*70)
    print()

    # Example: Basic decoding
    print("Example: Turbo Decoding with Encoding")
    print("-" * 70)

    try:
        from turbo_encode import lteTurboEncode

        # Test data
        K = 40
        txBits = np.random.randint(0, 2, K, dtype=int)
        print(f"Original bits: {K} bits")

        # Encode
        encoded = lteTurboEncode(txBits)
        print(f"Encoded: {len(encoded)} bits")

        # Simulate noisy channel (AWGN)
        snr_db = 2.0
        # BPSK modulation: 0→+1, 1→-1
        modulated = np.where(encoded == 0, 1.0, -1.0)
        # Add noise
        noise_var = 10 ** (-snr_db / 10)
        noise = np.random.randn(len(modulated)) * np.sqrt(noise_var)
        received = modulated + noise
        # Compute LLRs
        softBits = received * (2.0 / noise_var)

        # Decode
        rxBits = lteTurboDecode(softBits, nturbodecits=8)
        print(f"Decoded: {len(rxBits)} bits")

        # Check errors
        errors = np.sum(rxBits != txBits)
        ber = errors / K
        print(f"Bit errors: {errors} / {K} (BER = {ber:.4f})")

    except ImportError:
        print("turbo_encode.py not found")

    print()
    print("="*70)
    print("Max-Log-MAP SISO turbo decoder with Numba optimization")
    print("="*70)
