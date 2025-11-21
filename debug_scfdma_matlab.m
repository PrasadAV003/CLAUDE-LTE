% MATLAB SC-FDMA Debug Script
% Run this in MATLAB and compare outputs with Python version
clear all; clc;

fprintf('================================\n');
fprintf('MATLAB SC-FDMA Debug Output\n');
fprintf('================================\n\n');

% Configuration
ue.NULRB = 6;
ue.CyclicPrefixUL = 'Normal';
ue.Windowing = 0;  % No windowing for simplicity

% Simple test input: 14 symbols (2 slots), using fixed binary data
nSC = 72;
nSymbols = 14;  % 2 slots for Normal CP (7 symbols per slot)

% Fixed binary sequence for reproducibility
binary_str = '0000000100110010010001010111011011001101111111101000100110111010';
base_seq = double(binary_str - '0');  % Convert to 0s and 1s

% Convert binary to BPSK symbols (0→-1, 1→+1)
bpsk_seq = 2*base_seq - 1;

% Replicate to fill nSC subcarriers
full_seq = repmat(bpsk_seq, 1, ceil(nSC/length(bpsk_seq)));
full_seq = full_seq(1:nSC);

% Create grid with same sequence for all symbols
grid_in = repmat(full_seq(:), 1, nSymbols);

fprintf('=== INPUT ===\n');
fprintf('Grid shape: [%d, %d]\n', size(grid_in, 1), size(grid_in, 2));
fprintf('First 5 subcarriers of symbol 1: [%.4f, %.4f, %.4f, %.4f, %.4f]\n', ...
    grid_in(1:5, 1));
fprintf('\n');

% MODULATE using MATLAB
fprintf('=== MODULATION ===\n');
[waveform, info] = lteSCFDMAModulate(ue, grid_in);

fprintf('NFFT: %d\n', info.Nfft);
fprintf('Windowing: %d\n', info.Windowing);
fprintf('CP lengths: [');
fprintf('%d ', info.CyclicPrefixLengths);
fprintf(']\n');
fprintf('Waveform shape: [%d, %d]\n', size(waveform, 1), size(waveform, 2));
fprintf('Waveform[1:5]: [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(waveform(1)), imag(waveform(1)), ...
    real(waveform(2)), imag(waveform(2)), ...
    real(waveform(3)), imag(waveform(3)), ...
    real(waveform(4)), imag(waveform(4)), ...
    real(waveform(5)), imag(waveform(5)));
fprintf('Waveform power: %.6f\n', sum(abs(waveform).^2));
fprintf('\n');

% DEMODULATE using MATLAB
fprintf('=== DEMODULATION ===\n');
grid_out = lteSCFDMADemodulate(ue, waveform);

fprintf('Output grid shape: [%d, %d]\n', size(grid_out, 1), size(grid_out, 2));
fprintf('First 5 subcarriers of symbol 1: [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(grid_out(1, 1)), imag(grid_out(1, 1)), ...
    real(grid_out(2, 1)), imag(grid_out(2, 1)), ...
    real(grid_out(3, 1)), imag(grid_out(3, 1)), ...
    real(grid_out(4, 1)), imag(grid_out(4, 1)), ...
    real(grid_out(5, 1)), imag(grid_out(5, 1)));
fprintf('\n');

% ERROR CALCULATION
fprintf('=== ERROR ===\n');
error = grid_in - grid_out;
mse = mean(abs(error(:)).^2);
fprintf('MSE: %.6e\n', mse);
fprintf('Max error: %.6e\n', max(abs(error(:))));
fprintf('First 5 errors (symbol 1): [%.6e, %.6e, %.6e, %.6e, %.6e]\n', ...
    abs(error(1:5, 1)));
fprintf('\n');

if mse < 1e-20
    fprintf('✓ Perfect round-trip!\n');
elseif mse < 1e-10
    fprintf('✓ Excellent round-trip!\n');
else
    fprintf('✗ Round-trip has errors\n');
end

% MANUAL STEP-BY-STEP (Single symbol, no windowing)
fprintf('\n================================\n');
fprintf('MANUAL STEP-BY-STEP (Symbol 1)\n');
fprintf('================================\n\n');

nFFT = info.Nfft;
cpLength = info.CyclicPrefixLengths(1);
firstSC = (nFFT/2) - (nSC/2) + 1;  % MATLAB 1-indexed

fprintf('--- TX: Subcarrier Mapping ---\n');
freq_in = zeros(nFFT, 1);
freq_in(firstSC:(firstSC+nSC-1)) = grid_in(:, 1);
fprintf('Mapped to bins %d to %d (MATLAB 1-indexed)\n', firstSC, firstSC+nSC-1);
fprintf('Non-zero bins: [');
fprintf('%d ', find(abs(freq_in) > 0.5));
fprintf(']\n');
fprintf('freq_in[%d] = %.6f%+.6fi\n', firstSC, real(freq_in(firstSC)), imag(freq_in(firstSC)));
fprintf('\n');

fprintf('--- TX: fftshift ---\n');
freq_shifted = fftshift(freq_in, 1);
fprintf('After fftshift, non-zero bins: [');
fprintf('%d ', find(abs(freq_shifted) > 0.5));
fprintf(']\n');
fprintf('freq_shifted[1] = %.6f%+.6fi\n', real(freq_shifted(1)), imag(freq_shifted(1)));
fprintf('\n');

fprintf('--- TX: IFFT ---\n');
time_out = ifft(freq_shifted);
fprintf('time_out[1:3] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(time_out(1)), imag(time_out(1)), ...
    real(time_out(2)), imag(time_out(2)), ...
    real(time_out(3)), imag(time_out(3)));
fprintf('Time power: %.6f\n', sum(abs(time_out).^2));
fprintf('\n');

fprintf('--- TX: Add CP with half-SC shift ---\n');
extended = [time_out(end-cpLength+1:end); time_out];
phase_idx = double((-cpLength:(nFFT-1))');
phase = exp(1i * pi * (double(phase_idx) / double(nFFT)));
tx_waveform = extended .* phase;
fprintf('CP length: %d\n', cpLength);
fprintf('Extended length: %d\n', length(extended));
fprintf('tx_waveform[1:3] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(tx_waveform(1)), imag(tx_waveform(1)), ...
    real(tx_waveform(2)), imag(tx_waveform(2)), ...
    real(tx_waveform(3)), imag(tx_waveform(3)));
fprintf('\n');

fprintf('--- RX: Skip CP ---\n');
cpFraction = 0.55;
fftStart = fix(cpLength * cpFraction) + 1;  % MATLAB 1-indexed
samples = tx_waveform(fftStart:(fftStart+nFFT-1));
fprintf('fftStart: %d (MATLAB 1-indexed)\n', fftStart);
fprintf('Extract indices %d to %d\n', fftStart, fftStart+nFFT-1);
fprintf('samples[1:3] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(samples(1)), imag(samples(1)), ...
    real(samples(2)), imag(samples(2)), ...
    real(samples(3)), imag(samples(3)));
fprintf('\n');

fprintf('--- RX: Half-SC shift correction ---\n');
idx = double((0:(nFFT-1))');  % MATLAB: 0-indexed for this calculation
halfsc = exp(1i * pi * (idx + fftStart - 1 - cpLength) / double(nFFT));  % Adjust for MATLAB indexing
samples = samples .* halfsc;
fprintf('samples after halfsc[1:3] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(samples(1)), imag(samples(1)), ...
    real(samples(2)), imag(samples(2)), ...
    real(samples(3)), imag(samples(3)));
fprintf('\n');

fprintf('--- RX: FFT ---\n');
freq_rx = fft(samples);
fprintf('After FFT[1:3] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(freq_rx(1)), imag(freq_rx(1)), ...
    real(freq_rx(2)), imag(freq_rx(2)), ...
    real(freq_rx(3)), imag(freq_rx(3)));
fprintf('\n');

fprintf('--- RX: Phase correction (BEFORE fftshift) ---\n');
phaseCorr = exp(-1i * 2 * pi * ((cpLength - (fftStart-1)) * idx) / double(nFFT));  % Adjust for MATLAB indexing
freq_rx = freq_rx .* phaseCorr;
fprintf('After phase corr[1:3] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(freq_rx(1)), imag(freq_rx(1)), ...
    real(freq_rx(2)), imag(freq_rx(2)), ...
    real(freq_rx(3)), imag(freq_rx(3)));
fprintf('\n');

fprintf('--- RX: fftshift ---\n');
freq_rx_shifted = fftshift(freq_rx, 1);
fprintf('After fftshift[%d] = %.6f%+.6fi\n', firstSC, real(freq_rx_shifted(firstSC)), imag(freq_rx_shifted(firstSC)));
fprintf('\n');

fprintf('--- RX: Extract subcarriers ---\n');
grid_rx = freq_rx_shifted(firstSC:(firstSC+nSC-1));
fprintf('Extracted bins %d to %d\n', firstSC, firstSC+nSC-1);
fprintf('grid_rx[1:5] = [%.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi, %.6f%+.6fi]\n', ...
    real(grid_rx(1)), imag(grid_rx(1)), ...
    real(grid_rx(2)), imag(grid_rx(2)), ...
    real(grid_rx(3)), imag(grid_rx(3)), ...
    real(grid_rx(4)), imag(grid_rx(4)), ...
    real(grid_rx(5)), imag(grid_rx(5)));
fprintf('\n');

fprintf('--- COMPARISON ---\n');
manual_error = abs(grid_in(:, 1) - grid_rx);
fprintf('Manual MSE: %.6e\n', mean(manual_error.^2));
fprintf('Manual max error: %.6e\n', max(manual_error));
fprintf('First 5 errors: [%.6e, %.6e, %.6e, %.6e, %.6e]\n', manual_error(1:5));
