% SC-FDMA Round-trip Test with Fixed Binary Data
% Standard LTE uplink modulation/demodulation test

clear all;
close all;
clc;

fprintf('============================================================\n');
fprintf('MATLAB SC-FDMA Debug Output (Fixed Binary Data)\n');
fprintf('============================================================\n\n');

% Fixed binary data sequence
binary_str = '0000000100110010010001010111011011001101111111101000100110111010';
fprintf('Binary sequence: %s\n', binary_str);
fprintf('Binary length: %d bits\n\n', length(binary_str));

% UE configuration - Standard LTE
ue = struct();
ue.NULRB = 1;                    % 1 RB = 12 subcarriers
ue.CyclicPrefixUL = 'Normal';
ue.NTxAnts = 1;

% Resource grid dimensions
nSC = ue.NULRB * 12;    % 12 subcarriers
nSymbols = 7;           % 7 OFDM symbols (1 slot, Normal CP)

% Convert binary string to array
nBits = nSC * nSymbols * 2;  % QPSK: 2 bits per symbol
fprintf('Required bits for QPSK: %d\n', nBits);

% Use the binary sequence (repeat if needed)
bits_needed = nBits;
binary_data = repmat(binary_str, 1, ceil(bits_needed/length(binary_str)));
binary_data = binary_data(1:bits_needed);

% Convert to numeric array
bits = zeros(bits_needed, 1);
for i = 1:bits_needed
    bits(i) = str2double(binary_data(i));
end

fprintf('Using first %d bits from sequence\n\n', bits_needed);

% QPSK modulation
symbols = lteSymbolModulate(bits, 'QPSK');

% Reshape into grid [nSC x nSymbols]
grid = reshape(symbols, nSC, nSymbols);

fprintf('=== INPUT GRID ===\n');
fprintf('Grid shape: [%d, %d]\n', size(grid, 1), size(grid, 2));
fprintf('Grid power: %.6f\n', sum(abs(grid(:)).^2));
fprintf('First 5 subcarriers of symbol 1:\n');
for i = 1:5
    fprintf('  grid(%d,1) = %.6f%+.6fi\n', i, real(grid(i,1)), imag(grid(i,1)));
end
fprintf('\n');

% SC-FDMA MODULATION
fprintf('=== MODULATION ===\n');
waveform = lteSCFDMAModulate(ue, grid);
fprintf('Waveform length: %d samples\n', length(waveform));
fprintf('Waveform power: %.6f\n', sum(abs(waveform).^2));
fprintf('First 5 waveform samples:\n');
for i = 1:5
    fprintf('  waveform(%d) = %.6f%+.6fi\n', i, real(waveform(i)), imag(waveform(i)));
end
fprintf('\n');

% SC-FDMA DEMODULATION
fprintf('=== DEMODULATION ===\n');
grid_rx = lteSCFDMADemodulate(ue, waveform);
fprintf('Recovered grid shape: [%d, %d]\n', size(grid_rx, 1), size(grid_rx, 2));
fprintf('Recovered power: %.6f\n', sum(abs(grid_rx(:)).^2));
fprintf('First 5 subcarriers of symbol 1:\n');
for i = 1:5
    fprintf('  grid_rx(%d,1) = %.6f%+.6fi\n', i, real(grid_rx(i,1)), imag(grid_rx(i,1)));
end
fprintf('\n');

% ERROR ANALYSIS
fprintf('=== ERROR ANALYSIS ===\n');
error = grid - grid_rx;
mse = mean(abs(error(:)).^2);
max_error = max(abs(error(:)));
fprintf('MSE: %.6e\n', mse);
fprintf('Max error: %.6e\n', max_error);
fprintf('First 5 errors (magnitude):\n');
for i = 1:5
    fprintf('  |error(%d,1)| = %.6e\n', i, abs(error(i,1)));
end
fprintf('\n');

if mse < 1e-20
    fprintf('✓ PERFECT round-trip!\n');
elseif mse < 1e-10
    fprintf('✓ Good round-trip (acceptable numerical error)\n');
else
    fprintf('✗ FAILED - Round-trip has significant errors\n');
end

fprintf('\n============================================================\n');
