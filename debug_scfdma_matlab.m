% Simple SC-FDMA Round-trip Test - Matching MATLAB Example
% Based on MATLAB lteSCFDMAModulate documentation

clear all;
close all;

fprintf('============================================================\n');
fprintf('MATLAB SC-FDMA Debug Output\n');
fprintf('============================================================\n\n');

% UE configuration - Basic LTE
ue = struct();
ue.NULRB = 6;           % 6 RBs = 72 subcarriers
ue.CyclicPrefixUL = 'Normal';
ue.NTxAnts = 1;

% Create resource grid
fprintf('=== INPUT ===\n');
nSC = ue.NULRB * 12;    % 72 subcarriers
nSymbols = 14;          % 1 subframe (2 slots)

% Generate QPSK symbols
rng(42);  % Fixed seed for reproducibility
nBits = nSC * nSymbols * 2;  % 2 bits per QPSK symbol
bits = randi([0,1], nBits, 1);

% QPSK modulation - MATLAB style
symbols = lteSymbolModulate(bits, 'QPSK');

% Reshape into grid [nSC x nSymbols]
grid = reshape(symbols, nSC, nSymbols);

fprintf('Grid shape: [%d, %d]\n', size(grid, 1), size(grid, 2));
fprintf('First 5 subcarriers of symbol 1:\n');
disp(grid(1:5, 1).');
fprintf('Grid power: %.6f\n', sum(abs(grid(:)).^2));
fprintf('\n');

% MODULATE
fprintf('=== MODULATION ===\n');
waveform = lteSCFDMAModulate(ue, grid);
fprintf('Waveform length: %d\n', length(waveform));
fprintf('First 5 samples:\n');
disp(waveform(1:5).');
fprintf('Waveform power: %.6f\n', sum(abs(waveform).^2));
fprintf('\n');

% DEMODULATE
fprintf('=== DEMODULATION ===\n');
grid_rx = lteSCFDMADemodulate(ue, waveform);
fprintf('Output grid shape: [%d, %d]\n', size(grid_rx, 1), size(grid_rx, 2));
fprintf('First 5 subcarriers of symbol 1:\n');
disp(grid_rx(1:5, 1).');
fprintf('Output power: %.6f\n', sum(abs(grid_rx(:)).^2));
fprintf('\n');

% ERROR CALCULATION
fprintf('=== ERROR ===\n');
error = grid - grid_rx;
mse = mean(abs(error(:)).^2);
max_error = max(abs(error(:)));
fprintf('MSE: %.6e\n', mse);
fprintf('Max error: %.6e\n', max_error);
fprintf('First 5 errors (symbol 1): ');
fprintf('%.6f ', abs(error(1:5, 1)));
fprintf('\n\n');

if mse < 1e-20
    fprintf('✓ Perfect round-trip!\n');
elseif mse < 1e-10
    fprintf('✓ Good round-trip (acceptable numerical error)\n');
else
    fprintf('✗ Round-trip has significant errors\n');
end

fprintf('\n============================================================\n');
