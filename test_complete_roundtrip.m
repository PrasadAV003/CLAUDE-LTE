%% LTE Channel Coding - Complete Round-Trip Test
% Tests encode-decode pipeline for 40, 1088, and 6145 bit sequences
% Saves ALL intermediate results to a single comprehensive file

function test_complete_roundtrip()
    % Base sequence (64 bits)
    binary_str = '0000000100110010010001010111011011001101111111101000100110111010';
    base_seq = int8(arrayfun(@str2num, num2cell(binary_str)));

    % Test sizes: 40, 1088, 6145 (not 6144 to trigger multi-block segmentation)
    test_sizes = [40, 1088, 6145];

    % Open single output file for all results
    output_filename = 'complete_roundtrip_results_matlab.txt';
    fid = fopen(output_filename, 'w');

    % Write header
    fprintf(fid, '%s\n', repmat('=', 1, 80));
    fprintf(fid, 'LTE CHANNEL CODING - COMPLETE ROUND-TRIP TEST RESULTS (MATLAB)\n');
    fprintf(fid, '%s\n', repmat('=', 1, 80));
    fprintf(fid, 'Test date: %s\n', datestr(now));
    fprintf(fid, 'Test cases: %d\n', length(test_sizes));
    fprintf(fid, 'Test sizes: %s\n', mat2str(test_sizes));
    fprintf(fid, '\n');
    fprintf(fid, 'Base sequence (64 bits):\n');
    fprintf(fid, '  Binary: %s\n', binary_str);
    fprintf(fid, '\n');
    fprintf(fid, 'Complete pipeline tested:\n');
    fprintf(fid, '  ENCODE: CRC → Segment → Turbo → Rate Match\n');
    fprintf(fid, '  CHANNEL: Perfect (no noise)\n');
    fprintf(fid, '  DECODE: Rate Recover → Turbo → Desegment → CRC\n');
    fprintf(fid, '\n');
    fprintf(fid, '%s\n', repmat('=', 1, 80));

    % Run tests
    results_summary = {};

    for i = 1:length(test_sizes)
        size_val = test_sizes(i);
        fprintf('Running test for %d bits...\n', size_val);

        % Create test sequence
        test_seq = create_test_sequence(base_seq, size_val);

        % Run round-trip test
        test_name = sprintf('%d_bits', size_val);
        [decoded, errors, status] = run_complete_roundtrip_test(test_seq, test_name, fid);

        results_summary{i} = struct('size', size_val, 'errors', errors, 'status', status);

        if errors >= 0
            fprintf('  Completed: %s (%d errors)\n', status, errors);
        else
            fprintf('  Completed: %s\n', status);
        end
    end

    % Write summary at the end
    fprintf(fid, '\n\n%s\n', repmat('=', 1, 80));
    fprintf(fid, 'OVERALL SUMMARY\n');
    fprintf(fid, '%s\n', repmat('=', 1, 80));
    fprintf(fid, '\n');

    fprintf(fid, '%-15s %-15s %-15s\n', 'Size', 'Errors', 'Status');
    fprintf(fid, '%s\n', repmat('-', 1, 45));

    for i = 1:length(results_summary)
        r = results_summary{i};
        if r.errors >= 0
            error_str = sprintf('%d', r.errors);
        else
            error_str = 'N/A';
        end
        fprintf(fid, '%-15d %-15s %-15s\n', r.size, error_str, r.status);
    end

    fprintf(fid, '\n');

    % Overall result
    all_passed = true;
    for i = 1:length(results_summary)
        if results_summary{i}.errors > 0
            all_passed = false;
            break;
        end
    end

    if all_passed
        fprintf(fid, '✓ ALL TESTS PASSED - PERFECT RECONSTRUCTION!\n');
    else
        fprintf(fid, '✗ SOME TESTS FAILED - See details above\n');
    end

    fprintf(fid, '\n%s\n', repmat('=', 1, 80));
    fprintf(fid, 'END OF REPORT\n');
    fprintf(fid, '%s\n', repmat('=', 1, 80));

    fclose(fid);

    fprintf('\nAll results saved to: %s\n', output_filename);
    fprintf('\nSummary:\n');
    fprintf('%-15s %-15s %-15s\n', 'Size', 'Errors', 'Status');
    fprintf('%s\n', repmat('-', 1, 45));
    for i = 1:length(results_summary)
        r = results_summary{i};
        if r.errors >= 0
            fprintf('%-15d %-15d %-15s\n', r.size, r.errors, r.status);
        else
            fprintf('%-15d %-15s %-15s\n', r.size, 'N/A', r.status);
        end
    end
end


function test_seq = create_test_sequence(base_sequence, target_length)
    % Create test sequence by repeating base sequence
    base_len = length(base_sequence);

    if target_length <= base_len
        test_seq = base_sequence(1:target_length);
    else
        repeats = ceil(target_length / base_len);
        extended = repmat(base_sequence, repeats, 1);
        test_seq = extended(1:target_length);
    end
end


function save_array_to_file(fid, name, data, show_all)
    % Save array data to file with formatting
    if nargin < 4
        show_all = false;
    end

    fprintf(fid, '\n%s:\n', name);
    fprintf(fid, '  Length: %d\n', length(data));
    fprintf(fid, '  Data type: %s\n', class(data));

    if show_all || length(data) <= 200
        % Show all data for short sequences
        if length(data) <= 50
            fprintf(fid, '  Data: %s\n', mat2str(data(:)'));
        else
            fprintf(fid, '  Data: [%s ... ] (%d elements)\n', ...
                mat2str(data(1:min(20,end))'), length(data));
        end
    else
        % Show first and last 100 for long sequences
        fprintf(fid, '  First 50: %s\n', mat2str(data(1:50)'));
        fprintf(fid, '  Last 50: %s\n', mat2str(data(end-49:end)'));
    end
end


function [decoded_data, errors, result] = run_complete_roundtrip_test(test_bits, test_name, fid)
    % Complete round-trip test with all intermediate stages logged

    fprintf(fid, '\n%s\n', repmat('=', 1, 80));
    fprintf(fid, 'TEST: %s\n', test_name);
    fprintf(fid, '%s\n', repmat('=', 1, 80));
    fprintf(fid, 'Test date: %s\n', datestr(now));
    fprintf(fid, 'Input length: %d bits\n', length(test_bits));
    fprintf(fid, '\n');

    % ========================================================================
    % ENCODE CHAIN
    % ========================================================================

    fprintf(fid, '%s\n', repmat('-', 1, 80));
    fprintf(fid, 'ENCODING CHAIN\n');
    fprintf(fid, '%s\n\n', repmat('-', 1, 80));

    % Stage 1: Original input
    fprintf(fid, 'STAGE 1: ORIGINAL INPUT\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    save_array_to_file(fid, 'Original bits', test_bits, true);
    fprintf(fid, '  Ones: %d\n', sum(test_bits == 1));
    fprintf(fid, '  Zeros: %d\n', sum(test_bits == 0));

    % Stage 2: CRC-24A attachment
    fprintf(fid, '\n\nSTAGE 2: CRC-24A ATTACHMENT (Transport Block CRC)\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    data_with_crc = lteCRCEncode(test_bits, '24A');

    fprintf(fid, 'Input length: %d bits\n', length(test_bits));
    fprintf(fid, 'CRC length: 24 bits\n');
    fprintf(fid, 'Output length: %d bits\n', length(data_with_crc));
    save_array_to_file(fid, 'Data with CRC', data_with_crc);
    save_array_to_file(fid, 'CRC bits only', data_with_crc(end-23:end), true);

    % Stage 3: Code block segmentation
    fprintf(fid, '\n\nSTAGE 3: CODE BLOCK SEGMENTATION\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    code_blocks = lteCodeBlockSegment(data_with_crc);

    if iscell(code_blocks)
        C = length(code_blocks);
        fprintf(fid, 'Number of blocks (C): %d\n', C);

        for idx = 1:C
            cb = code_blocks{idx};
            fprintf(fid, '\n  Code Block %d:\n', idx-1);
            save_array_to_file(fid, sprintf('    Block %d data', idx-1), cb);
            fprintf(fid, '    Filler bits (-1): %d\n', sum(cb == -1));
        end
    else
        C = 1;
        fprintf(fid, 'Number of blocks (C): 1\n');
        save_array_to_file(fid, '  Single block data', code_blocks);
        fprintf(fid, '  Filler bits (-1): %d\n', sum(code_blocks == -1));
    end

    % Stage 4: Turbo encoding
    fprintf(fid, '\n\nSTAGE 4: TURBO ENCODING\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    turbo_encoded = lteTurboEncode(code_blocks);

    if iscell(turbo_encoded)
        fprintf(fid, 'Number of blocks: %d\n', length(turbo_encoded));
        fprintf(fid, 'Rate: 1/3 (each K bits → 3*(K+4) bits)\n');

        for idx = 1:length(turbo_encoded)
            te = turbo_encoded{idx};
            fprintf(fid, '\n  Turbo Encoded Block %d:\n', idx-1);
            fprintf(fid, '    Length: %d\n', length(te));
            save_array_to_file(fid, sprintf('    Block %d encoded', idx-1), te);
            fprintf(fid, '    NULL bits (-1): %d\n', sum(te == -1));
        end
        total_turbo_bits = sum(cellfun(@length, turbo_encoded));
    else
        fprintf(fid, 'Output length: %d bits\n', length(turbo_encoded));
        save_array_to_file(fid, 'Turbo encoded data', turbo_encoded);
        fprintf(fid, 'NULL bits (-1): %d\n', sum(turbo_encoded == -1));
        total_turbo_bits = length(turbo_encoded);
    end

    % Stage 5: Rate matching
    fprintf(fid, '\n\nSTAGE 5: RATE MATCHING\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));

    E_total = floor(total_turbo_bits * 1.2);
    rate_matched = lteRateMatchTurbo(turbo_encoded, E_total, 0);

    fprintf(fid, 'Redundancy version (rv): 0\n');
    fprintf(fid, 'Target output length (E): %d\n', E_total);
    fprintf(fid, 'Actual output length: %d\n', length(rate_matched));
    save_array_to_file(fid, 'Rate matched data', rate_matched);

    % ========================================================================
    % CHANNEL SIMULATION
    % ========================================================================

    fprintf(fid, '\n\n%s\n', repmat('-', 1, 80));
    fprintf(fid, 'CHANNEL TRANSMISSION\n');
    fprintf(fid, '%s\n\n', repmat('-', 1, 80));

    % Convert to soft values (LLRs)
    % LLR convention: positive = bit 0, negative = bit 1
    received_soft = double(rate_matched);
    received_soft(rate_matched == 0) = 5.0;
    received_soft(rate_matched == 1) = -5.0;

    fprintf(fid, 'Channel type: Perfect (no noise)\n');
    fprintf(fid, 'LLR mapping: bit 0 → +5.0, bit 1 → -5.0\n');
    fprintf(fid, 'Transmitted bits: %d\n', length(rate_matched));
    fprintf(fid, 'Received soft values: %d\n', length(received_soft));

    % ========================================================================
    % DECODE CHAIN
    % ========================================================================

    fprintf(fid, '\n\n%s\n', repmat('-', 1, 80));
    fprintf(fid, 'DECODING CHAIN\n');
    fprintf(fid, '%s\n\n', repmat('-', 1, 80));

    % Stage 6: Rate recovery
    fprintf(fid, 'STAGE 6: RATE RECOVERY (De-Rate-Match)\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    rate_recovered = lteRateRecoverTurbo(received_soft, length(test_bits), 0);

    if iscell(rate_recovered)
        fprintf(fid, 'Number of blocks: %d\n', length(rate_recovered));
        for idx = 1:length(rate_recovered)
            fprintf(fid, '  Block %d length: %d\n', idx-1, length(rate_recovered{idx}));
        end
    else
        fprintf(fid, 'Output length: %d soft bits\n', length(rate_recovered));
    end

    % Stage 7: Turbo decoding
    fprintf(fid, '\n\nSTAGE 7: TURBO DECODING (Max-Log-MAP)\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    turbo_decoded = lteTurboDecode(rate_recovered, 5);

    if iscell(turbo_decoded)
        fprintf(fid, 'Number of blocks: %d\n', length(turbo_decoded));
        fprintf(fid, 'Iterations: 5\n');
        fprintf(fid, 'Algorithm: Max-Log-MAP\n');
        for idx = 1:length(turbo_decoded)
            fprintf(fid, '  Block %d length: %d\n', idx-1, length(turbo_decoded{idx}));
        end
    else
        fprintf(fid, 'Output length: %d bits\n', length(turbo_decoded));
        fprintf(fid, 'Iterations: 5\n');
        fprintf(fid, 'Algorithm: Max-Log-MAP\n');
    end

    % Stage 8: Code block desegmentation
    fprintf(fid, '\n\nSTAGE 8: CODE BLOCK DESEGMENTATION\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));
    [desegmented, crc_errors] = lteCodeBlockDesegment(turbo_decoded, length(test_bits));

    fprintf(fid, 'Expected output length: %d bits (data + transport CRC)\n', length(test_bits) + 24);
    fprintf(fid, 'Actual output length: %d bits\n', length(desegmented));

    if ~isempty(crc_errors)
        fprintf(fid, 'CRC errors per block: %s\n', mat2str(crc_errors));
        if all(crc_errors == 0)
            fprintf(fid, '✓ All block CRCs passed\n');
        else
            fprintf(fid, '✗ Some block CRCs failed\n');
        end
    else
        fprintf(fid, 'CRC errors: N/A (single block)\n');
    end

    % Stage 9: CRC-24A decoding
    fprintf(fid, '\n\nSTAGE 9: CRC-24A DECODING (Transport Block CRC)\n');
    fprintf(fid, '%s\n', repmat('-', 1, 40));

    [decoded_data, crc_error] = lteCRCDecode(desegmented, '24A');

    fprintf(fid, 'Input length: %d bits\n', length(desegmented));
    fprintf(fid, 'CRC length: 24 bits\n');
    fprintf(fid, 'Output length: %d bits\n', length(decoded_data));
    fprintf(fid, 'CRC error value: %d\n', crc_error);
    if crc_error == 0
        fprintf(fid, 'CRC status: PASS\n');
    else
        fprintf(fid, 'CRC status: FAIL\n');
    end
    save_array_to_file(fid, 'Final decoded data', decoded_data, true);

    % ========================================================================
    % VERIFICATION
    % ========================================================================

    fprintf(fid, '\n\n%s\n', repmat('=', 1, 80));
    fprintf(fid, 'VERIFICATION RESULTS\n');
    fprintf(fid, '%s\n\n', repmat('=', 1, 80));

    if length(decoded_data) == length(test_bits)
        errors = sum(decoded_data ~= test_bits);
        ber = errors / length(test_bits);

        fprintf(fid, 'Original length: %d bits\n', length(test_bits));
        fprintf(fid, 'Decoded length: %d bits\n', length(decoded_data));
        fprintf(fid, 'Bit errors: %d\n', errors);
        fprintf(fid, 'Bit Error Rate (BER): %.6f\n', ber);

        if errors == 0
            fprintf(fid, '\n✓ PERFECT MATCH - NO ERRORS!\n');
            result = 'PASS';
        else
            fprintf(fid, '\n✗ ERRORS DETECTED - %d bit errors\n', errors);
            error_positions = find(decoded_data ~= test_bits);
            fprintf(fid, 'Error positions (first 20): %s\n', mat2str(error_positions(1:min(20,end))'));

            % Show comparison for first 10 errors
            fprintf(fid, '\nError details (first 10):\n');
            for i = 1:min(10, length(error_positions))
                pos = error_positions(i);
                fprintf(fid, '  Position %d: Expected %d, Got %d\n', ...
                    pos, test_bits(pos), decoded_data(pos));
            end
            result = 'FAIL';
        end
    else
        fprintf(fid, '✗ LENGTH MISMATCH!\n');
        fprintf(fid, 'Original length: %d bits\n', length(test_bits));
        fprintf(fid, 'Decoded length: %d bits\n', length(decoded_data));
        errors = -1;
        result = 'LENGTH_MISMATCH';
    end

    fprintf(fid, '\n%s\n\n', repmat('=', 1, 80));
end
