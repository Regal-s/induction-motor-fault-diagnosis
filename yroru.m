%% 6-CHANNEL UART TRANSMITTER - CONFIGURABLE VERSION
% Continuously transmits 16-bit ADC data from CSV via UART
% Channels: CH1, CH2, CH3, CH4, CH5, CH6 (V1, V2, V3, I1, I2, I3)
% Press Ctrl+C to stop
% Author: PMU System
% Date: 2025
% Updated: Configurable frequency, baud rate, and COM port

clear; clc; close all;


%% ======================= USER CONFIGURATION ========================
%  MODIFY THESE PARAMETERS AS NEEDED
%  ===================================================================

% -------- COMMUNICATION SETTINGS --------
PORT_NAME = 'COM23';          % Serial port (check Device Manager)
BAUD_RATE = 921600;          % Baud rate (2 Mbaud recommended for 15kHz)
                              % Options: 921600, 1000000, 2000000

% -------- SAMPLING SETTINGS --------
Fs = 2400;                   % Sample rate in Hz
                              % Options: 10000, 12000, 15000, etc.
                              % Note: Fs > 14285 requires > 2 Mbaud

% -------- DATA SOURCE --------
CSV_FILE = "C:\Users\nboruv\Downloads\pmu_test_vectors\combined_all_scenarios.csv";

% -------- PACKET FORMAT --------
SYNC_BYTE = uint8(hex2dec('AA'));
CHECK_BYTE = uint8(hex2dec('55'));

% -------- TRANSMISSION CONTROL --------
CONTINUOUS_MODE = true;       % true = loop forever, false = single pass
SHOW_PROGRESS = true;         % Show statistics every second
INTER_LOOP_PAUSE = 0;         % Pause between loops (seconds), 0 = none
BATCH_SIZE = 100;             % Packets per write (higher = more efficient)

%% ====================== CONFIGURATION VALIDATION ====================

% Calculate minimum required baud rate
% Formula: 14 bytes/packet * 10 bits/byte * Fs packets/sec
min_baud = 14 * 10 * Fs;
recommended_baud = ceil(min_baud * 1.15);  % 15% margin

fprintf('================================================================\n');
fprintf('        6-CHANNEL UART TRANSMITTER - CONFIGURABLE\n');
fprintf('================================================================\n\n');

fprintf('CONFIGURATION SUMMARY:\n');
fprintf('  COM Port:        %s\n', PORT_NAME);
fprintf('  Baud Rate:       %d baud (%.2f Mbaud)\n', BAUD_RATE, BAUD_RATE/1e6);
fprintf('  Sample Rate:     %d Hz\n', Fs);
fprintf('  Packet Size:     14 bytes\n');
fprintf('  Data Rate:       %.2f Mbaud required\n', min_baud/1e6);
fprintf('\n');

% Baud rate validation
if BAUD_RATE < min_baud
    fprintf('  *** WARNING ***\n');
    fprintf('  Configured baud rate (%d) is LESS than required (%d)!\n', BAUD_RATE, min_baud);
    fprintf('  Recommended: %d baud or higher\n', recommended_baud);
    fprintf('  Data WILL be lost at this configuration.\n\n');
    
    response = input('  Continue anyway? (y/n): ', 's');
    if ~strcmpi(response, 'y')
        fprintf('  Aborted. Please increase BAUD_RATE or decrease Fs.\n');
        return;
    end
else
    margin = (BAUD_RATE - min_baud) / min_baud * 100;
    fprintf('  Baud rate OK (%.1f%% margin)\n\n', margin);
end

% CP2108 limit warning
if BAUD_RATE > 2000000
    fprintf('  *** WARNING ***\n');
    fprintf('  CP2108 USB-UART bridge maxes out at ~2 Mbaud.\n');
    fprintf('  Baud rates above 2M may not work reliably.\n\n');
end

%% ======================== LOAD CSV DATA ========================

fprintf('Step 1: Loading CSV data...\n');

if ~isfile(CSV_FILE)
    error('CSV file not found: %s', CSV_FILE);
end

try
    data_table = readtable(CSV_FILE, 'VariableNamingRule', 'preserve');
    fprintf('  Column names: ');
    disp(data_table.Properties.VariableNames);
catch ME
    error('Failed to read CSV: %s', ME.message);
end

num_cols = width(data_table);
fprintf('  Number of columns: %d\n', num_cols);

if num_cols < 6
    error('Need at least 6 columns, found %d', num_cols);
end

% Extract channels
CH1 = data_table{:, 1};
CH2 = data_table{:, 2};
CH3 = data_table{:, 3};
CH4 = data_table{:, 4};
CH5 = data_table{:, 5};
CH6 = data_table{:, 6};

num_samples = length(CH1);
duration_sec = num_samples / Fs;

fprintf('  Loaded %d samples (%.2f seconds at %d Hz)\n', num_samples, duration_sec, Fs);
fprintf('  Power cycles at 50Hz: %.1f\n\n', duration_sec * 50);

%% ======================== DATA CONVERSION ========================

fprintf('Step 2: Converting to Q15 format...\n');

all_data = [CH1(:); CH2(:); CH3(:); CH4(:); CH5(:); CH6(:)];
data_min = min(all_data);
data_max = max(all_data);

fprintf('  Input range: %.4f to %.4f\n', data_min, data_max);

% Smart conversion based on data type
if isa(CH1, 'int16') || isa(CH1, 'uint16')
    fprintf('  Data is already 16-bit integer\n');
    CH1_Q15 = int16(CH1); CH2_Q15 = int16(CH2); CH3_Q15 = int16(CH3);
    CH4_Q15 = int16(CH4); CH5_Q15 = int16(CH5); CH6_Q15 = int16(CH6);
    
elseif data_max <= 32767 && data_min >= -32768 && all(CH1 == round(CH1))
    fprintf('  Data appears to be Q15 integers\n');
    CH1_Q15 = int16(round(CH1)); CH2_Q15 = int16(round(CH2)); CH3_Q15 = int16(round(CH3));
    CH4_Q15 = int16(round(CH4)); CH5_Q15 = int16(round(CH5)); CH6_Q15 = int16(round(CH6));
    
elseif data_max <= 65535 && data_min >= 0 && all(CH1 == round(CH1))
    fprintf('  Data is unsigned 16-bit, converting to signed\n');
    CH1_Q15 = int16(round(CH1) - 32768); CH2_Q15 = int16(round(CH2) - 32768);
    CH3_Q15 = int16(round(CH3) - 32768); CH4_Q15 = int16(round(CH4) - 32768);
    CH5_Q15 = int16(round(CH5) - 32768); CH6_Q15 = int16(round(CH6) - 32768);
else
    fprintf('  Data is floating point, normalizing to Q15\n');
    data_range = data_max - data_min;
    normalize = @(x) 2*(x - data_min)/data_range - 1;
    Q15_scale = 32767;
    CH1_Q15 = int16(round(normalize(CH1) * Q15_scale));
    CH2_Q15 = int16(round(normalize(CH2) * Q15_scale));
    CH3_Q15 = int16(round(normalize(CH3) * Q15_scale));
    CH4_Q15 = int16(round(normalize(CH4) * Q15_scale));
    CH5_Q15 = int16(round(normalize(CH5) * Q15_scale));
    CH6_Q15 = int16(round(normalize(CH6) * Q15_scale));
end

fprintf('  Q15 output range: %d to %d\n\n', ...
    min([CH1_Q15; CH2_Q15; CH3_Q15; CH4_Q15; CH5_Q15; CH6_Q15]), ...
    max([CH1_Q15; CH2_Q15; CH3_Q15; CH4_Q15; CH5_Q15; CH6_Q15]));

%% ======================== BUILD PACKETS ========================

fprintf('Step 3: Building %d packets...\n', num_samples);

% Packet: SYNC(1) + CH1(2) + CH2(2) + CH3(2) + CH4(2) + CH5(2) + CH6(2) + CHECK(1) = 14 bytes
all_packets = zeros(num_samples, 14, 'uint8');

for i = 1:num_samples
    all_packets(i, 1) = SYNC_BYTE;
    
    temp_val = typecast(CH1_Q15(i), 'uint16');
    all_packets(i, 2) = uint8(bitshift(temp_val, -8));
    all_packets(i, 3) = uint8(bitand(temp_val, uint16(255)));
    
    temp_val = typecast(CH2_Q15(i), 'uint16');
    all_packets(i, 4) = uint8(bitshift(temp_val, -8));
    all_packets(i, 5) = uint8(bitand(temp_val, uint16(255)));
    
    temp_val = typecast(CH3_Q15(i), 'uint16');
    all_packets(i, 6) = uint8(bitshift(temp_val, -8));
    all_packets(i, 7) = uint8(bitand(temp_val, uint16(255)));
    
    temp_val = typecast(CH4_Q15(i), 'uint16');
    all_packets(i, 8) = uint8(bitshift(temp_val, -8));
    all_packets(i, 9) = uint8(bitand(temp_val, uint16(255)));
    
    temp_val = typecast(CH5_Q15(i), 'uint16');
    all_packets(i, 10) = uint8(bitshift(temp_val, -8));
    all_packets(i, 11) = uint8(bitand(temp_val, uint16(255)));
    
    temp_val = typecast(CH6_Q15(i), 'uint16');
    all_packets(i, 12) = uint8(bitshift(temp_val, -8));
    all_packets(i, 13) = uint8(bitand(temp_val, uint16(255)));
    
    all_packets(i, 14) = CHECK_BYTE;
end

fprintf('  First packet: '); fprintf('%02X ', all_packets(1, :)); fprintf('\n');
fprintf('  Last packet:  '); fprintf('%02X ', all_packets(end, :)); fprintf('\n\n');

%% ======================== OPEN SERIAL PORT ========================

fprintf('Step 4: Opening %s at %d baud...\n', PORT_NAME, BAUD_RATE);

delete(instrfindall);

try
    s = serialport(PORT_NAME, BAUD_RATE);
    s.Timeout = 5;
    try s.OutputBufferSize = 65536; catch, end
    fprintf('  Port opened successfully\n\n');
catch ME
    error('Failed to open %s: %s\nCheck Device Manager!', PORT_NAME, ME.message);
end

flush(s);
pause(0.2);

%% ======================== TRANSMISSION ========================

fprintf('================================================================\n');
fprintf('STARTING TRANSMISSION: %d Hz, %.2f Mbaud\n', Fs, BAUD_RATE/1e6);
fprintf('================================================================\n');
fprintf('Press Ctrl+C to stop\n\n');

packet_period = 1/Fs;
loop_count = 0;
total_packets = 0;
total_bytes = 0;
start_time = tic;
last_update = tic;

try
    while CONTINUOUS_MODE
        loop_count = loop_count + 1;
        i = 1;
        
        while i <= num_samples
            batch_start = tic;
            batch_end = min(i + BATCH_SIZE - 1, num_samples);
            batch_count = batch_end - i + 1;
            
            batch_data = reshape(all_packets(i:batch_end, :)', 1, []);
            write(s, batch_data, "uint8");
            
            total_packets = total_packets + batch_count;
            total_bytes = total_bytes + batch_count * 14;
            
            elapsed = toc(batch_start);
            target_time = batch_count * packet_period;
            if target_time > elapsed
                pause(target_time - elapsed);
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            end
            
            i = batch_end + 1;
        end
        
        if SHOW_PROGRESS && toc(last_update) >= 1.0
            elapsed_total = toc(start_time);
            rate = total_packets / elapsed_total;
            throughput = (total_bytes * 8) / (elapsed_total * 1e6);
            
            fprintf('Loop: %5d | Packets: %10d | Rate: %8.1f pkt/s | %.2f Mbps\n', ...
                loop_count, total_packets, rate, throughput);
            last_update = tic;
        end
        
        if INTER_LOOP_PAUSE > 0, pause(INTER_LOOP_PAUSE); end
    end
catch ME
    if strcmp(ME.identifier, 'MATLAB:interruption')
        fprintf('\n\nStopped by user (Ctrl+C)\n');
    else
        fprintf('\n\nError: %s\n', ME.message);
    end
end

%% ======================== SUMMARY ========================

elapsed_total = toc(start_time);
actual_rate = total_packets / elapsed_total;
actual_throughput = (total_bytes * 8) / (elapsed_total * 1e6);

fprintf('\n================================================================\n');
fprintf('TRANSMISSION SUMMARY\n');
fprintf('================================================================\n');
fprintf('Configuration:       %s @ %d baud, %d Hz\n', PORT_NAME, BAUD_RATE, Fs);
fprintf('Total loops:         %d\n', loop_count);
fprintf('Total packets:       %d\n', total_packets);
fprintf('Total bytes:         %d (%.2f MB)\n', total_bytes, total_bytes/1e6);
fprintf('Duration:            %.2f seconds\n', elapsed_total);
fprintf('Actual rate:         %.1f packets/sec (target: %d)\n', actual_rate, Fs);
fprintf('Actual throughput:   %.2f Mbps\n', actual_throughput);
fprintf('Efficiency:          %.1f%%\n', actual_rate / Fs * 100);
fprintf('================================================================\n');

try flush(s); delete(s); clear s; fprintf('\nPort closed.\n'); catch, end