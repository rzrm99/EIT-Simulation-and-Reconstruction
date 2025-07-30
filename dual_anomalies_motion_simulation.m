


% % Create circular model with 16 electrodes
% imdl = mk_common_model('c2C2', 16);
% fmdl = imdl.fwd_model;
% 
% % Create time-varying images
% num_frames = 10;
% imgs(num_frames) = mk_image(fmdl, 1); % preallocate
% theta = linspace(0, 2*pi, num_frames+1); theta(end) = [];
% 
% for k = 1:num_frames
%     img = mk_image(fmdl, 1);
%     x = fmdl.nodes(:,1);
%     y = fmdl.nodes(:,2);
% 
%     % Moving anomaly 1
%     r1 = sqrt((x - 0.4*cos(theta(k))).^2 + (y - 0.4*sin(theta(k))).^2);
%     % Moving anomaly 2 (opposite direction)
%     r2 = sqrt((x + 0.4*cos(theta(k))).^2 + (y + 0.4*sin(theta(k))).^2);
% 
%     img.elem_data(r1 < 0.15) = 2;  % brighter
%     img.elem_data(r2 < 0.15) = 0.5; % darker
% 
%     imgs(k) = img;
% end
% 
% % Reference homogeneous image
% img_ref = mk_image(fmdl, 1);
% vh = fwd_solve(img_ref);
% 
% % Create noisy measurements and inverse model
% imdl = select_imdl(mk_common_model('c2C2', 16), {'Basic GN dif'});
% imdl.reconst_type = 'difference';
% imdl.solve = @inv_solve_diff_GN_one_step;
% imdl.hyperparameter.value = 1e-3;
% 
% % Noise level (5% of signal)
% noise_level = 0.05;
% 
% % Reconstruct and visualize all frames
% figure;
% rows = 3;
% cols = ceil(num_frames / rows);
% 
% for k = 1:num_frames
%     vi = fwd_solve(imgs(k));
% 
%     % Add Gaussian noise
%     noise = noise_level * std(vi.meas(:)) * randn(size(vi.meas));
%     vi.meas = vi.meas + noise;
% 
%     % Inverse reconstruction
%     img_rec = inv_solve(imdl, vh, vi);
% 
%     rows = 3;
%     cols = ceil(num_frames / rows);
%     subplot(rows, cols, k);
% 
%     show_fem(img_rec, 1);
%     title(sprintf('Frame %d', k));
% end
% sgtitle('TV Reconstruction of Moving Dual Anomalies');





% 
% --- SETUP ---
clc; clear; close all;

% Create model
imdl = mk_common_model('c2C2', 16);
fmdl = imdl.fwd_model;

% Configure inverse model
imdl.solve = @inv_solve_diff_GN_one_step;
imdl.reconst_type = 'difference';
imdl.hyperparameter.value = 1e-2;

% Time frames
num_frames = 10;
theta = linspace(0, 2*pi, num_frames+1); theta(end) = [];

imgs(num_frames) = mk_image(fmdl, 1);  % true phantom images
recs = cell(num_frames, 1);            % reconstructions

% Reference (homogeneous)
img_ref = mk_image(fmdl, 1);
vh = fwd_solve(img_ref);

% Noise level
noise_level = 0.03;

% --- SIMULATE + RECONSTRUCT ---
for k = 1:num_frames
    img = mk_image(fmdl, 1);
    x = fmdl.nodes(:,1); y = fmdl.nodes(:,2);

    r1 = sqrt((x - 0.4*cos(theta(k))).^2 + (y - 0.4*sin(theta(k))).^2);
    r2 = sqrt((x + 0.4*cos(theta(k))).^2 + (y + 0.4*sin(theta(k))).^2);

    img.elem_data(r1 < 0.15) = 2;
    img.elem_data(r2 < 0.15) = 0.5;
    imgs(k) = img;

    vi = fwd_solve(img);
    noise = noise_level * std(vi.meas(:)) * randn(size(vi.meas));
    vi.meas = vi.meas + noise;

    recs{k} = inv_solve(imdl, vh, vi);  % Use curly braces for cell array
end

% --- EXPORT VIDEO ---
v = VideoWriter('eit_GN.avi'); v.FrameRate = 2; open(v);
figure('Position', [100, 100, 1000, 500]);

for k = 1:num_frames
    subplot(1,2,1); show_fem(imgs(k), 1); title(sprintf('True (Frame %d)', k));
    subplot(1,2,2); show_fem(recs{k}, 1); title(sprintf('GN Recon (Frame %d)', k));
    writeVideo(v, getframe(gcf));
end

close(v);
disp('✅ Video saved as "eit_GN.avi"');