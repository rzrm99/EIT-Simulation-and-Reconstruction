
% --- SETUP ---
clc; clear; close all;

% Thorax-shaped model with 16 electrodes
imdl = mk_common_model('a2C', 16);  % 'a2C' = adult thorax, circular electrodes
fmdl = imdl.fwd_model;

% Inverse model setup (Gauss-Newton)
imdl.solve = @inv_solve_diff_GN_one_step;
imdl.reconst_type = 'difference';
imdl.hyperparameter.value = 1e-2;

% Time frames for breathing cycle
num_frames = 10;
breath = linspace(1.5, 0.5, num_frames);  % simulate exhale → inhale

% Store frames
imgs(num_frames) = mk_image(fmdl, 1);
recs = cell(num_frames, 1);

% Reference homogeneous conductivity (fully exhaled)
img_ref = mk_image(fmdl, 1);
vh = fwd_solve(img_ref);

% Coordinates
x = fmdl.nodes(:,1); y = fmdl.nodes(:,2);

% Noise level
noise_level = 0.03;

% --- SIMULATE BREATHING CYCLE ---
for k = 1:num_frames
    img = mk_image(fmdl, 1);  % start with background = 1

    % Left lung ellipse: centered at (-0.3, 0), radius ~0.2 x 0.4
    rL = ((x + 0.3)/0.2).^2 + (y/0.4).^2;
    img.elem_data(rL < 1) = breath(k);  % change over time

    % Right lung ellipse: centered at (+0.3, 0)
    rR = ((x - 0.3)/0.2).^2 + (y/0.4).^2;
    img.elem_data(rR < 1) = breath(k);

    imgs(k) = img;

    % Forward solve + noise
    vi = fwd_solve(img);
    noise = noise_level * std(vi.meas(:)) * randn(size(vi.meas));
    vi.meas = vi.meas + noise;

    % Inverse reconstruction
    recs{k} = inv_solve(imdl, vh, vi);
end

% --- EXPORT VIDEO ---
v = VideoWriter('eit_lung_sim.avi'); v.FrameRate = 2; open(v);
figure('Position', [100, 100, 1000, 500]);

for k = 1:num_frames
    subplot(1,2,1); show_fem(imgs(k), 1);
    title(sprintf('True Conductivity (Frame %d)', k));

    subplot(1,2,2); show_fem(recs{k}, 1);
    title(sprintf('GN Recon (Frame %d)', k));

    writeVideo(v, getframe(gcf));
end
close(v);
disp('✅ Lung simulation video saved as "eit_lung_sim.avi"');
