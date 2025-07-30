% Create a simple circular model
imdl = mk_common_model('c2c2', 16);  % 16 electrodes on a circular tank
fmdl = imdl.fwd_model;

% Add a circular anomaly
img = mk_image(fmdl, 1);  % background conductivity = 1
x = fmdl.nodes(:,1);
y = fmdl.nodes(:,2);
r = sqrt((x - 0.4).^2 + (y - 0.4).^2);
img.elem_data(r < 0.2) = 2;  % anomaly with conductivity = 2

% Simulate voltage data
vh = fwd_solve(mk_image(fmdl, 1));  % homogeneous data
vi = fwd_solve(img);                % inhomogeneous data

% Assign simulated data to inverse model
imdl.fwd_model = fmdl;
imdl = select_imdl(imdl, {'Basic GN dif'});
imdl.solve = @inv_solve_diff_GN_one_step;
imdl.hyperparameter.value = 1e-2;

% Run inverse solution
img_rec = inv_solve(imdl, vh, vi);

% Display results
subplot(1,3,1);
show_fem(img); title('True Conductivity');

subplot(1,3,2);
show_fem(img_rec); title('Reconstructed Image');

subplot(1,3,3);
plot(vi.meas - vh.meas); title('Voltage Difference');
xlabel('Measurement Index'); ylabel('Voltage (V)');
