%[T, timing] = load_bpvo_result_mat('kitti_intensity_bpvo.mat');
load('kitti_intensity_bpvo.mat', 'T', 'timing', 'iters');

T_intensity = T;
timing_intensity = timing;
iters_inensity = iters;

e_intensity = kitti_eval(T_intensity);

load('kitti_bitplanes_bpvo.mat', 'T', 'timing', 'iters');
T_bitplanes = T;
timing_bitplanes = timing;
iters_bitplanes = iters;

e_bitplanes = kitti_eval(T_bitplanes);

kitti.plot_errors(e_intensity, 'Raw Intensity', ...
  'LineWidth', 1.0, 'Color', 'k', 'LineStyle', '-', 'Marker', '+');

kitti.plot_errors(e_bitplanes, 'Bit-Planes', ...
  'LineWidth', 1.0, 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'Marker', 'o');

e_viso = kitti.load_other('other/viso2');
kitti.plot_errors(e_viso, 'Viso2', ...
  'LineWidth', 1, 'Color', [0.7 0.7 0.7], 'LineStyle', '-.', 'Marker', 's');

legend('Raw Intensity', 'Bit-Planes', 'Viso2', 'location', 'northeast')
