%files = glob('~/code/bpvo/build/kitti_intensity_*_trajectory.txt');
files = glob('~/code/bpvo/results/kitti_intensity_*_poses.txt');

T = load_kitti_pose_from_txt(files);

%T_bp = load_kitti_pose_from_txt(...
%  glob('~/code/bpvo/build/kitti_bitplanes_*trajectory.txt'));

err_intensity = kitti_eval(T);

if exist('T_bp', 'var')
  err_bp = kitti_eval(T_bp);
end

e_viso = kitti.load_other('other/viso2');

clf;

kitti.plot_errors(err_intensity, 'Raw Intensity', ...
  'LineWidth', 1.0, 'Color', 'k', 'LineStyle', '-', 'Marker', '+');

if exist('T_bp', 'var')
  kitti.plot_errors(err_bp, 'Bit-Planes', ...
    'LineWidth', 1, 'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'Marker', 'o');
end

kitti.plot_errors(e_viso, 'Viso2', ...
  'LineWidth', 1, 'Color', [0.7 0.7 0.7], 'LineStyle', '-.', 'Marker', 's');


