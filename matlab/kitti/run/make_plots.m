e_bpvo = kitti.load_other('160329/plot_error/');
e_viso = kitti.load_other('other/viso2');

kitti.plot_errors(e_bpvo, 'Raw Intensity', 'LineWidth', 1.0, 'Color', 'k', ...
  'LineStyle', '-', 'Marker', '+');
kitti.plot_errors(e_viso, 'Viso2', 'LineWidth', 1.0, 'Color', '[0.7 0.7 0.7]', ...
  'LineStyle', '-', 'Marker', 's');

