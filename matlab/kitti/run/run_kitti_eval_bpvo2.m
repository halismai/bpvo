e_bpvo = load_bpvo_results('/home/halismai/code/bpvo/build/kitti_eval');
e_viso = kitti.load_other('other/viso2');

for i = 0 : 10
  files{i+1} = sprintf('/home/halismai/code/bpvo/results/intensity/%02d.txt', i);
end
T = load_kitti_pose_from_txt(files);

err = kitti_eval(T);

kitti.plot_errors(e_bpvo, 'Raw Intensity', ...
  'LineWidth', 1.0, 'Color', 'k', 'LineStyle', '-', 'Marker', '+');

kitti.plot_errors(e_viso, 'Viso2', ...
  'LineWidth', 1, 'Color', [0.7 0.7 0.7], 'LineStyle', '-.', 'Marker', 's');


