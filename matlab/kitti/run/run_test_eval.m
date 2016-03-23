load ~/code/mf/results/phovo_dspace_kitti_all_1x1.mat
clearvars -except T

err    = kitti_eval( T );
e_viso = kitti.load_other('other/viso2');

%kitti.plot_errors(len_err, speed_err);


clf;
spec1 = {'LineWidth', 2, 'Color', 'k', 'LineStyle', '-', 'Marker', '+'};
spec2 = {'LineWidth', 2, 'Color', 'k', 'LineStyle', '--', 'Marker', 's'};
kitti.plot_errors(err, '3d_1x1', spec1{:});
kitti.plot_errors(e_viso, 'viso2', spec2{:});

