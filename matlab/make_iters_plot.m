num_iters = load('../build/tsukuba_intensity_iters.txt');
time_ms = load('../build/tsukuba_intensity_time_ms.txt');
C_est = load('../build/tsukuba_intensity_path.txt')';
kf_inds = load('../build/tsukuba_intensity_kf_inds.txt');

C_gt = data.tsukuba.ground_truth;

figure(1);
plot_with_stats(num_iters);
xlabel('Frame number');
ylabel('# iterations');
title('Raw Intensity')


matlab2tikz('figs/tsukuba_intensity_iters.tex', 'width', '\fwidth', 'height', '\fheight');

figure(2);
plot_with_stats(time_ms)
xlabel('Frame number');
ylabel('Time (ms)');
title('Raw Intensity')

matlab2tikz('figs/tsukuba_intensity_time.tex', 'width', '\fwidth', 'height', '\fheight');

err = 0;
for i = 1 : size(C_est, 2)
  err = err + norm( C_gt(:,i) - C_est(:,i) );
end



