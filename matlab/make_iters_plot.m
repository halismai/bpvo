output_prefix = 'tsukuba_bitplanes';
num_iters     = load(sprintf('../build/%s_iters.txt', output_prefix));
time_ms       = load(sprintf('../build/%s_time_ms.txt', output_prefix));
C_est         = load(sprintf('../build/%s_path.txt', output_prefix))';
kf_inds       = load(sprintf('../build/%s_kf_inds.txt', output_prefix));

C_gt = data.tsukuba.ground_truth;

figure(1);
plot_with_stats(num_iters);
xlabel('Frame number');
ylabel('# iterations');
title('Title')

matlab2tikz(sprintf('figs/%s_iters.tex', output_prefix), 'width', '\fwidth', 'height', '\fheight');

figure(2);
plot_with_stats(time_ms)
xlabel('Frame number');
ylabel('Time (ms)');
title('Title')

matlab2tikz(sprintf('figs/%s_time.tex', output_prefix), 'width', '\fwidth', 'height', '\fheight');

err1 = 0;
for i = 1 : size(C_est, 2)
  err1 = err1 + norm( C_gt(:,i) - C_est(:,i) );
end





