descriptor = 'intensity';

num_iters = load(['../build/tsukuba_' descriptor '_9_iter_num.txt']);
time_ms   = load(['../build/tsukuba_' descriptor '_9_iter_time.txt']);
C         = load(['../build/tsukuba_' descriptor '_9_path.txt'])';

C_gt = data.tsukuba.ground_truth;
c_err_rms = rms( reshape(C_gt(:,1:size(C,2)) - C, [], 1) );

fprintf('Error %g\n', c_err_rms);

plot_with_stats( num_iters );
xlabel('Frame number');
ylabel('# iterations');
title('Bit-Planes');
axis tight;
matlab2tikz(sprintf('figs/%s_iters.tex', ['tsukuba_' descriptor '_9']), ...
  'width', '\fwidth', 'height', '\fheight');

plot_with_stats( time_ms);
xlabel('Frame number');
ylabel('Time (ms)');
title('Bit-Planes');
axis tight;
matlab2tikz(sprintf('figs/%s_time.tex', ['tsukuba_' descriptor '_9']), ...
  'width', '\fwidth', 'height', '\fheight');


