function [T, timing, iters] = eval_kitti_seq(data_loader, vo, T_gt, do_show)

  if nargin < 4, do_show = false; end
  if nargin < 3, T_gt = []; end

  f_inds = data_loader.frameStart() : data_loader.numFrames();
  nf = length(f_inds);

  T  = repmat(eye(4), [1 1 nf]);
  timing = zeros(1, nf);
  iters = zeros(1, nf);

  if do_show
    clf; subplot(1,2,2);

    h_gt = plot33([0;0;0],'-'); hold on;
    set(h_gt, 'color', 'g', 'linewidth', 2);

    h_alg = plot33([0;0;0],'r-');
    set(h_alg, 'color', [0.3 0.3 0.3], 'linewidth', 2);

    I1=data_loader.getFrame(1);
    set(gcf, 'name', sprintf('resolution %dx%d', size(I1,2), size(I1,1)));

    cmap = parula;

    subplot(1,2,1);
    M = repmat([I1; I1], [1 1 3]);
    h_img = imshow(M); hold on;
  end


  for i = 1 : nf

    f_i = f_inds(i);
    [I1, I2, D] = data_loader.getFrame(f_i);

    t_s = tic;
    result = vo.addFrame(I1, D);
    timing(i) = toc(t_s) * 1000;
    iters(i) = result.optimizerStatistics(1).numIterations;

    result
    result.optimizerStatistics(1)

    cprintf.green('frame %d/%d %0.2f ms\n', i, nf, timing(i));
    %assert( is_kf );

    T_i = result.pose;

    if i == 1
      T(:,:,i) = inv(T_i);
    else
      R_i = T_i(1:3,1:3).';
      T_inv = [R_i -R_i*T_i(1:3,end); 0 0 0 1];
      T(:,:,i) = T(:,:,i-1) * T_inv;
    end


    if do_show
      update_pose_plot(h_alg, T(:,:,1:i));
      update_pose_plot(h_gt, T_gt(:,:,1:i));
      subplot(1,2,2);
      axis tight equal;

      M = [repmat(I1, [1 1 3]); grs2rgb(D, cmap)];
      subplot(1, 2, 1); set(h_img, 'cdata', M);
      drawnow;
    end

  end

end


function M = make_image(I, D, cmap)


end
