function [T, num_iters, timing] = run_bpvo(vo, data_loader, f_inds, options)

  if nargin < 4, options = set_default_options; end


  if options.do_show
    clf;
    subplot(1, 2, 1);
    I1  = data_loader.getFrame(f_inds(1));
    h_img = imshow( repmat([I1; I1],[1 1 3]) );
    cmap = parula;
    subplot(1, 2, 2);
    h_cpath = plot33([0; 0; 0], 'k-');
    set(h_cpath, 'color', 'k', 'linewidth', 2);

    set(gcf, 'name', sprintf('resolution %dx%d', size(I1,2), size(I1,1)));
  end

  T = repmat(eye(4), [1 1 length(f_inds)]);
  timing = zeros(1, length(f_inds));
  num_iters = zeros(1, length(f_inds));


  total_time = 0.0;
  for i = 1 : length(f_inds)
    [I1, I2, D] = data_loader.getFrame( f_inds(i) );

    t = tic;
    result = vo.addFrame( I1, D );
    t = toc(t) * 1000.0;

    total_time = total_time + ( t / 1000.0 );

    result
    result.optimizerStatistics(1)

    cprintf.green('Frame %d/%d time %0.2f ms [%0.2f Hz]\n', ...
      i, length(f_inds), t, i / total_time);

    timing(i) = t;
    num_iters(i) = result.optimizerStatistics(1).numIterations;

    T_i = invert_pose(result.pose);
    if i > 1
      T(:,:,i) = T(:,:,i-1) * T_i;
    else
      T(:,:,i) = T_i;
    end

    if options.do_show
      update_pose_plot(h_cpath, T(:,:,1:i));
      subplot(1, 2, 2); axis tight equal; view([0 -1 0]);

      M = [repmat(I1, [1 1 3]); grs2rgb(D, cmap)];
      subplot(1, 2, 1); set(h_img, 'cdata', M);
      drawnow;
    end

    if ~isempty( options.points_prefix )
      if ~isempty( result.pointCloud.X )
        ii = result.pointCloud.w > options.good_points_threshold & ...
          result.pointCloud.X(3,:) < options.max_depth;

        fn = sprintf('%s/%05d', options.points_prefix, i);
        X = transform_points( result.pointCloud.T, result.pointCloud.X(:,ii) );

        x = normHomog( data_loader.K * result.pointCloud.X(:,ii) );
        C = data_loader.getColor( f_inds(i), round(x) );

        toply_mex(fn, single(X), C);
      end
    end

  end


end

function o = set_default_options(o)
  o = check_struct_field(o, 'points_prefix', []);
  o = check_struct_field(o, 'goot_points_threshold', 0.75);
  o = check_struct_field(o, 'max_depth', 8.0);
  o = check_struct_field(o, 'do_show', true);
end
