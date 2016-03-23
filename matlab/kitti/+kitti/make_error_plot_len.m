function [x,y_t,y_r] = make_error_plot_len(errors, lengths)
  % function [x,y_t,y_r] = make_error_plot_len(errors, lengths)
  %
  % prepares the data for plotting error as a function of path length
  %
  % INPUT
  %   errors    strucutre from kitti.calc_seq_error.m
  %   lengths   desired path lengths (default uses kitti eval defaults)
  %
  % OUPTPUT
  %     x      values to go on the x-axis (path lengths)
  %     y_t    translation errors (to go on the y-axis) units: percentage
  %     y_r    rotation errors    (to go on the y-axis) units: degrees
  %

  if nargin < 2, lengths = 100:100:800; end

  x = [];
  y_t = [];
  y_r = [];

  for i = 1 : length(lengths)

    len  = lengths(i);

    ind = find( abs((errors.len - len)) < 1.0 );

    if length(ind) > 2
      x(end+1) = len;
      y_t(end+1) = mean( errors.t_err(ind) );
      y_r(end+1) = mean( errors.r_err(ind) );
    end

  end

  y_t = y_t * 100;
  y_r = y_r * 57.3;

end % make_error_plot_len

