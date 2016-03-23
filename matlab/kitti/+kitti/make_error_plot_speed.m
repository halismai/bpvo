function [x, yt, yr] = make_error_plot_speed(errors)
  % function [x, yt, yr] = make_error_plot_speed(errors)
  %
  % prepares data for plotting error as a function of vechile speed
  % SEE ALSO kitti.make_error_plot_len
  %

  x  = [];
  yt = [];
  yr = [];


  for speed = 2 : 2 : 25

    ind = find( abs(errors.speed - speed) < 1.0 );
    if length(ind)> 2
      x(end+1) = speed * 3.6;
      yt(end+1) = 100 * mean( errors.t_err(ind) );
      yr(end+1) = rad2deg( mean( errors.r_err(ind) ) );
    end

  end

end
