function [err_cell, len_err, speed_err, errors] = kitti_eval( T )
  % function errors = kitti_eval( T )
  %
  % INPUT
  %     T           is a cell array of poses per dataset (the 11 training datasets)
  %     fig_handle  figure handle to plot (e.g. gcf) if not supplied function
  %                 will not plot errors
  %
  %
  % OUTPUT
  %   error structure with average translation and rotation errors as a function
  %   of path length and vehicle speed


  assert( iscell(T) && length(T) == 11, 'T must be a cell array of size 11x1' );

  errors = struct('t_err', [], 'r_err', [], 'len', [], 'speed', []);
  for i = 1 : length(T)
    fprintf('processing sequence %d/%d\n', i, 11)
    [~, T_gt] = kitti.load_gt(i-1);
    errors = cat_struct_fields(errors, kitti.calc_seq_error(T_gt, T{i}));
  end

  len_err = struct('x', [], 'y_t', [], 'y_r', []);
  [len_err.x, len_err.y_t, len_err.y_r] = kitti.make_error_plot_len(errors);

  speed_err = struct('x', [], 'y_t', [], 'y_r', []);
  [speed_err.x, speed_err.y_t, speed_err.y_r] = kitti.make_error_plot_speed(errors);

  err_cell = kitti.error_struct_to_cell(len_err, speed_err);

end %  kitti_eval

