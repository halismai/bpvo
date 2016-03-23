function errors = make_avg_errors(poses)

  errors = struct('t_err', [], 'r_err', [], 'len', [], 'speed', []);

  for i = 1 : length(poses)
    T = poses{i};
    T_gt = kitti.load_ground_truth(i-1);
    err = kitti.calc_sequence_errors(T_gt, T);

    errors.t_err = [errors.t_err err.t_err];
    errors.r_err = [errors.r_err err.r_err];
    errors.len   = [errors.len err.len];
    errors.speed = [errors.speed err.speed];
  end

end
