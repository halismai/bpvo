function errors = calc_seq_error(T_gt, T_est, lengths, step_size)
  % function errors = calc_seq_error(T_gt, T_est)
  %
  % calculates a sequence error
  %
  % INPUT
  %   T_gt      ground truth pose
  %   T_est     estimated pose (your results)
  %   lengths   segment lengths [optional] will use deafult kitti evaluation
  %             segment lengths
  %   step_size optional [default = 10]
  %
  % OUTPUT
  %  errors structure with errors ready for plotting

  if nargin < 4, step_size = 10; end  % every second
  if nargin < 3, lengths = 100:100:800; end

  dist = kitti.trajectory_distance(T_gt);
  errors = struct('first_frame', [], 'r_err', [], 't_err', [], 'len', [], 'speed', []);

  for first_frame = 1 : step_size : size(T_gt, 3)

    for i = 1 : length(lengths)
      len = lengths(i);
      last_frame = last_frame_from_seq_len(dist, first_frame, len);

      if isempty(last_frame)
        continue; %, if sequence not long enough
      end

      T_delta_gt  = T_gt(:,:,first_frame)  \ T_gt(:,:,last_frame);
      T_delta_est = T_est(:,:,first_frame) \ T_est(:,:,last_frame);
      pose_error = T_delta_est \ T_delta_gt;

      r_err = rotation_error(pose_error);
      t_err = translation_error(pose_error);

      num_frames = last_frame - first_frame; % we start from 1
      speed      = len / (0.1*num_frames);

      errors.first_frame(end+1) = first_frame;
      errors.r_err(end+1) = r_err / len;
      errors.t_err(end+1) = t_err / len;
      errors.len(end+1) = len;
      errors.speed(end+1) = speed;
    end

  end

end % calc_seq_error

function err = rotation_error(T)
  %function err = rotation_error(T)
  %
  % computes the rotation error from a 'delta' pose
  s = sum( diag(T(1:3,1:3)) );
  d = 0.5 * ( s - 1.0 );
  err = acos( max( min(d, 1.0), -1.0) );
end % rotation_error

function err = translation_error(T)
  %function err = translation_error(T)
  %
  % computes the translation error from a 'delta' pose
  err = norm(T(1:3,end));
end % translation_error

function f_num = last_frame_from_seq_len(dist, first_frame, len)
  %function f_num = last_frame_from_seq_len(dist, first_frame, len)
  f_num = [];
  ind = find( dist > dist(first_frame)+len );
  if ~isempty(ind)
    f_num = ind(1);
  end
end % last_frame_from_seq_len


