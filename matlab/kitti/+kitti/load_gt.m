function [C,T] = load_gt(seq_num, base_dir)
  %function [C,T] = load_gt(seq_num, base_dir)
  %
  % loads the ground truth associated with sequence
  %
  % INPUT
  %   seq_num     sequence number
  %   base_dir    data location
  %
  % OUTPUT
  %   C           camera center 3xN (wrt to the first frame in the sequence)
  %   T           camera matrices 4x4xN (wrt to first frame in the sequence)
  %
  % If the sequence number does not have a ground truth available, you will see
  % a warning and get empty outputs

  if nargin < 2, base_dir = kitti.gt_dir; end
  if base_dir(end) == '/', base_dir(end) = []; end
  fn = sprintf('%s/%02d.txt', base_dir, seq_num);

  try
    data = load(fn);
    T = zeros(4, 4, size(data,1));
    C = zeros(3, size(data,1));

    for i = 1 : size(data,1)
      T(:,:,i) = [reshape(data(i,:), 4, 3)'; 0 0 0 1];
      C(:,i) = T(1:3,end,i);
    end

  catch
    warning('no ground truth\n');
    C = [];
    T = [];
  end

end
