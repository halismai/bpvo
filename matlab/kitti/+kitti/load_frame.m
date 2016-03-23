function [I1,I2] = load_frame(n, seq_num, base_dir)
  % function [I1,I2] = load_frame(n, seq_num, base_dir)

  if nargin < 3, base_dir = kitti.data_dir; end
  if base_dir(end) == '/', base_dir(end) = []; end


  fn1 = sprintf('%s/%02d/image_0/%06d.png', base_dir, seq_num, n);
  fn2 = sprintf('%s/%02d/image_1/%06d.png', base_dir, seq_num, n);

  I1 = imread(fn1);
  I2 = imread(fn2);

end
