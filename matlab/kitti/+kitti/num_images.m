function n = num_images(seq_num, base_dir)
  % function n = num_images(seq_num, base_dir)
  %
  % returns the number of images at a sequence
  %

  if nargin < 2, base_dir = kitti.data_dir; end
  if base_dir(end) == '/', base_dir(end) = []; end

  n=length( dir(sprintf('%s/%02d/image_0/*.png', base_dir, seq_num))) - 1;
end % num_images
