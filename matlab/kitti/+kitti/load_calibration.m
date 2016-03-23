function [P, baseline] = load_calibration(seq_num, base_dir)
  % function [P, baseline] = load_calibration(seq_num, base_dir)
  %
  % loads the calibration associated with a sequence number
  %
  % INPUT
  %   seq_num  sequence number
  %   base_dir location of kitti data on disk
  %
  % OUTPUT
  %   P        camera matrix array of 3x4x4
  %     P(:,:,1) left camera grayscale
  %     P(:,:,2) right camera grayscale
  %     P(:,:,3) left camera color
  %     P(:,:,4) right camera color
  %
  %  baseline the stereo rig baseline (grayscale)

  if nargin < 2, base_dir = kitti.data_dir; end
  if base_dir(end) == '/', base_dir(end) = []; end
  fn = sprintf('%s/%02d/calib.txt', base_dir, seq_num);

  fid = fopen(fn,'r');
  if fid < 0, error('kitti.load_calibration:could not open file %s\n', fn); end

  P = zeros(3,4,4);

  % parse the file
  for i=1:4
    tline = fgetl(fid);

    if ~ischar(tline), break; end

    [~, tline] = strtok(tline);
    p = strread(tline);
    P(:,:,i) = reshape(p,4,3)';
  end

  fclose(fid);
  baseline = -P(1,4,2) ./ P(1,1,2);

end
