function T = load_kitti_pose_from_txt(files)

  T = cell(1, length(files));

  for i = 1 : length(files)
    data = load(files{i});
    T_i = repmat(eye(4), [1 1 size(data,1)]);
    for j = 1 : size(data, 1)
      if length(data(j,:)) == 3*4
        T_i(1:3,:,j) = reshape(data(j,:), [4 3])';
      else
        T_i(:,:,j) = reshape(data(j,:), [4 4])';
      end
    end
    T{i} = T_i;
  end
end
