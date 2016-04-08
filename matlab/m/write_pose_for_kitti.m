function write_pose_for_kitti(dname, T)

  if ~exist(dname, 'dir')
    mkdir(dname);
  end

  for i = 1 : length(T)
    fn = sprintf('%s/%02d.txt', dname, i-1);
    fprintf('writing to %s\n', fn);
    fid = fopen(fn, 'w');
    T_i = T{i};
    for j = 1 : size(T_i,3)
      P = T_i(:,:,j);
      fprintf(fid, strcat(repmat('%f ', [1 12]), '\n'), ...
        P(1,1), P(1,2), P(1,3), P(1,4), ...
        P(2,1), P(2,2), P(2,3), P(2,4), ...
        P(3,1), P(3,2), P(3,3), P(3,4));
    end
    fclose(fid);
  end

end
