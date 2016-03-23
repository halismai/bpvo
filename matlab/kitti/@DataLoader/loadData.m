function loadData(obj, p_results)

  name = lower( obj.name );
  switch( name )
    case 'tsukuba'

      % Tsukuba dataset
      obj.K = [615 0 320; 0 615 240; 0 0 1];
      obj.baseline = 10/100;
      obj.num_frames = 1800;
      obj.frame_start = 1;
      [obj.C_gt, obj.T_gt] = load_tsukuba_pose;
      obj.has_gt_pose = true;
      obj.has_gt_dmap = true;

    case 'kitti'

      num = p_results.SequenceNumber;
      [P, obj.baseline] = kitti.load_calibration( num );
      obj.K = P(1:3,1:3,1); % left camera K
      obj.num_frames = kitti.num_images( num );
      obj.frame_start = 0;

      if num <= 10
        obj.has_gt_pose = true;
        obj.has_gt_dmap = false;
        [obj.C_gt, obj.T_gt] = kitti.load_gt( num );
      else
        obj.has_gt_pose = false;
        obj.has_gt_dmap = false;
      end

    case 'wean'
      % this is the 3.8mm lens
      obj.K = [487.1091 0 320.7883; 0 487.1091 245.8451; 0 0 1];
      obj.baseline = 0.12;
      obj.frame_start = -1; % globs all
    otherwise
      error(['unknown dataset ' obj.name]);
  end


end
