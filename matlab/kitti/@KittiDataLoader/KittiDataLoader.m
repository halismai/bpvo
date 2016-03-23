classdef KittiDataLoader <  DataLoader
  %classdef KittiDataLoader <  DataLoader

  properties (SetAccess = protected)
    K_full_res_;
  end

  methods
    function obj = KittiDataLoader(varargin)
      obj = obj@DataLoader( varargin{:} );
      obj.set_inputs;
      obj.input_parser.parse( varargin{:} );

      s = obj.input_parser.Results.ScaleFactor;

      obj.frames_base_dir = obj.getOption('BaseDirectory');
      if obj.frames_base_dir(end) == '/', obj.frames_base_dir(end) = []; end
      obj.frames_base_dir = strcat(obj.frames_base_dir, '/sequences');

      obj.poses_base_dir = obj.getOption('BaseDirectory');
      if obj.poses_base_dir(end) == '/', obj.poses_base_dir(end)=[]; end
      obj.poses_base_dir = strcat(obj.poses_base_dir, '/poses');

      seq_num = obj.getOption('SequenceNumber');

      [P, obj.baseline] = kitti.load_calibration(seq_num, obj.frames_base_dir);
      obj.K_full_res_ = P(1:3,1:3,1);
      obj.K = P(1:3,1:3,1) / s; obj.K(end) = 1;

      obj.frame_start = 0;
      obj.frame_cntr = obj.frame_start + 1;
      obj.num_frames = kitti.num_images(seq_num, obj.frames_base_dir);

      if obj.getOption('SequenceNumber') <= 10
        [obj.C_gt, obj.T_gt] = kitti.load_gt(seq_num, obj.poses_base_dir);
      end

    end
  end % methods

  methods

    function [I1,I2,D] = getNextFrame(obj)
      if obj.frame_cntr > obj.num_frames
        warning('no more frames');
        I1 = []; I2 = []; D = [];
      else
        [I1,I2,D] = obj.getFrame( obj.frame_cntr );
        obj.frame_cntr = obj.frame_cntr + 1;
      end
    end

    function [I1,I2,D] = getFrame(obj, frame_num)

      f_i = frame_num + 1;
      if f_i < length( obj.preloaded_I1  )
        I1 = obj.preloaded_I1{f_i};
        I2 = obj.preloaded_I2{f_i};
        D  = obj.preloaded_D{f_i};
      else
        [I1,I2] = kitti.load_frame( frame_num, ...
          obj.getOption('SequenceNumber'), obj.frames_base_dir);

        s = obj.getOption('ScaleFactor');
        if s ~= 1
          I1 = imresize( I1,  1/s ); %, 'bilinear' );
          I2 = imresize( I2, 1/s ); %, 'bilinear' );
        end

        if nargout > 2
          D = obj.computeDisparity(I1,I2);
        end
      end
    end

    function [I1,I2] = getFrameFullRes(obj, frame_num)
      [I1,I2] = kitti.load_frame(frame_num, ...
        obj.getOption('SequenceNumber'), obj.frames_base_dir);
    end

    function K = K_full(obj)
      K = obj.K_full_res_;
    end

    function preLoadImages(obj, num_preload)

      fprintf('preloading %d frames\n', num_preload)
      obj.preloaded_I1{ num_preload+1 } = []; % KITTI starts from 0
      obj.preloaded_I2{ num_preload+1 } = [];
      obj.preloaded_D{ num_preload+1 } = [];

      h = waitbar( 0 );

      tic
      for i = obj.frame_start : num_preload

        [I1,I2,D] = obj.get_frame_preload(i);

        obj.preloaded_I1{i+1} = I1;
        obj.preloaded_I2{i+1} = I2;
        obj.preloaded_D{i+1}  = single( D );

        if mod(i,10)
          waitbar( i / num_preload );
        end

      end
      fprintf('done in %f s\n', toc);
      close(h);

    end

  end % methods


  methods (Access = private)
    function set_inputs(obj)
      base_dir = kitti.data_dir;
      obj.input_parser.addOptional('SequenceNumber', 0, @isscalar);
      obj.input_parser.addOptional('BaseDirectory', base_dir, @isstr);
    end

    function [I1,I2,D] = get_frame_preload(obj, frame_num)
      [I1,I2] = kitti.load_frame( frame_num, ...
        obj.getOption('SequenceNumber'), obj.frames_base_dir);

      s = obj.getOption('ScaleFactor');
      if s ~= 1
        I1 = imresize( I1,  1/s );
        I2 = imresize( I2, 1/s );
      end

      D = obj.computeDisparity(I1,I2);
    end

  end % methods

  properties (SetAccess = protected)
    frames_base_dir;
    poses_base_dir;
  end

  properties (SetAccess = private)
    preloaded_I1;
    preloaded_I2;
    preloaded_D;
  end

end % classdef
