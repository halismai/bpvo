classdef FrieburgRGBD < DataLoader

  methods

    function obj = FrieburgRGBD( varargin )
      obj = obj@DataLoader( varargin{:} );
      obj.set_inputs;
      obj.input_parser.parse( varargin{:} );

      obj.initData();

    end

  end


  methods (Access = protected)
    function initData(obj)
      obj.frames_base_dir_ = obj.getOption('BaseDirectory');
      if obj.frames_base_dir_(end) == '/', obj.frames_base_dir_(end) = []; end

      obj.load_data();

    end


    function load_data(obj)
      switch obj.getOption('Name')
        case 'fr1xyz'
          dname = 'rgbd_dataset_freiburg1_xyz';
        otherwise
          error('unknown dataset name %s\n', obj.getOption('Name'));
      end % switch

      dname = sprintf('%s/%s', obj.frames_base_dir_, dname);
      obj.image_files_ = glob(sprintf('%s/rgb/*.png', dname));
      obj.depth_files_ = glob(sprintf('%s/depth/*.png',dname));
      assert( length(obj.image_files_) == length(obj.depth_files_) )


      % get the ground truth

    end
  end

  methods (Access = private)
    function set_inputs( obj )
      obj.input_parser.addOptional('Name', 'fr1xyz', @isstr);
      obj.input_parser.addOptional('BaseDirectory', ...
        '/media/external_lpv/rgbd/', @isstr);
    end
  end

  properties (SetAccess = protected)
    frames_base_dir_;
    image_files_;
    depth_files_;
  end

end % FrieburgRGBD
