classdef DataLoader < handle

  methods

    function obj = DataLoader( varargin )
      obj.initOptions;
    end % DataLoader


    function obj = setStereoAlgorithm(obj, stereo_algorithm)
      %function obj = setStereoAlgorithm(obj, stereo_algorithm)
      obj.stereo_algorithm = stereo_algorithm;
    end

    function K = getK(obj)
      %function K = getK(obj)
      K = obj.K;
    end

    function b = getBaseline(obj)
      %function b = getBaseline(obj)
      b = obj.baseline;
    end

    function [K, b] = getCalibration(obj)
      %function [K, b] = getCalibration(obj)
      K = obj.K;
      b = obj.baseline;
    end

    function [C,T] = getGroundTruthPose(obj)
      %function [C,T] = getGroundTruthPose(obj)
      C = obj.C_gt;
      T = obj.T_gt;
    end

    function fs = frameStart(obj)
      %function fs = frameStart(obj)
      fs = obj.frame_start;
    end

    function nf = numFrames(obj)
      %function nf = numFrames(obj)
      nf = obj.num_frames;
    end

    function val = getOption(obj, name)
      %val = getfield( obj.input_parser.Results, name );
      val = obj.input_parser.Results.(name);
    end
  end % methods

  methods (Abstract)
    %methods (Abstract)
    [I1,I2,D] = getNextFrame(obj);
    [I1,I2,D] = getFrame(obj, frame_num);
  end


  methods (Access = private)

    function initOptions(obj)
      % function initOptions(obj)
      %
      % Initializes options common to all datasets

      obj.input_parser = inputParser;
      obj.input_parser.CaseSensitive    = false;
      obj.input_parser.KeepUnmatched    = true;
      obj.input_parser.PartialMatching  = true;

      % scaling factor to reduce image size.
      % For example, ScaleFactor 2 means cut the image by half
      obj.input_parser.addOptional('ScaleFactor', 1, @isscalar);

      % number of images to preload in memory
      obj.input_parser.addOptional('NumPreLoad', 0, @isscalar);
    end

  end


  methods
    function D = computeDisparity(obj, I1, I2)
       %function D = computeDisparity(obj, I1, I2)
      D = obj.stereo_algorithm.computeDisparity(I1, I2);
    end
  end % methods

  properties ( SetAccess = protected )
    input_parser;       % input parser with options

    stereo_algorithm;   % pre-configured stereo algorithm to use
    frame_cntr;         % current frame number
  end % properties

  properties ( SetAccess = protected )
    num_frames;  % number of frames for the data set
    frame_start; % first frame number, 0 or 1
    C_gt;        % ground truth camera center
    T_gt;        % ground truth pose
    K;           % calibration, the K matrix (3x3)
    baseline;    % stereo baseline

    num_preload; % number of images to preload
  end % properties

end % classdef

