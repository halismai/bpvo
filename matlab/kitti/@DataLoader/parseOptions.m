function parseOptions(obj, varargin)

  p = inputParser;

  p.addRequired('Name', @isstr);
  p.addOptional('ScaleFactor', 1, @isscalar); % image scaling (optional)

  % sequence number for KITTI
  p.addOptional('SequenceNumber', 0, @isscalar);

  % tsukuba illumination
  p.addOptional('TsukubaIllumination', 'fluorescent', @isstr);

  % make a disparity map out of ground truth *depth*
  % if false, the discretized disparity maps will be loaded
  p.addOptional('TsukubaUseDepth', false, @islogical);

  p.parse(varargin{:});

  obj.name = p.Results.Name;
  obj.scale_factor = p.Results.ScaleFactor;

  obj.p_results = p.Results;
  obj.loadData(p.Results);
end
