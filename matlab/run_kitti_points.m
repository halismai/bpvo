params = VoMex.DefaultParameters;
params.numPyramidLevels = 5;
params.descriptor = 'Intensity';
%params.descriptor = 'BitPlanes';
params.sigmaPriorToCensusTransform = 0.75;
params.sigmaBitPlanes = 1.75;
params.maxIterations = 100;
params.lossFunction = 'Tukey';
params.minTranslationMagToKeyFrame = 1.5 * 0;
params.minRotationMagToKeyFrame = 2.0;
params.maxFractionOfGoodPointsToKeyFrame = 0.7;
params.goodPointThreshold = 0.8;
params.parameterTolerance = 1e-7;
params.functionTolerance = 1e-6;
params.minSaliency = 0.01;
params.minValidDisparity = 0.001;
params.relaxTolerancesForCoarseLevels = 0;
params.minNumPixelsForNonMaximaSuppresssion = Inf;
params.nonMaxSuppRadius = -1;

scale = 1;
stereo = StereoBm(11, 0, 80/scale);

data_loader = KittiDataLoader('SequenceNumber', 0, 'ScaleFactor', 1);
data_loader.setStereoAlgorithm(stereo);

image_size = size( data_loader.getFrame(1) );
K = data_loader.K;
b = data_loader.getBaseline;
vo = VoMex(K, b, image_size, params);

T = [];

X = [];
C = [];
w = [];
max_depth = 8.0;
pt_weight_thresh = 0.8;

dname = '/tmp/points'

nf = 100;
for i = 1 : nf

  fprintf('Frame %d/%d\n', i, nf);
  [I1, I2, D] = data_loader.getFrame(i - 1);
  result = vo.addFrame(I1, D);

  T_i = invert_pose(result.pose);

  if i > 1
    T(:,:,end+1) = T(:,:,end) * T_i;
  else
    T = T_i;
  end

  if ~isempty( result.pointCloud.X )
    ii = result.pointCloud.w > pt_weight_thresh & result.pointCloud.X(3,:) < max_depth;

    fprintf('\tkeyframe %d points\n', length(ii));

    if isempty(dname)
      X = [X single(transform_points(result.pointCloud.T, result.pointCloud.X(:,ii)))];
      C = [C result.pointCloud.C(:,ii)];
      w = [w result.pointCloud.w(ii)];
    else
      X = single(transform_points(result.pointCloud.T, result.pointCloud.X(:,ii)));
      toply_mex(sprintf('%s/X_%03d.ply', dname, i), X, result.pointCloud.C(:,ii));
    end
  end

end
