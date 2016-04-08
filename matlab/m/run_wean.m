stereo = StereoBm(9, 0, 80);
data_loader = BumblebeeDataLoader();
data_loader.setStereoAlgorithm( stereo );
f_i = 800 : 1600;

params = VoMex.DefaultParameters;
params.numPyramidLevels = 4;
params.descriptor = 'Intensity';
%params.descriptor = 'BitPlanes';
params.sigmaPriorToCensusTransform = 0.75;
params.sigmaBitPlanes = 1.75;
params.maxIterations = 100;
params.lossFunction = 'Tukey';
params.minTranslationMagToKeyFrame = 0.1;
params.minRotationMagToKeyFrame = 2.0;
params.maxFractionOfGoodPointsToKeyFrame = 0.7;
params.goodPointThreshold = 0.8;
params.parameterTolerance = 1e-7;
params.functionTolerance = 1e-6;
params.minSaliency = 0.01;
params.minValidDisparity = 0.001;
params.relaxTolerancesForCoarseLevels = 0;
%params.minNumPixelsForNonMaximaSuppresssion = Inf;
%params.nonMaxSuppRadius = -1;

vo = VoMex(data_loader.K, data_loader.getBaseline, ...
  size(data_loader.getFrame(1)), params);

options = struct('points_prefix', '/tmp/wean/', 'good_points_threshold', 0.5,  ...
  'max_depth', 5.0, 'do_show', true);

[T, num_iters, timing] = run_bpvo(vo, data_loader, f_i, options);
