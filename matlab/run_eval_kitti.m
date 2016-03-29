T{11} = [];
timing{11} = [];
iters{11} = [];

params = VoMex.DefaultParameters;
params.numPyramidLevels = 5;
params.descriptor = 'Intensity';
params.sigmaPriorToCensusTransform = 0.75;
params.sigmaBitPlanes = 1.75;
params.maxIterations = 300;
params.lossFunction = 'Tukey';
params.minTranslationMagToKeyFrame = 0.0;
params.minRotationMagToKeyFrame = 2.5;
params.maxFractionOfGoodPointsToKeyFrame = 0.8;
params.parameterTolerance = 1e-7;
params.functionTolerance = 1e-6;
params.minSaliency = 2.5;

stereo = StereoBm(11, 0, 96);

do_show = true;


for seq_num = 0 : 10

  data_loader = KittiDataLoader('SequenceNumber', seq_num, 'ScaleFactor', 1);
  data_loader.setStereoAlgorithm(stereo);

  if do_show
    [C_gt, T_gt] = kitti.load_gt(seq_num);
  end

  I = data_loader.getFrame(1);
  image_size = [size(I,1) size(I,2)];

  K = data_loader.K();
  b = data_loader.getBaseline();
  vo = VoMex(K, b, image_size, params);

  s_i = seq_num + 1;
  [T{s_i}, timing{s_i}, iters{s_i}] = eval_kitti_seq(data_loader, vo, T_gt, do_show);

end
