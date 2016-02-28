#include "utils/program_options.h"
#include "utils/dataset_loader_thread.h"

#include "bpvo/vo.h"
#include "bpvo/timer.h"
#include "bpvo/debug.h"

using namespace bpvo;

int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba.cfg", "config file")
      ("numframes,n", int(500), "number of frames to process")
      .parse(argc, argv);

  std::string conf_fn = options.get<std::string>("config");

  auto dataset = Dataset::Create(conf_fn);
  auto params = AlgorithmParameters(conf_fn);
  params.minTranslationMagToKeyFrame = 0.0;

  VisualOdometry vo(dataset.get(), params);

  std::vector<UniquePointer<DatasetFrame>> data;

  data.push_back( dataset->getFrame(1) );
  data.push_back( dataset->getFrame(2) );
  data.push_back( dataset->getFrame(3) );
  data.push_back( dataset->getFrame(4) );
  data.push_back( dataset->getFrame(5) );

  int numframes = options.get<int>("numframes");
  int numiters = 0;
  double total_time = 0.0;
  for(int i = 0; i < numframes; ++i)
  {
    const auto& frame = data[ i % data.size() ];

    Timer timer;
    auto result = vo.addFrame(frame->image().ptr<uint8_t>(), frame->disparity().ptr<float>());
    auto t = timer.stop().count();
    total_time += t / 1000.0;
    for(auto o : result.optimizerStatistics)
      numiters += o.numIterations;

    int num_iters = result.optimizerStatistics.front().numIterations;
    fprintf(stdout, "Frame %03d/%d @ %0.2f Hz %04d\r", i, numframes,
            i / total_time, num_iters);
    fflush(stdout);
  }

  fprintf(stdout, "\n");
  Info("done @ %0.2f Hz %f iters\n", numframes / total_time, numiters / (float) numframes);

  return 0;
}



