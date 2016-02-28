#include "utils/dataset_loader_thread.h"
#include "utils/bounded_buffer.h"
#include "utils/program_options.h"

#include "bpvo/config.h"
#include "bpvo/vo.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"
#include "bpvo/utils.h"

#include <iostream>
#include <fstream>

#if defined(WITH_PROFILER)
#include <gperftools/profiler.h>
#endif

using namespace bpvo;

int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba.cfg", "config file")
      ("output,o", "output.txt", "trajectory output file")
      ("numframes,n", int(100), "number of frames to process")
      .parse(argc, argv);

  //
  // configure dataset and VO
  //
  auto conf_fn = options.get<std::string>("config");
  auto data_loader = Dataset::Create(conf_fn);
  VisualOdometry vo(data_loader.get(), AlgorithmParameters(conf_fn));

  //
  // load the data into the buffer
  //
  int numframes = options.get<int>("numframes");
  typename DatasetLoaderThread::BufferType buffer(numframes);

  for(int i =0; i < numframes; ++i)
  {
    fprintf(stdout, "loading %d\r", i);
    fflush(stdout);

    auto frame = data_loader->getFrame(i);
    if(!frame) {
      break;
    }

    buffer.push( std::move(frame) );
  }

  fprintf(stdout, "\nRunning\n");

  Trajectory trajectory;
  typename DatasetLoaderThread::BufferType::value_type frame;

#if defined(WITH_PROFILER)
  ProfilerStart("/tmp/prof");
#endif

  double total_time = 0.0;
  int i = 0;
  while(i < numframes)
  {
    if(buffer.pop(&frame))
    {
      if(frame->image().empty())
      {
        Warn("no more data\n");
        break;
      }

      ++i;

      Timer timer;
      auto result = vo.addFrame(frame->image().ptr<uint8_t>(),
                                frame->disparity().ptr<float>());
      double tt = timer.stop().count();
      fprintf(stdout, "Frame %04d\r", i);
      fflush(stdout);
      total_time += tt / 1000.0;

      trajectory.push_back(result.pose);
    }
  }

#if defined(WITH_PROFILER)
  ProfilerStop();
#endif

  Info("Processed %d frames @ %2.fHz\n", numframes, numframes/total_time);

  {
    auto output_fn = options.get<std::string>("output");
    if(!output_fn.empty()) {
      Info("Writing trajectory to %s\n", output_fn.c_str());
      if(!trajectory.writeCameraPath(output_fn)) {
        Warn("failed to write trajectory to %s\n", output_fn.c_str());
      }
    }
  }

  return 0;
}

