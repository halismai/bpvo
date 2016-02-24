#include <utils/dataset.h>
#include <utils/dataset_loader_thread.h>
#include <utils/program_options.h>
#include <utils/viz.h>

#include <bpvo/timer.h>
#include <bpvo/config_file.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>

using namespace bpvo;

int main(int argc, char** argv)
{
  bpvo::ProgramOptions options(argv[0]);
  options("config,c", "../conf/tsukuba_stereo.cfg", "config file")
      ("numframes,n", int(256), "number of frames to process")
      .parse(argc, argv);

  int numframes = options.get<int>("numframes");
  auto conf_fn  = options.get<std::string>("config");
  typename DatasetLoaderThread::BufferType buffer(32);
  DatasetLoaderThread data_loader(Dataset::Create(conf_fn), buffer);

  UniquePointer<DatasetFrame> frame;

  int min_d = 0, num_d = 128;
  {
    ConfigFile cf(conf_fn);
    min_d = cf.get<int>("minDisparity", 0);
    num_d = cf.get<int>("numberOfDisparities", 128);
  }

  cv::Mat display_image;
  Timer timer;
  int k = 0, f_i = 0;
  while( 'q' != k && 27 != k && f_i < numframes )
  {
    if(buffer.pop(&frame, 5))
    {
      if(!frame)
        break; // nullptr frame is the end of the dataset

      colorizeDisparity(frame->disparity(), display_image, min_d, num_d);
      overlayDisparity(frame->image(), frame->disparity(), display_image,
                       0.5, min_d, num_d);
      cv::imshow("disparity", display_image);
      cv::imshow("image", frame->image());
      fprintf(stdout, "Frame %06d\r", f_i); fflush(stdout);
      f_i += 1;
    }

    k = 0xff & cv::waitKey(1);
    if(k == ' ') k = 0xff & cv::waitKey(0);
  }

  auto tt = timer.stop().count() / 1000.0;
  printf("\nProcessed %d frames @ %0.2f Hz\n", f_i, f_i / tt);

  return 0;
}
