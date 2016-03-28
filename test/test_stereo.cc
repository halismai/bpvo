#include "utils/stereo_algorithm.h"
#include "utils/program_options.h"
#include "utils/viz.h"
#include "utils/dataset.h"

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;
int main(int argc, char** argv)
{
  ProgramOptions options;
  options("config,c", "../conf/kitti_stereo.cfg", "config file")
      .parse(argc, argv);

  const auto conf_fn = options.get<std::string>("config");
  auto dataset = Dataset::Create(conf_fn);

  cv::Mat dmap;
  UniquePointer<DatasetFrame> frame;

  int f_i = 0;
  while( nullptr != (frame = dataset->getFrame(f_i++)))
  {
    overlayDisparity(frame->image(), frame->disparity(), dmap, 0.5, 0.5, 128);

    cv::imshow("dmap", dmap);
    int k = 0xff & cv::waitKey(2);
    if(k == ' ') k = 0xff & cv::waitKey(0);
    if(k == 'q') break;
  }

  return 0;
}
