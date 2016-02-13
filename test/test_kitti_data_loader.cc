#include "utils/data_loader.h"
#include "utils/viz.h"
#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  auto data_loader = DataLoader::FromConfig("../conf/kitti.cfg");
  typename DataLoaderThread::BufferType image_buffer(16);

  DataLoaderThread data_loader_thread(std::move(data_loader), image_buffer);
  typename DataLoaderThread::BufferType::value_type frame;

  int i = 0;
  while( i < 1000 ) {
    if(image_buffer.pop(&frame)) {
      cv::imshow("image", frame->image());
      cv::imshow("disparity", colorizeDisparity(frame->disparity()));
      int k = 0xff & cv::waitKey(5);
      if(k == ' ') k = 0xff & cv::waitKey(0);
      if(k == 'q' || k == 27)
        break;
      ++i;
    }
  }

  return 0;
}
