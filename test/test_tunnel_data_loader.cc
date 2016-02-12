#include "utils/tunnel_data_loader.h"
#include "utils/viz.h"

#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace bpvo;

int main()
{
  auto data_loader = DataLoader::FromConfig("../conf/tunnel.cfg");
  typename DataLoaderThread::BufferType image_buffer(16);

  std::cout << data_loader->calibration() << std::endl;
  std::cout << data_loader->imageSize() << std::endl;

  DataLoaderThread data_loader_thread(std::move(data_loader), image_buffer);
  SharedPointer<ImageFrame> frame;

  int i = 0;
  while( true ) {
    if(image_buffer.pop(&frame)) {
      cv::imshow("image", overlayDisparity(frame.get(), 0.8));
      printf("frame %d\n", i);
      int k = 0xff & cv::waitKey(5);
      if(k == ' ') k = 0xff & cv::waitKey(0);
      if(k == 'q' || k == 27)
        break;
      ++i;
    }
  }

  return 0;
}
