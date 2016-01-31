#include "test/data_loader.h"
#include "bpvo/timer.h"
#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  TsukubaDataLoader data_loader;
  UniquePointer<ImageFrame> frame;

  Timer timer;
  int f_i = 1, k = 0;
  while( nullptr != (frame = data_loader.getFrame(f_i++)) && k != 'q')
  {
    cv::imshow("image", frame->image());
    cv::imshow("disparity", colorizeDisparity(frame->disparity()));
    k = cv::waitKey(5) & 0xff;
  }

  auto t = timer.stop().count() / 1000.0;
  printf("read %d frames %0.2f Hz\n", f_i, f_i / t);
}
