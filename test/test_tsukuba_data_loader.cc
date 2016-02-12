#include "utils/data_loader.h"
#include "utils/viz.h"
#include "bpvo/utils.h"
#include "bpvo/timer.h"
#include <opencv2/highgui/highgui.hpp>

#include <thread>

using namespace bpvo;

typename DataLoaderThread::BufferType gBuffer(32);

void ProcessData()
{
  SharedPointer<ImageFrame> frame;

  Timer timer;
  int k = 0;
  int f_i = 1;
  while(  k != 'q')
  {
    if(gBuffer.pop(&frame)) {
      cv::imshow("image", frame->image());
      cv::imshow("disparity", colorizeDisparity(frame->disparity()));
      ++f_i;
    }

    k = cv::waitKey(5) & 0xff;
  }

  auto t = timer.stop().count() / 1000.0;
  printf("read %d frames %0.2f Hz\n", f_i, f_i / t);

  while(gBuffer.pop(&frame))
    ; // empty the buffer
}

int main()
{
  DataLoaderThread data_loader(UniquePointer<DataLoader>(new TsukubaDataLoader), gBuffer);
  std::thread t(ProcessData);
  t.join();
  data_loader.stop();

  return 0;
}

