#include "utils/dataset2.h"
#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  auto dataset = utils::Dataset::Create("../conf/tsukuba_data.cfg");
  UniquePointer<utils::DatasetFrame> frame;

  int f_i = 0, k = 0;
  while(k != 'q')
  {
    if(nullptr == (frame = dataset->getFrame(f_i++)))
      break;

    /*
    cv::imshow("image", frame->image());
    k = 0xff & cv::waitKey(5);
    */
    if(f_i > 2) break;
  }

}



