#include <cstdio>

#include <bpvo/vo_output_reader.h>
#include <bpvo/vo_output.h>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

int main()
{
  bpvo::VoOutputReader vo_output_reader(".", "vo_%05d.voout", 1);

  cv::namedWindow("image");
  for(int i = 0; ; ++i)
  {
    auto frame = vo_output_reader[i];

    if(!frame)
      break;

    cv::imshow("image", frame->image());
    int k = 0xff & cv::waitKey(5);
    if(k == 'q') break;
  }
}


