#include <bpvo/template_data.h>
#include <bpvo/timer.h>
#include <opencv2/core.hpp>

using namespace bpvo;

int main()
{
  AlgorithmParameters p;
  Matrix33 K(Matrix33::Identity());

  TemplateData data(p, K, 1.0f, 0);

  cv::Mat I(480, 640, CV_8UC1);
  cv::Mat D(480, 640, CV_32FC1);

  {
    auto I_ptr = I.ptr<uint8_t>();
    auto D_ptr = D.ptr<float>();
    for(int i = 0; i < I.rows * I.cols; ++i) {
      I_ptr[i] = static_cast<uint8_t>( 255.0 * rand() / (float) RAND_MAX );
      D_ptr[i] = 128.0f * (0.1f + rand() / (float) RAND_MAX);
    }
  }

  data.compute(I, D);

  auto t = TimeCode(100, [&]() { data.compute(I,D); });
  printf("time: %0.2f ms [%d] points\n", t, data.numPoints());

  return 0;
}

