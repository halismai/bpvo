#include <bpvo/template_data.h>
#include <bpvo/timer.h>
#include <opencv2/core.hpp>

#include <cstdlib>

using namespace bpvo;

int main(int argc, char** argv)
{
  int nrep = argc > 1 ? std::atoi(argv[1]) : 1;

  AlgorithmParameters p;
  Matrix33 K(Matrix33::Identity());

  TemplateData data(p, K, 1.0f, 0);

  int rows = 480 * 1;
  int cols = 640 * 1;

  cv::Mat I(rows, cols, CV_8UC1);
  cv::Mat D(rows, cols, CV_32FC1);

  {
    auto I_ptr = I.ptr<uint8_t>();
    auto D_ptr = D.ptr<float>();
    for(int i = 0; i < I.rows * I.cols; ++i) {
      I_ptr[i] = static_cast<uint8_t>( 255.0 * rand() / (float) RAND_MAX );
      D_ptr[i] = 128.0f * (0.1f + rand() / (float) RAND_MAX);
    }
  }

  data.compute(I, D);
  data.setInputImage(I);

  std::vector<float> residuals;
  std::vector<uint8_t> valid;
  data.computeResiduals(Matrix44::Identity(), residuals, valid);

  printf("error %f\n", std::accumulate(residuals.begin(), residuals.end(), 0.0f));
  printf("valid %f\n", std::count(valid.begin(), valid.end(), 1) / (double) valid.size());
  printf("ERROR NORM:  %f\n",
         Eigen::VectorXf::Map(residuals.data(), residuals.size()).lpNorm<Eigen::Infinity>());

  {
    auto t = TimeCode(nrep, [&]() { data.computeResiduals(Matrix44::Identity(), residuals, valid); });
    printf("computeResiduals time; %f ms\n", t);
  }

  {
    auto t = TimeCode(nrep, [&]() { data.compute(I,D); });
    printf("compute time: %0.2f ms for %d points\n", t, data.numPoints());
  }

  return 0;
}

