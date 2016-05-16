#include <bpvo/types.h>
#include <bpvo/dense_descriptor.h>
#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <bpvo/latch_descriptor.h>
#include <bpvo/central_difference_descriptor.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

static void Run(const cv::Mat& I, AlgorithmParameters p, DescriptorType t)
{
  p.descriptor = t;
  auto desc = DenseDescriptor::Create(p, t);
  desc->compute(I);

  auto tt = TimeCode(100, [&]() { desc->compute(I); });
  printf("Descriptor: %s Time %f\n", ToString(t).c_str(), tt);

  delete desc;
}

static void WriteChannel(std::string fn, const cv::Mat& C_,
                         bool with_laplacian = false)
{
  FILE* fp = fopen(fn.c_str(), "w");
  THROW_ERROR_IF( fp == NULL, "failed to open file" );

  cv::Mat C = C_;
  if(with_laplacian)
    cv::Laplacian(C_, C, -1);

  for(int y = 0; y < C.rows; ++y) {
    for(int x = 0; x <  C.cols; ++x) {
      fprintf(fp, "%f ", C.at<float>(y,x));
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  THROW_ERROR_IF(I.empty(), "failed to read image");

  AlgorithmParameters p;
  p.centralDifferenceRadius = 1;

  /*
  Run(I, p, DescriptorType::kIntensity);
  Run(I, p, DescriptorType::kIntensityAndGradient);
  Run(I, p, DescriptorType::kDescriptorFieldsFirstOrder);
  Run(I, p, DescriptorType::kBitPlanes);
  Run(I, p, DescriptorType::kLatch);*/
  Run(I, p, DescriptorType::kCentralDifference);

  {
    std::vector<int> bytes{1, 2, 4, 8, 16, 32, 64};
    for(auto b : bytes) {
      LatchDescriptor desc(b, false, 1);
      desc.compute(I);

      auto t = TimeCode(1, [&]() { desc.compute(I); });
      printf("bytes: %d time %f\n", b, t);

      for(int i = 0; i < desc.numChannels(); ++i) {
        WriteChannel(Format("channel_%d", i), desc.getChannel(i));
      }

      break;
    }
  }

  {
    CentralDifferenceDescriptor desc;//(3, 0.75, 1.75);
    desc.compute(I);
    for(int i = 0; i < desc.numChannels(); ++i) {
      WriteChannel(Format("cd_%02d", i), desc.getChannel(i));
    }
  }

  return 0;
}

