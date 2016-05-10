#include <bpvo/types.h>
#include <bpvo/dense_descriptor.h>
#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <opencv2/highgui.hpp>

using namespace bpvo;

static void Run(const cv::Mat& I, AlgorithmParameters p, DescriptorType t)
{
  p.descriptor = t;
  auto desc = DenseDescriptor::Create(p, t);
  desc->compute(I);

  auto tt = TimeCode(1000, [&]() { desc->compute(I); });
  printf("Descriptor: %s Time %f\n", ToString(t).c_str(), tt);

  delete desc;
}

int main()
{
  cv::Mat I = cv::imread("/home/halismai/lena.png", cv::IMREAD_GRAYSCALE);
  THROW_ERROR_IF(I.empty(), "failed to read image");

  AlgorithmParameters p;
  Run(I, p, DescriptorType::kIntensity);
  Run(I, p, DescriptorType::kIntensityAndGradient);
  Run(I, p, DescriptorType::kDescriptorFieldsFirstOrder);
  Run(I, p, DescriptorType::kBitPlanes);

  return 0;
}

