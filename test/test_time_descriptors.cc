#include <bpvo/types.h>
#include <bpvo/dense_descriptor.h>
#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//
// timing script for the different descriptors
//

using namespace bpvo;

static inline
void Run(const cv::Mat& I, AlgorithmParameters p, DescriptorType t)
{
  p.descriptor = t;
  auto desc = DenseDescriptor::Create(p, t);
  desc->compute(I);

  auto tt = TimeCode(1000, [&]() { desc->compute(I); });
  printf("descriptor: %24s time %0.2f ms\n", ToString(t).c_str(), tt);

  delete desc;
}

static inline void RunAll(const cv::Mat& I)
{
  AlgorithmParameters p;

  printf("image size %ix%i\n", I.cols, I.rows);

  Run(I, p, DescriptorType::kIntensity);
  Run(I, p, DescriptorType::kIntensityAndGradient);
  Run(I, p, DescriptorType::kLaplacian);
  Run(I, p, DescriptorType::kDescriptorFieldsFirstOrder);
  Run(I, p, DescriptorType::kDescriptorFieldsSecondOrder);
  Run(I, p, DescriptorType::kBitPlanes);

  printf("\n");
}



int main()
{
  auto I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png", cv::IMREAD_GRAYSCALE);

  {
    cv::Mat tmp;
    cv::resize(I, tmp, cv::Size(320/4, 240/4));
    RunAll(tmp);

    cv::resize(I, tmp, cv::Size(320/2, 240/2));
    RunAll(tmp);

    cv::resize(I, tmp, cv::Size(320, 240));
    RunAll(tmp);
    return 0;
  }


  cv::resize(I, I, cv::Size(2*I.cols, 2*I.rows));
  RunAll(I);

  cv::resize(I, I, cv::Size(2*I.cols, 2*I.rows));
  RunAll(I);

  return 0;
}
