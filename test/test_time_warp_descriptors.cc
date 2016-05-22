#include <bpvo/types.h>
#include <bpvo/dense_descriptor.h>
#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace bpvo;

static inline
double Run(const cv::Mat& I, DescriptorType t, int interp = CV_INTER_LINEAR)
{
  AlgorithmParameters p;
  p.descriptor = t;

  auto desc =  DenseDescriptor::Create(p, t);
  desc->compute(I);

  std::vector<cv::Mat> dst(desc->numChannels());
  for(size_t i = 0; i < dst.size(); ++i) {
    dst[i].create(I.size(), CV_32FC1);
  }

  cv::Mat_<float> xmap(I.rows, I.cols);
  cv::Mat_<float> ymap(I.rows, I.cols);
  for(int i = 0; i < xmap.rows; ++i) {
    for(int j = 0; j < xmap.cols; ++j) {
      xmap(i,j) = i + (rand()/(double)RAND_MAX) - 0.5f;
      ymap(i,j) = i + (rand()/(double)RAND_MAX) - 0.5f;
    }
  }

  auto func = [&]() {
    // warp all channels
    for(size_t i = 0; i < dst.size(); ++i) {
      cv::remap(desc->getChannel(i), dst[i], xmap, ymap, interp);
    }
  };

  return TimeCode(500, func);
}

static inline
void RunAll(const cv::Mat& I, int interp)
{
  Run(I, DescriptorType::kIntensity, interp);
  Run(I, DescriptorType::kIntensityAndGradient, interp);
  Run(I, DescriptorType::kLaplacian, interp);
  Run(I, DescriptorType::kDescriptorFieldsFirstOrder, interp);
  Run(I, DescriptorType::kDescriptorFieldsSecondOrder, interp);
  Run(I, DescriptorType::kBitPlanes, interp);
}

int main()
{
  auto I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png", cv::IMREAD_GRAYSCALE);

  cv::resize(I, I, cv::Size(2560, 1920));

  std::vector<cv::Mat> I_pyr;
  cv::buildPyramid(I, I_pyr, 6);

#if 0
  for(int i = I_pyr.size()-1; i >= 0; --i) {
    printf("size: %ix%i\n", I_pyr[i].cols, I_pyr[i].rows);
    RunAll(I_pyr[i], CV_INTER_LINEAR);
    printf("\n");
  }
#endif

  std::vector<DescriptorType> dtypes{
    DescriptorType::kIntensity,
        DescriptorType::kIntensityAndGradient,
        DescriptorType::kLaplacian,
        DescriptorType::kDescriptorFieldsFirstOrder,
        DescriptorType::kDescriptorFieldsSecondOrder,
        DescriptorType::kBitPlanes
  };

  for(auto& t : dtypes) {
    printf("descriptor %24s\n", ToString(t).c_str());
    for(int i = I_pyr.size()-1; i >= 0; --i) {
      auto tt = Run(I_pyr[i], t, CV_INTER_LINEAR);
      printf("\tsize %ix%i time %0.2f ms\n", I_pyr[i].cols, I_pyr[i].rows, tt);
    }
  }

  return 0;
}


