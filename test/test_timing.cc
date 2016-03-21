#include "bpvo/image_pyramid.h"
#include "bpvo/timer.h"
#include "bpvo/vo_frame.h"
#include "bpvo/debug.h"
#include "bpvo/dense_descriptor_pyramid.h"
#include "bpvo/template_data.h"
#include "utils/dataset.h"
#include "utils/viz.h"

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

void Run(const AlgorithmParameters& params, const cv::Mat& image, const cv::Mat& disparity,
         const Matrix33& K, float baseline)
{
  int N_rep = 1000;

  ImagePyramid image_pyramid(params.numPyramidLevels);
  {
    image_pyramid.compute(image);

    auto tt = TimeCode(N_rep, [&]() { image_pyramid.compute(image); });
    printf("ImagePyramid time: %0.2f ms\n", tt);
  }

  DenseDescriptorPyramid d_pyr(params);
  d_pyr.init(image_pyramid);
  {
    auto tt = TimeCode(N_rep, [&]() { d_pyr.init(image_pyramid); });
    printf("DenseDescriptorPyramid: %0.2f ms\n", tt);
  }

  VisualOdometryFrame vo_frame(K, baseline, params);
  vo_frame.setData(image, disparity);
  {
    // this will be the same time as ImagePyramid::compute +
    // DenseDescriptorPyramid::init with a slight addition from the copy of the
    // disaprity
    auto tt = TimeCode(N_rep, [&]() { vo_frame.setData(image, disparity); });
    printf("VisualOdometryFrame::setData %0.2f ms\n", tt);
  }

  vo_frame.setTemplate();
  {
    auto tt = TimeCode(N_rep, [&]() { vo_frame.setTemplate(); });
    printf("VisualOdometryFrame::setTemplate %0.2f ms\n", tt);
  }

  auto tdata= vo_frame.getTemplateDataAtLevel(0);
  ResidualsVector residuals;
  ValidVector valid;
  Matrix44 pose(Matrix44::Identity());
  tdata->computeResiduals(vo_frame.getDenseDescriptorAtLevel(0), pose, residuals, valid);

  {
    auto tt = TimeCode(N_rep,
                       [&](){
                       for(int i = 0; i < params.numPyramidLevels; ++i)
                       {
                        vo_frame.getTemplateDataAtLevel(i)->computeResiduals(
                            vo_frame.getDenseDescriptorAtLevel(i), pose, residuals, valid);
                       }
                       });

    printf("TemplateData::computeResiduals %0.2f ms\n", tt);
  }

}

int main()
{
  auto dataset = Dataset::Create("../conf/kitti.cfg");
  auto frame = dataset->getFrame(0);

  AlgorithmParameters params;
  params.descriptor = DescriptorType::kIntensity;
  params.numPyramidLevels = 4;

  const auto image = frame->image();
  const auto disparity = frame->disparity();

  std::cout << "image size: " << image.size() << std::endl;

  cv::Mat dst;
  overlayDisparity(image, disparity, dst);
  cv::imshow("image", dst);
  cv::waitKey(0);


  Info("Raw Intensity\n");
  Run(params, image, disparity, dataset->calibration().K, dataset->calibration().baseline);
  printf("\n");


  Info("Bit-Planes\n");
  params.descriptor = DescriptorType::kBitPlanes;
  Run(params, image, disparity, dataset->calibration().K, dataset->calibration().baseline);


  return 0;
}

