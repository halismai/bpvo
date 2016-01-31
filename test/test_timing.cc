#include <bpvo/template_data.h>
#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

using namespace bpvo;

std::vector<cv::Mat> MakePyramid(const cv::Mat& I, int N)
{
  std::vector<cv::Mat> ret(N);
  ret[0] = I;
  for(int i = 1; i < N; ++i)
    cv::pyrDown(ret[i-1], ret[i]);

  return ret;
}

static inline void RunTimingComputeData(const cv::Mat& I, const cv::Mat& D, int N)
{
  TemplateData data(AlgorithmParameters(), Matrix33::Identity(), 1.0f, 0);
  data.compute(I, D);

  auto t = TimeCode(N, [&] () { data.compute(I, D); });

  printf("compute() time is %0.2f for %d points %dx%d\n",
         t, data.numPoints(), I.cols, I.rows);
}

int main()
{
  auto I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png",
                      cv::IMREAD_GRAYSCALE);

  THROW_ERROR_IF(I.empty(), "could not read the image");

  auto I_pyr = MakePyramid(I, 5);
  auto D = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/groundtruth/disparity_maps/left/tsukuba_disparity_L_00001.png");
  THROW_ERROR_IF(D.empty(), "could not read disaprity");


  RunTimingComputeData(I, D, 100);

  return 0;
}

