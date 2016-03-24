#include "bpvo/intensity_descriptor.h"
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"
#include "bpvo/timer.h"
#include "utils/kitti_dataset.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace bpvo;

int main()
{
  KittiDataset dataset("../conf/kitti_seq_0.cfg");
  IntensityDescriptor desc;

  auto frame = dataset.getFrame(0);

  desc.compute(frame->image());

  cv::Mat smap;
  desc.computeSaliencyMap(smap);

  const int radius = 1;
  IsLocalMax<float> is_local_max(smap.ptr<float>(), smap.cols, radius);

  std::vector<cv::Point> uv;
  for(int r = radius; r < smap.rows - radius - 1; ++r)
    for(int c = radius; c < smap.cols - radius -1; ++c)
      if(smap.at<float>(r,c) > 2.5f && is_local_max(r,c))
        uv.push_back(cv::Point(c,r));

  printf("pixel selection got %zu\n", uv.size());

  cv::Mat dimg;
  cv::cvtColor(frame->image(), dimg, cv::COLOR_GRAY2RGB);
  for(size_t i = 0; i < uv.size(); ++i)
    cv::circle(dimg, uv[i], 1, CV_RGB(255,255,0), 1);

  cv::imshow(Format("%zu points", uv.size()), dimg);

  double t = TimeCode(1000, [&] () { desc.computeSaliencyMap(smap); });
  printf("time %f ms\n", t);

  t = TimeCode(100, [&]() {
               std::vector<cv::Point> uv;
               for(int r = radius; r < smap.rows - radius - 1; ++r)
               for(int c = radius; c < smap.cols - radius -1; ++c)
               if(is_local_max(r,c)) uv.push_back(cv::Point(c,r));
               });
  printf("local max time %f\n", t);

  cv::waitKey(0);

  return 0;
}
