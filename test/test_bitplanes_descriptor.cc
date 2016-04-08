#include "bpvo/bitplanes_descriptor.h"
#include "bpvo/timer.h"
#include "bpvo/census.h"
#include "bpvo/utils.h"

#include <opencv2/highgui/highgui.hpp>

#include <array>

using namespace bpvo;

static inline double RunTiming(BitPlanesDescriptor& desc, const cv::Mat& I)
{
  return TimeCode(100, [&]() { desc.compute(I); });
}

template <typename T> static
void WriteImage(std::string filename, const cv::Mat& I)
{
  FILE* fp = fopen(filename.c_str(), "w");
  if(!fp)
    return;

  for(int r = 0; r < I.rows; ++r)
  {
    for(int c = 0; c < I.cols; ++c)
      fprintf(fp, "%g ", (double) I.at<T>(r,c));
    fprintf(fp, "\n");
  }

  fclose(fp);
}

int main()
{
  cv::Mat I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png", cv::IMREAD_GRAYSCALE);

  if(I.empty()) {
    printf("failed to read image\n");
    return 0;
  }

  float sigma_ct = 0.75;
  float sigma_bp = 1.618;

  BitPlanesDescriptor desc(sigma_ct, sigma_bp);
  desc.compute(I);

  cv::Mat smap;
  desc.computeSaliencyMap(smap);

  WriteImage<float>("S", smap);

  for(int i = 0; i < desc.numChannels(); ++i)
  {
    WriteImage<float>(Format("C%d", i), desc.getChannel(i));
  }

  return 0;
}

