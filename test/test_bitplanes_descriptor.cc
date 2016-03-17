#include "bpvo/bitplanes_descriptor.h"
#include "bpvo/timer.h"
#include "bpvo/census.h"
#include "bpvo/rank.h"

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

  {
    BitPlanesDescriptor desc(-1.0, -1.0);

    auto t = RunTiming(desc, I);
    printf("no sigma %f\n", t);
  }

  {
    BitPlanesDescriptor desc(0.75, -1.0);
    auto t = RunTiming(desc, I);
    printf("ct sigma %f\n", t);
  }

  {
    BitPlanesDescriptor desc(0.75, 0.5);
    auto t = RunTiming(desc, I);
    printf("ct + bp sigma %f\n", t);

    WriteImage<float>("B", desc.getChannel(0));

    cv::Mat C = census(I, -0.75);
    WriteImage<uint8_t>("C", C);
  }

  WriteImage<uint8_t>("R", rankTransform(I));

  return 0;
}

