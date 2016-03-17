#include "bpvo/rank.h"
#include "bpvo/timer.h"
#include "bpvo/utils.h"
#include <opencv2/highgui/highgui.hpp>

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

using namespace bpvo;

int main()
{
  cv::Mat I = cv::imread("/home/halismai/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png", cv::IMREAD_GRAYSCALE);

  if(I.empty()) {
    printf("failed to read image\n");
    return 0;
  }


  std::array<cv::Mat,9> rank_planes;
  completeRankPlanes(I, rank_planes);

  for(int i = 0; i < 9; ++i) {
    WriteImage<float>(Format("R%d", i), rank_planes[i]);
  }
}

