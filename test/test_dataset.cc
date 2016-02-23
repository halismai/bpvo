#include <utils/dataset.h>
#include <utils/program_options.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>

using namespace bpvo;

inline cv::Mat ColorizeDisparity(const cv::Mat& src, int num_d = 128)
{
  double scale = 16.0 * (255.0 / (16.0 * num_d));

  cv::Mat ret;
  src.convertTo(ret, CV_8U, scale);
  cv::applyColorMap(ret, ret, cv::COLORMAP_JET);

  const auto* src_ptr = src.ptr<float>();
  for(int y = 0; y < ret.rows; ++y)
  {
    for(int x = 0; x < ret.cols; ++x)
    {
      if(src_ptr[y*src.cols + x] == 0)
        ret.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
    }
  }

  return ret;
}

int main(int argc, char** argv)
{
  bpvo::ProgramOptions options(argv[0]);
  options("config,c", "../conf/dataset.cfg", "config file")
      .parse(argc, argv);

  auto dataset = Dataset::Create(options.get<std::string>("config"));

  std::cout << dataset->calibration() << std::endl;

  UniquePointer<DatasetFrame> frame;

  int k = 0, f_i = 0;
  while( (frame = dataset->getFrame(f_i++)) && 'q' != k && 27 != k)
  {
    cv::imshow("image", frame->image());

    k = 0xff & cv::waitKey(10);
    if(k == ' ') k = 0xff & cv::waitKey(0);

    fprintf(stdout, "Frame %06d\r", f_i-1); fflush(stdout);

  }

  return 0;
}
