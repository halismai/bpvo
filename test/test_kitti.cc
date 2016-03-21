#include "bpvo/vo_kf2.h"
#include "bpvo/trajectory.h"
#include "bpvo/timer.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"

#include "utils/dataset.h"
#include "utils/program_options.h"
#include "utils/viz.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;

template <typename T> static inline
void WriteVector(std::string filename, const std::vector<T>& v)
{
  std::ofstream ofs(filename);
  for(auto v_i : v)
    ofs << v_i << "\n";

  ofs.close();
}


void RunKittiSquence(int sequence_number, std::string conf_fn, std::string output_prefix)
{
  {
    std::ofstream ofs("/tmp/kitti.cfg");
    ofs << "DataSet = kitti\n"
        << "DataSetRootDirectory = ~/data/kitti/dataset/\n"
        << "SequenceNumber = " << sequence_number << "\n"
        << "StereoAlgorithm = BlockMatching\n"
        << "SADWindowSize = 9\n"
        << "minDisparity = 1\n"
        << "numberOfDisparities = 96\n"
        << "trySmallerWindows = 1\n"
        << "scaleBy = 1\n";
    ofs.close();
  }

  auto dataset = Dataset::Create("/tmp/kitti.cfg");

  AlgorithmParameters params(conf_fn);
  VisualOdometryKeyFraming vo(dataset->calibration().K, dataset->calibration().baseline,
                              dataset->imageSize(), params);

  if(params.descriptor == DescriptorType::kBitPlanes)
    cv::setNumThreads(1);

  std::vector<int> num_iters;
  std::vector<int> kf_inds;
  std::vector<double> time_ms;;

  Result result;
  Trajectory trajectory;
  double total_time = 0.0;

  cv::Mat display_image;
  int f_i = 0;
  while( true )
  {
    auto frame = dataset->getFrame(f_i);
    if(!frame)
      break;

    Timer timer;
    result = vo.addFrame(frame->image().ptr<uint8_t>(), frame->disparity().ptr<float>());
    auto t_ms = timer.stop().count();
    total_time += t_ms / 1000.0; // seconds

    num_iters.push_back( result.optimizerStatistics[params.maxTestLevel].numIterations );
    time_ms.push_back(t_ms);
    if(result.isKeyFrame)
      kf_inds.push_back(f_i);

    trajectory.push_back( result.pose );

    fprintf(stdout, "Frame %04d %0.2f Hz [keyframe ? %s] iterations: %d\n",
            f_i, f_i / total_time, result.isKeyFrame ? "YES" : " NO",
            result.optimizerStatistics[0].numIterations);

    //fflush(stdout);

    f_i += 1;

    overlayDisparity(frame->image(), frame->disparity(), display_image,
                     0.5, 1, 96);
    cv::imshow("image", display_image);
    int k = cv::waitKey(5) & 0xff;
    if(k == ' ')
      k = cv::waitKey(0) & 0xff;
    if(k == 'q')
      break;
  }

  output_prefix = Format("%s_%02d", output_prefix.c_str(), sequence_number);
  trajectory.writeCameraPath(output_prefix + "_path.txt");
  WriteVector(output_prefix + "_iters.txt", num_iters);
  WriteVector(output_prefix + "_time_ms.txt", time_ms);
  WriteVector(output_prefix + "_kf_inds.txt", kf_inds);

  {
    // write the poses in kitti format
    std::ofstream ofs(output_prefix + "_trajectory.txt");
    for(size_t i = 0; i < trajectory.size(); ++i) {
      const auto& T = trajectory[i];
      for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 4; ++j)
          ofs << T(i,j) << " ";
      ofs << std::endl;
    }

    ofs.close();
  }
}

int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("config,c", "", "config file")
      ("output,o", "", "output prefix to store results")
      .parse(argc, argv);

  auto conf_fn = options.get<std::string>("config");
  auto output_prefix = options.get<std::string>("output");

  for(int i = 0; i < 11; ++i)
  {
    Info("SequenceNumber %d\n", i);
    RunKittiSquence(i, conf_fn, output_prefix);
  }

  return 0;
}

