#include "utils/dataset_loader_thread.h"
#include "utils/program_options.h"
#include "utils/viz.h"

#include "bpvo/vo.h"
#include "bpvo/debug.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"
#include "bpvo/config.h"
#include "bpvo/utils.h"
#include "bpvo/point_cloud.h"
#include "bpvo/config_file.h"
#include "bpvo/parallel.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#if defined(WITH_DMV)
#include <dmv/vo_data.h>
#include <utils/cv_cereal.h>
#include <cereal/archives/binary.hpp>
#endif

using namespace bpvo;

void writePointCloud(std::string prefix, int i, const PointCloud& pc,
                     float min_weight = 0.9f, float max_depth = 5.0f);

#if defined(WITH_DMV)
class DataWriterThread
{
  typedef std::pair<std::string, UniquePointer<dmv::VoData>> Data;
  typedef BoundedBuffer<Data> BufferType;

 public:
  DataWriterThread(std::string prefix, size_t capacity = 256)
      : _buffer(capacity), _prefix(prefix) {}

  ~DataWriterThread()
  {
    _stop = true;
    if(_thread && _thread->joinable())
      _thread->join();
  }

  void start()
  {
    _thread = make_unique<std::thread>(&DataWriterThread::start_, this);
  }

  void add(const cv::Mat& I, const PointCloud& pc);

 protected:
  int _counter = 0;
  BufferType _buffer;

  std::string _prefix;
  UniquePointer<std::thread> _thread;
  std::atomic<bool> _stop{false};

  void start_();
}; // DataWriterThread
#endif

int main(int argc, char** argv)
{
  fprintf(stdout, "%s\n", BPVO_BUILD_STR);

  printf("num threads %d\n", getNumThreads());

  ProgramOptions options;
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba.cfg", "config file")
      ("output,o", "output.txt", "trajectory output file")
      ("numframes,n", int(100), "number of frames to process")
      ("buffersize,b", int(16), "buffer size to load images")
      ("points,p", "", "store the points to files with the given prefix")
      ("dontshow,x", "do not show images")
#if defined(WITH_DMV)
      ("dmv,d", "", "store dmv data to the given prefix")
#endif
      .parse(argc, argv);

  auto max_frames = options.get<int>("numframes");
  auto conf_fn = options.get<std::string>("config");
  auto do_show = !options.hasOption("dontshow");
  auto points_prefix = options.get<std::string>("points");
#if defined(WITH_DMV)
  auto data_prefix = options.get<std::string>("dmv");
  UniquePointer<DataWriterThread> data_writer_thread;
  if(!data_prefix.empty()) {
    data_writer_thread = make_unique<DataWriterThread>(data_prefix);
    data_writer_thread->start();
  }
#endif

  auto dataset = Dataset::Create(conf_fn);

  int buffer_size = options.get<int>("buffersize");
  typename DatasetLoaderThread::BufferType image_buffer(buffer_size);

  AlgorithmParameters params(conf_fn);
  std::cout << "------- AlgorithmParameters -------" << std::endl;
  std::cout << params << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  auto maxTestLevel = params.maxTestLevel;
  auto vo = VisualOdometry(dataset.get(), params);

  Trajectory trajectory;
  UniquePointer<DatasetFrame> frame;

  std::cout << dataset->calibration() << std::endl;
  DatasetLoaderThread data_loader_thread(std::move(dataset), image_buffer);

  float min_weight = 0.9, max_depth = 5.0;
  int min_disparity = 0, num_disparities = 128; // for colorization purposes
  {
    ConfigFile cf(conf_fn);

    min_weight = cf.get<float>("minPointWeight", 0.9);
    max_depth = cf.get<float>("maxDepth", 5.0);

    min_disparity = cf.get<int>("minDisparity", 0);
    num_disparities = cf.get<int>("numberOfDisparities", 128);
  }

  cv::Mat display_image;

  double total_time = 0.0;
  int f_i = 0;
  while(f_i < max_frames)
  {
    if(image_buffer.pop(&frame, 2))
    {
      if(!frame)
      {
        printf("no more data\n");
        break;
      }

      Timer timer;
      auto result = vo.addFrame(frame->image().ptr<uint8_t>(),
                                frame->disparity().ptr<float>());
      double tt = timer.stop().count();
      total_time += (tt / 1000.0);

      if(result.pointCloud)
      {
        if(!points_prefix.empty())
          writePointCloud(points_prefix, f_i, *result.pointCloud, min_weight, max_depth);

        if(data_writer_thread)
          data_writer_thread->add(frame->image(), *result.pointCloud);
      }

      f_i += 1;
      trajectory.push_back(result.pose);

      int num_iters = result.optimizerStatistics[maxTestLevel].numIterations;
      if(num_iters == params.maxIterations)
      {
        printf("\n");
        Warn("maximum iterations reached %d\n", params.maxIterations);
      }

      fprintf(stdout, "Frame %05d %*.2f ms @ %*.2f Hz %03d iters %20s num_points %-*d\r",
              f_i-1, 6, tt, 5, (f_i - 1) / total_time,  num_iters,
              ToString(result.keyFramingReason).c_str(), 8, vo.numPointsAtLevel());
      fflush(stdout);

      if(do_show)
      {
        overlayDisparity(frame->image(), frame->disparity(), display_image,
                         0.5f, min_disparity, num_disparities);

        cv::imshow("image", display_image);
        int k = 0xff & cv::waitKey(1);
        if(k == ' ') k = cv::waitKey(0);
        if(k == 'q' || k == 27)
          break;
      }
    }
  }

  fprintf(stdout, "\n");
  Info("Processed %d frames @ %0.2f Hz\n", f_i, f_i / total_time);

  {
    auto output_fn = options.get<std::string>("output");
    if(!output_fn.empty()) {
      Info("Writing trajectory to %s\n", output_fn.c_str());
      if(!trajectory.writeCameraPath(output_fn)) {
        Warn("failed to write trajectory to %s\n", output_fn.c_str());
      }
    }

    std::ofstream ofs("poses.txt");
    for(size_t i = 0;i < trajectory.size(); ++i) {
      ofs << trajectory[i] << std::endl;
    }
    ofs.close();
  }

#if 0
  // will auto shutdown
  data_loader_thread.stop();
  while(data_loader_thread.isRunning())
    Sleep(10);
#endif

  Info("done\n");

  return 0;
}


void writePointCloud(std::string prefix, int i, const PointCloud& pc,
                     float min_weight, float max_depth)
{
  PointCloud pc_out;
  pc_out.reserve( pc.size() );
  for(size_t i = 0; i < pc.size(); ++i)
  {
    if(pc[i].weight() > min_weight && pc[i].xyzw().z() <= max_depth)
    {
      auto p = pc[i];
      p.xyzw() = pc.pose() * p.xyzw();
      pc_out.push_back(p);
    }
  }

  ToPlyFile(Format("%s_%05d.ply", prefix.c_str(), i), pc_out);
}

#if defined(WITH_DMV)

void DataWriterThread::add(const cv::Mat& I, const PointCloud& pc)
{
  auto p = UniquePointer<dmv::VoData>(new dmv::VoData(I, pc));
  auto fn = Format("%s_%04d.bin", _prefix.c_str(), _counter);

  //Timer timer;
  _buffer.push(Data(fn, std::move(p)));
  //double t = timer.stop().count();
  //printf("\npush time %f\n", t);

  _counter++;
}

void DataWriterThread::start_()
{
  Data d;

  try {
    while(!_stop) {
      if(_buffer.pop(&d, 5)) {
        //Timer timer;
        std::ofstream ofs(d.first, std::ios::binary);

        if(!ofs.is_open()) {
          Warn("failed to open %s\n", d.first.c_str());
        } else {
          cereal::BinaryOutputArchive ar(ofs);
          ar(*d.second);
        }

        //double t = timer.stop().count();
        //printf("write time %f\n\n", t);
      }
    }
  } catch(const std::exception& ex) {  }
}

#endif

