/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#include "apps/vo_app.h"
#include "bpvo/vo.h"
#include "bpvo/utils.h"
#include "bpvo/config_file.h"
#include "bpvo/point_cloud.h"
#include "bpvo/timer.h"
#include "bpvo/trajectory.h"

#include "utils/dataset.h"
#include "utils/dataset_loader_thread.h"
#include "utils/viz.h"

#include <atomic>
#include <thread>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

namespace bpvo {

class Viewer
{
 public:
  Viewer(const VoApp::ViewerOptions& options)
      : _options(options) {}

  inline void setMinDisparity(float v) { _min_disparity = v; }
  inline void setMaxDisparity(float v) { _max_dispartiy = v; }

  inline float getMinDisparity() const { return _min_disparity; }
  inline float getMaxDisparity() const { return _max_dispartiy; }

  inline void init()
  {
    switch(_options.image_display_mode)
    {
      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftOnly:
      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparityOverlay:
        {
          cv::namedWindow("image");
        } break;

      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparity:
        {
          cv::namedWindow("image");
          cv::namedWindow("disparity");
        } break;

      case VoApp::ViewerOptions::ImageDisplayMode::None:
        break; // nothing
    }
  }

  inline bool showImages(const DatasetFrame* frame)
  {
    bool keep_running = true;
    switch(_options.image_display_mode)
    {
      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparityOverlay:
        {
          overlayDisparity(frame->image(), frame->disparity(), _display_image,
                           0.5, _min_disparity, _max_dispartiy);
          cv::imshow("image", _display_image);
        } break;
      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftOnly:
        {
          cv::imshow("image", frame->image());
        }; break;

      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparity:
        {
          colorizeDisparity(frame->disparity(), _display_image, _min_disparity,
                            _max_dispartiy);

          cv::imshow("image", frame->image());
          cv::imshow("disparity", _display_image);
        } break;

      case VoApp::ViewerOptions::ImageDisplayMode::None:
        break; // unhanded case warning
    }

    if(_options.image_display_mode != VoApp::ViewerOptions::ImageDisplayMode::None)
      keep_running = handleKey();

    return keep_running;
  }

 private:
  VoApp::ViewerOptions _options;
  cv::Mat _display_image;
  float _min_disparity = 1.0f;
  float _max_dispartiy = 128.f;

  inline bool handleKey()
  {
    int k = cv::waitKey(5) & 0xff;
    if(k == ' ') // pause
      k = cv::waitKey(0) & 0xff;

    return k != 'q';
  }
}; // Viewer

struct VoApp::Impl
{
  typedef typename DatasetLoaderThread::BufferType DataBufferType;

  Impl(Options, std::string, UniquePointer<Dataset>);
  ~Impl();

  void run();
  void stop();

  inline bool isRunning() const { return _is_running; }

  std::atomic<bool> _is_running;
  Options _options;
  DataBufferType _data_buffer;
  AlgorithmParameters _params;
  VisualOdometry _vo;
  DatasetLoaderThread _data_loader_thread;

  UniquePointer<Viewer> _viewer;
  UniquePointer<std::thread> _vo_thread;

  int _num_frames_processed;

  void mainLoop();

  void initViewer();
  void showImages(const DatasetFrame*);

  cv::Mat _display_image;

  float _min_weight = 0.9;

  //
  // maximum depth to use for display/writing to disk
  //
  float _max_point_depth = 5.0;


  std::vector<float> _iter_time_ms;
  std::vector<int> _iter_num;

}; // VoApp::Impl

VoApp::ViewerOptions::ViewerOptions()
  : image_display_mode(ImageDisplayMode::ShowLeftAndDisparityOverlay) {}

VoApp::Options::Options()
    : trajectory_prefix()
    , points_prefix()
    , data_buffer_size(16)
    , max_num_frames(-1)
    , viewer_options()
    , store_iter_time(false)
    , store_iter_num(false)
{
}

VoApp::VoApp(Options options, std::string conf_fn, UniquePointer<Dataset> dataset)
    : _impl(make_unique<Impl>(options, conf_fn, std::move(dataset)))
{
  Sleep(100); // wait for things to start up
}

VoApp::~VoApp() { stop(); }

void VoApp::run() { _impl->run(); }

void VoApp::stop() { _impl->stop(); }

bool VoApp::isRunning() const { return _impl->isRunning(); }

const Trajectory& VoApp::getTrajectory() const
{
  return _impl->_vo.trajectory();
}

const std::vector<float>& VoApp::getIterationTime() const
{
  return _impl->_iter_time_ms;
}

const std::vector<int>& VoApp::getNumIterations() const
{
  return _impl->_iter_num;
}

VoApp::Impl::
Impl(Options options, std::string conf_fn, UniquePointer<Dataset> dataset)
  : _is_running{false}
  , _options(options)
  , _data_buffer(_options.data_buffer_size)
  , _params(conf_fn)
  , _vo(dataset.get(), _params)
  , _data_loader_thread(std::move(dataset), _data_buffer)
  , _viewer(make_unique<Viewer>(_options.viewer_options))
  , _num_frames_processed(0)
{
  // parse additional stuff from the config file we could query the classes for
  // this, but it gets messy so we re-parse them here
  try {
    ConfigFile cf(conf_fn);
    _min_weight = cf.get<float>("minPointWeight", 0.9);

    _viewer->setMinDisparity( cf.get<float>("minDisparity", 1.0f) );
    _viewer->setMaxDisparity( cf.get<float>("numberOfDisparities", 64.0f) +
                            _viewer->getMinDisparity() );

    std::string loss = cf.get<std::string>("lossFunction", "");
    if(icompare("L2", loss)) // there is no weighting when doing L2 loss
      _min_weight = 0.0f;
  } catch(...) {}
}

VoApp::Impl::~Impl() { stop(); }

void VoApp::Impl::run()
{
  THROW_ERROR_IF(_is_running, "VoApp is already running");

  _is_running = true;
  _vo_thread = make_unique<std::thread>(&VoApp::Impl::mainLoop, this);
}

void VoApp::Impl::stop()
{
  _is_running = false;
  if(_vo_thread && _vo_thread->joinable()) {
    _vo_thread->join();
  }
}

static inline
bool writePointCloud(std::string fn, const PointCloud& pc, float min_weight, float max_depth)
{
  PointCloud pc_out;
  pc_out.reserve( pc.size() );

  for(size_t i = 0; i < pc.size(); ++i)
  {
    if(pc[i].weight() > min_weight && pc[i].xyzw().z() <= max_depth) {
      auto p = pc[i];
      p.xyzw() = pc.pose() * p.xyzw();
      pc_out.push_back(p);
    }
  }

  return ToPlyFile(fn, pc_out);
}


template <typename T> static inline
bool WriteVector(std::string fn, const std::vector<T>& data)
{
  std::ofstream ofs(fn);
  if(ofs.is_open()) {
    for(const auto& v : data)
      ofs << v << "\n";

    return true;
  } else {
    return false;
  }
}

void VoApp::Impl::mainLoop()
{
  _num_frames_processed = 0;

  _iter_time_ms.resize(0);
  _iter_num.resize(0);

  _viewer->init();

  UniquePointer<DatasetFrame> frame;
  Result vo_result;
  int pc_idx = 0;
  double total_time = 0.0;
  while(_is_running && _data_loader_thread.isRunning())
  {
    if(_options.max_num_frames > 0 && _num_frames_processed > _options.max_num_frames)
      break;

    if(_data_buffer.pop(&frame, 5))
    {
      if(!frame) {
        Warn("no more data\n");
        break; // no more data
      }

      Timer timer;
      vo_result = _vo.addFrame(frame);
      double tt = timer.stop().count();
      total_time += (tt / 1000.0);

      int num_iters = vo_result.optimizerStatistics[_params.maxTestLevel].numIterations;
      if(num_iters == _params.maxIterations)
      {
        // display a warning that the max number of iterations has been reached,
        // this could indicate insufficient number of iterations specified in
        // AlgorithmParameters, or the algorithm is having trouble estimating
        // the pose for this frame
        fprintf(stdout, "\n");
        Warn("Max iterations reached %d at frame %d\n", _params.maxIterations,
             _num_frames_processed);
      }

      fprintf(stdout, "Frame %05d %*.2f ms @ %*.2f Hz %03d iters %20s num_points %-*d\r",
              _num_frames_processed, 6, tt, 5, _num_frames_processed / total_time,  num_iters,
              ToString(vo_result.keyFramingReason).c_str(), 8, _vo.numPointsAtLevel());
      fflush(stdout);

      if(_options.store_iter_num)
        _iter_num.push_back( num_iters );

      if(_options.store_iter_time)
        _iter_time_ms.push_back( tt );

      if(!_viewer->showImages(frame.get()))
        break;

      if(vo_result.pointCloud != nullptr && !_options.points_prefix.empty())
      {
        auto point_cloud_fn = Format("%s_%05d.ply", _options.points_prefix.c_str(), pc_idx);
        if(!writePointCloud(point_cloud_fn, *vo_result.pointCloud, _min_weight, _max_point_depth))
        {
          Warn("Failed to write point cloud to: '%s'\n", point_cloud_fn.c_str());
        }

        ++pc_idx;
      }

      ++_num_frames_processed;
    }
  }

  _data_loader_thread.stop();

  if(!_options.trajectory_prefix.empty())
  {
    const auto& trajectory = _vo.trajectory();
    const auto camera_path_fn = Format("%s_path.txt", _options.trajectory_prefix.c_str());
    Info("Writing camera path to '%s'\n", camera_path_fn.c_str());
    if(!trajectory.writeCameraPath(camera_path_fn)) {
      Warn("Failed to write camera trajectory path");
    }

    const auto poses_fn = Format("%s_poses.txt", _options.trajectory_prefix.c_str());
    Info("Writing poses to '%s'\n", poses_fn.c_str());
    if(!trajectory.write(poses_fn))
      Warn("Failed to write camera poses\n");

    if(_options.store_iter_num) {
      WriteVector(Format("%s_iter_time.txt", _options.trajectory_prefix.c_str()), _iter_time_ms);
    }

    if(_options.store_iter_num) {
      WriteVector(Format("%s_iter_num.txt", _options.trajectory_prefix.c_str()), _iter_num);
    }
  }

  _is_running = false;
}

} // bpvo

