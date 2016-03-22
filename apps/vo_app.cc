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

  inline void showImages(const DatasetFrame* frame)
  {
    switch(_options.image_display_mode)
    {
      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparityOverlay:
        {
          overlayDisparity(frame->image(), frame->disparity(), _display_image,
                           0.5f, _min_disparity, _max_dispartiy);
          cv::imshow("image", _display_image);
        } break;
      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftOnly:
        {
          cv::imshow("image", frame->image());
        }; break;

      case VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparity:
        {
          colorizeDisparity(frame->disparity(), _display_image, _min_disparity, _max_dispartiy);

          cv::imshow("image", frame->image());
          cv::imshow("disparity", _display_image);
        } break;

      case VoApp::ViewerOptions::ImageDisplayMode::None:
        break; // unhanded case warning
    }
  }

 private:
  VoApp::ViewerOptions _options;
  cv::Mat _display_image;
  float _min_disparity = 0.0f;
  float _max_dispartiy = 64.f;
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

}; // VoApp::Impl

VoApp::ViewerOptions::ViewerOptions()
  : image_display_mode(ImageDisplayMode::ShowLeftAndDisparityOverlay) {}

VoApp::Options::Options()
    : trajectory_prefix()
    , points_prefix()
    , data_buffer_size(16)
    , max_num_frames(-1)
    , viewer_options()
{
}

VoApp::VoApp(Options options, std::string conf_fn, UniquePointer<Dataset> dataset)
    : _impl(make_unique<Impl>(options, conf_fn, std::move(dataset))) {}

VoApp::~VoApp() { stop(); }

void VoApp::run() { _impl->run(); }

void VoApp::stop() { _impl->stop(); }

bool VoApp::isRunning() const { return _impl->isRunning(); }

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
  if(_is_running) {
    if(_vo_thread && _vo_thread->joinable()) {
      _is_running = false;
      _vo_thread->join();
    }
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


void VoApp::Impl::mainLoop()
{
  _num_frames_processed = 0;

  _viewer->init();

  UniquePointer<DatasetFrame> frame;
  Result vo_result;
  int pc_idx = 0;
  double total_time = 0.0;
  while(_is_running)
  {
    if(_options.max_num_frames > 0 && _num_frames_processed > _options.max_num_frames)
      break;

    if(_data_buffer.pop(&frame, 5))
    {
      if(!frame) {
        break; // no more data
      }

      Timer timer;
      vo_result = _vo.addFrame(frame);
      double tt = timer.stop().count();
      total_time += tt;

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

      _viewer->showImages(frame.get());

      if(vo_result.pointCloud != nullptr && !_options.points_prefix.empty())
      {
        auto point_cloud_fn = Format("%s_%05d.ply", _options.points_prefix.c_str(), pc_idx);
        if(!writePointCloud(point_cloud_fn, *vo_result.pointCloud, _min_weight, _max_point_depth))
        {
          Warn("Failed to write point cloud to: '%s'\n", point_cloud_fn.c_str());
        }
      }

      ++_num_frames_processed;
    }
  }

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
  }

  _is_running = false;
}

} // bpvo

