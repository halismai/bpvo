#include <bpvo/vo_output.h>

#include <dmv/photo_bundle.h>
#include <dmv/image_data.h>
#include <dmv/scene_point.h>
#include <dmv/descriptor.h>
#include <dmv/patch.h>

#include <opencv2/core/core.hpp>

#include <algorithm>
#include <iostream>
#include <thread>
#include <utility>
#include <deque>

namespace bpvo {
namespace dmv {

std::ostream& operator<<(std::ostream& os, const PhotoBundle::Result& r)
{
  os << "Number of points = " << r.points.size() << "\n";
  os << "Number of poses  = " << r.trajectory.size() << "\n";
  os << "Number of iterations = " << r.numIterations << "\n";
  os << "Initial cost = " << r.initialCost << "\n";
  os << "Final cost = " << r.finalCost << "\n";
  os << "Time = " << r.timeInSeconds << " seconds\n";
  os << "Termination message = " << r.termMessage;

  return os;
}

struct PhotoBundle::Impl
{
 public:
  typedef Patch3x3 DescType;
  typedef DescriptorBase<DescType> Descriptor;
  typedef ScenePoint<Descriptor> ScenePointType;
  typedef std::deque<ImageData> ImageDataBuffer;

 public:
  Impl(const Matrix33& K, const PhotoBundleConfig& config)
      : _K(K.cast<double>()), _config(config), _image_data(_config.bundleWindowSize) {}

  void addData(const cv::Mat& image, const PointCloud& pc);

  Result startOptimization();


  int _frame_counter = 0;
  Eigen::Matrix<double,3,3> _K; //< the camera calibration
  PhotoBundleConfig _config;    //< algorithm config

  ImageDataBuffer _image_data; //< image data
  std::vector<ScenePointType> _points; //< points

  Trajectory _trajectory;

 protected:
  void addNewPoints(const cv::Mat& image, const PointCloud& pc);

  /** \return number of old points removed */
  int removeOldPoints();
}; // PhotoBundle::Impl

PhotoBundle::PhotoBundle(const Matrix33& K, const PhotoBundleConfig& config)
  : _impl(make_unique<Impl>(K, config)) {}

PhotoBundle::~PhotoBundle() {}

void PhotoBundle::addData(const VoOutput* frame)
{
  // we load the image here, so we won't have to load it multiple times in case
  // it is read from disk
  if(frame) {
    auto image = frame->image();
    _impl->addData(image, frame->pointCloud());
  }
}

auto PhotoBundle::optimize() -> Result { return _impl->startOptimization(); }


void PhotoBundle::Impl::addData(const cv::Mat& image, const PointCloud& pc)
{
  _trajectory.push_back(pc.pose());

  ImageData image_data;
#define ADD_DATA_THREADED 0

#if ADD_DATA_THREADED
  std::thread t1([&]() { image_data.set(_frame_counter, image); });
  std::thread t2(&Impl::addNewPoints, this, std::ref(image), std::ref(pc));

  t1.join();
  t2.join();
#else
  image_data.set(_frame_counter, image);
  addNewPoints(image, pc);
#endif

  _frame_counter++;
  _image_data.push_back(image_data);
}

void PhotoBundle::Impl::
addNewPoints(const cv::Mat& image, const PointCloud& pc)
{
  int nremoved = removeOldPoints();
  printf("removed %d\n", nremoved);

  int npts = pc.size();

  PointVector points;
  if(npts > _config.maxPointsPerImage)
  {
    //
    // keep only the n-th highest weighted points
    //
    typedef std::pair<float, int> WeightWithIndex;
    std::vector<WeightWithIndex> tmp(npts);
    for(int i = 0; i < npts; ++i)
      tmp[i] = WeightWithIndex(pc[i].weight(), i);

    std::nth_element(tmp.begin(), tmp.begin() + _config.maxPointsPerImage, tmp.end(),
                     [](const WeightWithIndex& a, const WeightWithIndex& b) {
                       return a.first > b.first; });

    points.resize(  _config.maxPointsPerImage );
    for(int i = 0; i < _config.maxPointsPerImage; ++i)
      points[i] = pc[ tmp[i].second ].xyzw();
  } else
  {
    points.resize(npts);
    for(int i = 0; i < npts; ++i)
      points[i] = pc[i].xyzw();
  }

  auto fx = _K(0,0), fy = _K(1,1), cx = _K(0,2), cy = _K(1,2);
  for(size_t i = 0; i < points.size(); ++i)
  {
    const auto& p = points[i];
    float u = (fx * p.x()) / p.z() + cx,
          v = (fy * p.y()) / p.z() + cy;

    DescType desc;
    desc.setFromImage(image, ImagePoint(u,v));

    Point Xw = pc.pose() * p;
    _points.push_back(ScenePointType(_frame_counter, Xw, std::move(desc)));
  }
}


int PhotoBundle::Impl::removeOldPoints()
{
  int old_size = _points.size();

  auto is_point_old = [=](const ScenePointType& p)
  {
    return _frame_counter - p.referenceFrameId() > _config.bundleWindowSize;
  }; // is_point_old

  auto it = std::remove_if(_points.begin(), _points.end(), is_point_old);
  _points.erase(it, _points.end());

  return old_size - _points.size();
}

auto PhotoBundle::Impl::startOptimization() -> Result
{
  return Result();
}

} // dmv
} // bpvo


