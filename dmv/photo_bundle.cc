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

  void addData(const cv::Mat&, const Matrix44&, const PointVector&, const WeightsVector&);
  Result startOptimization();


  int _frame_counter = 0;
  Eigen::Matrix<double,3,3> _K; //< the camera calibration
  PhotoBundleConfig _config;    //< algorithm config

  ImageDataBuffer _image_data; //< image data
  std::vector<ScenePointType> _points; //< points

  Trajectory _trajectory;

 protected:
  void addNewPoints(const cv::Mat&, const PointVector&, const WeightsVector&);
  void removeOldPoints();
}; // PhotoBundle::Impl

PhotoBundle::PhotoBundle(const Matrix33& K, const PhotoBundleConfig& config)
  : _impl(make_unique<Impl>(K, config)) {}

PhotoBundle::~PhotoBundle() {}

void PhotoBundle::addData(const cv::Mat& I, const Matrix44& pose, const PointVector& points,
                          const WeightsVector& weights)
{
  _impl->addData(I, pose, points, weights);
}

auto PhotoBundle::optimize() -> Result { return _impl->startOptimization(); }

void PhotoBundle::Impl::
addData(const cv::Mat& image, const Matrix44& pose, const PointVector& points,
        const WeightsVector& weights)
{
  _trajectory.push_back(pose);

  ImageData data;
  std::thread t1([&] () { data.set(_frame_counter, image);} );
  std::thread t2(&Impl::addNewPoints, this, std::ref(image), std::ref(points), std::ref(weights));

  t1.join();
  t2.join();

  _image_data.push_back(std::move(data));
  _frame_counter++;
}

void PhotoBundle::Impl::
addNewPoints(const cv::Mat& image, const PointVector& points, const WeightsVector& weights)
{
  THROW_ERROR_IF( points.size() != weights.size(), "size mismatch" );

  removeOldPoints();

  PointVector tmp_points;
  const PointVector* point_ptr = nullptr;
  if((int) points.size() > _config.maxPointsPerImage)
  {
    typedef std::pair<typename WeightsVector::value_type, size_t> WeightWithIndex;
    std::vector<WeightWithIndex> tmp(weights.size());
    for(size_t i = 0; i < weights.size(); ++i)
      tmp[i] = WeightWithIndex(weights[i], i);

    std::nth_element(tmp.begin(), tmp.begin() + _config.maxPointsPerImage, tmp.end(),
                     [](const WeightWithIndex& a, const WeightWithIndex& b) {
                        return a.first < b.first; });

    tmp_points.resize( _config.maxPointsPerImage );
    for(int i = 0; i < _config.maxPointsPerImage; ++i)
      tmp_points[i] = points[tmp[i].second];

    point_ptr = &tmp_points;
  } else
  {
    point_ptr = &points;
  }

  const auto& pose = _trajectory.back();
  for(const auto p : *point_ptr)
  {
    float u = (_K(0,0) * p.x()) / p.z() + _K(0,2),
          v = (_K(1,1) * p.x()) / p.z() + _K(1,2);

    Descriptor desc;
    desc.setFromImage(image, ImagePoint(u,v));
    _points.push_back( ScenePointType(_frame_counter, pose * p, std::move(desc)) );
  }

}

void PhotoBundle::Impl::removeOldPoints()
{
  auto is_point_old = [=](const ScenePointType& p)
  {
    return _frame_counter - p.referenceFrameId() > _config.bundleWindowSize;
  }; // is_point_old

  auto it = std::remove_if(_points.begin(), _points.end(), is_point_old);
  _points.erase(it, _points.end());
}

auto PhotoBundle::Impl::startOptimization() -> Result
{
  return Result();
}

} // dmv
} // bpvo

