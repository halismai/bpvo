#include <bpvo/vo_output.h>
#include <bpvo/debug.h>

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
#include <list>

#include <boost/circular_buffer.hpp>

#if defined(WITH_DMV)
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <dmv/se3.h>
#endif

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
  typedef boost::circular_buffer<ImageData> ImageDataBuffer;

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
  void addNewPoints(const cv::Mat& image, const PointCloud& pc, const Matrix44& pose);
  void updateTrackVisibility(const ImageData& image_data, const Matrix44& pose);

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

  _image_data.push_back( ImageData(_frame_counter, image) );
  updateTrackVisibility(_image_data.back(), _trajectory.back().inverse());
  addNewPoints(image, pc, _trajectory.back());

  _frame_counter++;
  int nremoved = removeOldPoints();
  printf("removed %d\n", nremoved);
}

void PhotoBundle::Impl::
addNewPoints(const cv::Mat& image, const PointCloud& pc, const Matrix44& pose)
{
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

    ImagePoint uv(u, v);

    DescType desc;
    desc.setFromImage(image, uv);

    Point Xw = pose * p;
    ScenePointType scene_point(_frame_counter, Xw, std::move(desc),
                               typename ScenePointType::ZnccPatchType(image, uv));

    _points.push_back(scene_point);
  }
}

static inline ImagePoint projectPoint(const Matrix34& P, const Point& X)
{
  Eigen::Matrix<typename Point::Scalar,3,1> x = P * X;
  float z_i = 1.0f / x[2];
  return ImagePoint(x[0]*z_i, x[1]*z_i);
}

void PhotoBundle::Impl::updateTrackVisibility(const ImageData& image_data, const Matrix44& pose)
{
  typedef typename ScenePointType::ZnccPatchType ZnccPatchType;
  constexpr int R = ZnccPatchType::Radius;

  const auto& I = image_data.I();

  int max_rows = I.rows - R - 1,
      max_cols = I.cols - R - 1;

  int num_updated = 0;
  const Matrix34 P = _K.cast<float>() * pose.block<3,4>(0,0);

  for(size_t i = 0; i < _points.size(); ++i)
  {
    auto& p = _points[i];
    if(image_data.id() - p.referenceFrameId() < _config.maxFrameDistance)
    {
      ImagePoint uv = p.project(P);

      if(uv[0] >= R && uv[0] < max_cols && uv[1] >= R && uv[1] < max_rows)
      {
        auto score = ZnccPatchType(I, uv).zncc(p.znccPatch());
        if(score > _config.minZncc)
        {
          p.addFrameId(image_data.id());
          ++num_updated;
        }
      }
    }
  }

  printf("updated %d/%d\n", num_updated, (int) _points.size());
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
#if !defined(WITH_DMV)
  Fatal("compile WITH_DMV\n");
#else
  if(_image_data.size() < 3)
  {
    Warn("number of images is too small");
    return Result();
  }

  const auto id_start = _image_data.front().id();

  //
  // conver the trajectory to poses
  //
#endif
}

} // dmv
} // bpvo

