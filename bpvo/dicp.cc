#include "bpvo/dicp.h"
#include "bpvo/utils.h"
#include "bpvo/dense_descriptor.h"
#include "bpvo/imgproc.h"
#include <opencv2/core/core.hpp>

#include <Eigen/Geometry>

namespace bpvo {

struct Calibration
{
  Calibration(const Matrix33& K, float b)
      : _K(K), _baseline(b)
  {
    float fx = _K(0,0), fy = _K(1,1), bf = b * fx;
    _G <<
        fx, 0.0, 0.0, 0.0,
        0.0, fy, 0.0, 0.0,
        0.0, 0.0, 0.0, bf,
        0.0, 0.0, 1.0, 0.0;

    _G_inv = _G.inverse();
  }

  inline Matrix44 H(const Matrix44& T) const { return _G * T * _G_inv; }

  inline Point makePoint(float x, float y, float d) const
  {
    return Point(x - _K(0,2), y - _K(1,2), d, 1.0);
  }

  Matrix33 _K;
  float _baseline;
  Matrix44 _G, _G_inv;
}; // Calibration

struct OptimizationData
{
  typedef Eigen::Matrix<float,1,6> Jacobian;
  typedef typename EigenAlignedContainer<Jacobian>::type JacobianVector;
  typedef typename EigenAlignedContainer<Point>::type PointVector;

  PointVector _points;
  JacobianVector _jacobians;
  ResidualsVector _pixels;

  /**
   * set the data from a dense descritpor
   */
  inline void set(const Calibration& calib, const DenseDescriptor* desc, const DisparityPyramidLevel& D)
  {
    //
    // sets the data with pixel selection if the image size is big enoug,
    // otherwise we do the thing densely
    //
    cv::Mat smap;
    desc->computeSaliencyMap(smap);

    int rows = smap.rows, cols = smap.cols;
    IsLocalMax<float> is_local_max(nullptr, cols, -1);
    if(rows*cols >= 320*240) {
      is_local_max.setPointer( smap.ptr<const float>() );
      is_local_max.setStride( cols );
      is_local_max.setRadius( 1 );
    }

    const int border = 2;

    std::vector<uint16_t> uv;
    uv.reserve( rows * cols * 0.5 );
    for(int y = border; y < smap.rows - border - 1; ++y) {
      auto srow = smap.ptr<float>(y);
      for(int x = border; x < smap.cols - border - 1; ++x) {
        if(srow[x] > 0.01 && is_local_max(y,x)){
          float d = D(y,x);
          if(d > 0.01) {
            _points.push_back( calib.makePoint(x, y, d) );
            uv.push_back(x);
            uv.push_back(y);
          }
        }
      }
    }
  }

  inline void set(const Calibration& calib, const DenseDescriptor* desc, const PointVector& uvd)
  {
    clear();

    cv::Mat smap;
    desc->computeSaliencyMap(smap);

    int rows = desc->rows(), cols = desc->cols();
    IsLocalMax<float> is_local_max(nullptr, cols, -1);
    if(rows*cols >= 320*240) {
      is_local_max.setPointer( smap.ptr<const float>() );
      is_local_max.setRadius(1);
      is_local_max.setStride(cols);
    }

    std::vector<cv::Point2i> uv;
    uv.reserve(uvd.size());

    for(size_t i = 0; i < uvd.size(); ++i)
    {
      int y = std::round( uvd[i].y() ),
          x = std::round( uvd[i].x() );

      if(smap.at<float>(y,x) > 0.01f && is_local_max(y,x))
      {
        _points.push_back( calib.makePoint(x, y, uvd[i].z()) );
        uv.push_back( cv::Point2i(x,y) );
      }
    }

    auto f = calib._K(0,0),
         b = calib._baseline,
         bf = b*f,
         f_i = 1.0f / f;


    // set the image data and jacobians per channel
    for(int i = 0; i < desc->numChannels(); ++i)
    {
      const auto& C = desc->getChannel(i);
      for(const auto& p : uv)
      {
        auto x = p.x, y = p.y;
        auto Ix = 0.5f * (C.at<float>(y, x+1) - C.at<float>(y, x-1)),
             Iy = 0.5f * (C.at<float>(y+1, x) - C.at<float>(y-1, x));

        _pixels.push_back( C.at<float>(y,x) );

        auto z = x*Ix + y*Iy;
        float xx = static_cast<float>(-x),
              yy = static_cast<float>(-y);
        Jacobian J;
        J[0] = -f*Iy - f_i * (yy * z);
        J[1] = f*Ix + f_i * (xx * z);
        J[2] = y*Ix - x*Iy;
      }
    }

  }

  inline void resize(size_t n)
  {
    _points.resize( n );
    _jacobians.resize( n );
    _pixels.resize( n );
  }

  inline void clear() { this->resize(0); }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // OptimizationData

struct DirectDisparityIcp::Impl
{
  Impl(int pyr_level, const Matrix33& K, float b)
      : _pyr_level(pyr_level), _calib(K, b) {}

  inline void setData(const DenseDescriptor* desc, const cv::Mat& D)
  {
    typename OptimizationData::PointVector valid_points; //  points with valid disparity

    int rows = desc->rows(), cols = desc->cols();
    valid_points.resize( rows * cols * 0.5 );

    const DisparityPyramidLevel D_level( D, _pyr_level );
    for(int y = 2; y < rows - 3; ++y)
      for(int x = 2; x < cols - 3; ++x) {
        if(D_level(y,x) > 0.01)
          valid_points.push_back( Point(x, y, D_level(y,x), 1.0f) );
      }

    _image_data.set( _calib, desc, valid_points );

  }


  int _pyr_level;
  Calibration _calib;
  OptimizationData _image_data;
  OptimizationData _disparity_data;
}; // DirectDisparityIcp


DirectDisparityIcp::
DirectDisparityIcp(int pyr_level, const Matrix33& K, float b)
  : _impl(make_unique<Impl>(pyr_level, K, b)) {}

DirectDisparityIcp::~DirectDisparityIcp() {}

void DirectDisparityIcp::setData(const DenseDescriptor* desc, const cv::Mat& D)
{
  THROW_ERROR_IF( D.type() != CV_32FC1 , "disparity must be CV_32FC1");

  _impl->setData(desc, D);
}


} // bpvo

