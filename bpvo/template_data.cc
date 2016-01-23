#include "bpvo/template_data.h"
#include "bpvo/bitplanes.h"
#include <opencv2/core.hpp>
#include <utility>

namespace bpvo {

TemplateData::TemplateData(const AlgorithmParameters& p, const Matrix33& K,
                           const float& baseline, int pyr_level)
    : _K(K), _baseline(baseline), _pyr_level(pyr_level)
    , _sigma_ct(p.sigmaPriorToCensusTransform)
    , _sigma_bp(p.sigmaBitPlanes) {}

void TemplateData::reserve(size_t n)
{
  _jacobians.reserve(n);
  _points.reserve(n);
  _pixels.reserve(n);
}

auto TemplateData::X(size_t i) const -> const typename PointVector::value_type&
{
  assert( i < _points.size() );
  return _points[i];
}

auto TemplateData::X(size_t i) -> typename PointVector::value_type&
{
  assert( i < _points.size() );
  return _points[i];
}

auto TemplateData::J(size_t i) const -> const typename JacobianVector::value_type&
{
  assert( i < _jacobians.size() );
  return _jacobians[i];
}

auto TemplateData::J(size_t i) -> typename JacobianVector::value_type&
{
  assert( i < _jacobians.size() );
  return _jacobians[i];
}

auto TemplateData::I(size_t i) const -> const typename PixelVector::value_type&
{
  assert( i < _pixels.size() );
  return _pixels[i];
}

auto TemplateData::I(size_t i) -> typename PixelVector::value_type&
{
  assert( i < _pixels.size() );
  return _pixels[i];
}

/**
 */
struct IsLocalMax
{
  inline IsLocalMax(int stride, int radius)
      : _stride(stride), _radius(radius)
  {
    assert( _radius > 0 && _stride > 0);
  }

  inline void setStride(int s) { _stride = s; }
  inline void setRadius(int r) { _radius = r; }

  template <typename T> inline
  bool operator()(int row, int col, const T* ptr) const
  {
    auto v = ptr[row*_stride + col];
    for(int r = -_radius; r <= _radius; ++r)
      for(int c = -_radius; c <= _radius; ++c)
        if(!(!r && !c) && ptr[(row+r)*_stride + col + c] >= v)
          return false;

    return true;
  }

 private:
  int _stride, _radius;
}; // IsLocalMax


static void computeGradientAbsMag(const BitPlanesData& data, cv::Mat& dst)
{
  dst.create(data.I[0].size(), CV_32FC1);
  dst.setTo(cv::Scalar(0.0f));

  int n = dst.rows * dst.cols;
  auto dst_ptr = dst.ptr<float>();
  for(const auto& g : data.G) {

    auto src_ptr = g.ptr<float>();
#pragma omp simd
    for(int i = 0; i < n; ++i)
    {
      dst_ptr[i] += std::fabs(src_ptr[2*i+0]) + std::fabs(src_ptr[2*i+1]);
    }
  }
}

typedef Eigen::Matrix<float,2,6> WarpJacobian;

WarpJacobian MakeJacobian(const Point& xyz, float fx, float fy)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];

  WarpJacobian J;
  J(0,0) = -fx*x*y*1.0/(z*z);
  J(0,1) =  fx*1.0/(z*z)*(x*x+z*z);
  J(0,2) = -(fx*y)/z;
  J(0,3) = fx/z;
  J(0,4) = 0.0;
  J(0,5) = -fx*x*1.0/(z*z);

  J(1,0) = -fy*1.0/(z*z)*(y*y+z*z);
  J(1,1) = fy*x*y*1.0/(z*z);
  J(1,2) = (fy*x)/z;
  J(1,3) = 0.0;
  J(1,4) = fy/z;
  J(1,5) = -fy*y*1.0/(z*z);

  return J;
}

void TemplateData::compute(const cv::Mat& image, const cv::Mat& disparity)
{
  assert( disparity.type() == cv::DataType<float>::type );
  assert( image.type() == cv::DataType<uint8_t>::type );

  auto bitplanes = computeBitPlanes(image, _sigma_ct, _sigma_bp);
  auto do_nonmax_supp = image.rows*image.cols >= AlgorithmParameters::MIN_NUM_FOR_PIXEL_PSELECTION;

  cv::Mat_<float> G;
  computeGradientAbsMag(bitplanes, G);
  auto gmag_ptr = G.ptr<const float>();

  int radius = 1;
  IsLocalMax is_local_max(image.cols, radius);

  float fx = _K(0,0), fy = _K(1,1), cx = _K(0,2), cy = _K(1,2);
  float Bf = _baseline * fx;

  const cv::Mat_<float>& D = (const cv::Mat_<float>&) disparity;
  int ds = 1 << _pyr_level;


  auto n_reserve = 0.5f*image.rows*image.cols;
  this->reserve(n_reserve);

  typename EigenAlignedContainer<WarpJacobian>::value_type Jw;
  Jw.reserve(n_reserve);

  std::vector<std::pair<int,int>> inds;
  inds.reserve(n_reserve);

  for(int y = 2; y < image.rows - 3; ++y) {
    for(int x = 2; x < image.cols - 3; ++x) {
      float d = D(y*ds, x*ds);
      if(d > 0.01f && G(y,x) > 1e-3) {
        if(!do_nonmax_supp || is_local_max(y, x, gmag_ptr)) {
          Point xyz;
          xyz[2] = Bf * (1.0f / d);
          xyz[0] = (x - cx) * xyz[2] / fx;
          xyz[1] = (y - cy) * xyz[2] / fy;

          _points.push_back(xyz);
          Jw.push_back(MakeJacobian(xyz, fx, fy));
          inds.push_back(std::make_pair(y,x));
        }
      }
    }
  }

  auto n = inds.size();
  _pixels.resize(8 * n);
  _jacobians.resize(8 * n);

  typedef Eigen::Matrix<float,1,2> ImageGradient;

#pragma omp parallel for
  for(int b = 0; b < 8; ++b) {
    auto pixels_ptr = _pixels.data() + b*n;
    auto jacobians_ptr = _jacobians.data() + b*n;
    const cv::Mat_<float>& B = (const cv::Mat_<float>&) bitplanes.I[b];
    const cv::Mat_<float>& IxIy = (const cv::Mat_<float>&) bitplanes.G[b];
    for(size_t i = 0; i < n; ++i) {
      int r = inds[i].first,
          c = inds[i].second;
      pixels_ptr[i] = B(r,c);
      auto g = IxIy.at<cv::Vec2f>(r,c);
      jacobians_ptr[i] = ImageGradient(g[0], g[1]) * Jw[i];
    }
  }
}

} // bpvo

