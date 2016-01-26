#include "bpvo/template_data.h"
#include "bpvo/bitplanes.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"
#include "bpvo/partial_warp.h"
#include <opencv2/core.hpp>

#include <utility>
#include <iostream>

#define TEMPLATE_DATA_EXTRACT_SERIAL 0

#if TEMPLATE_DATA_EXTRACT_SERIAL
#define TBB_PREVIEW_SERIAL_SUBSET 1
#endif

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace bpvo {

/**
 * Wrapper to process input images. Responsible for computing the bitplanes as
 * well interpolation, etc
 */
struct TemplateData::InputData
{
  /**
   * \param s1 sigma to be applied before computing the CT
   * \param s2 sigma to be applied after extracting bits
   */
  InputData(float s1, float s2):
      _sigma_ct(s1), _sigma_bp(s2) {}

  /**
   * sets the input image
   */
  inline void setImage(const cv::Mat& image)
  {
    _rows = image.rows;
    _cols = image.cols;
    _data = computeBitPlanes(image, _sigma_ct, _sigma_bp);
  }

  /**
   * \return true if a point is within the image bounds (minus an offset for
   * linear interpolation)
   */
  template <class PointType> inline
  bool isValid(const PointType& p) const {
    return p[0] >= 0 && p[0] < _cols - 1 && p[1] >= 0 && p[1] < _rows - 1;
  }

  BitPlanesData _data;          //< pre-computed dense bit-planes
  float _sigma_ct, _sigma_bp;   //< bit-planes parameters
  int _rows = 0, _cols = 0;     //< rows and cols of the input image
}; // TemplateData::InputData


struct DataExtractor
{
  /**
   * first element is the index in the image, which will convert back to row and
   * column coordinates. The last element is the disparity from the disparity map
   */
  typedef std::vector<std::pair<int,float>> IndicesAndDisparity;

  typedef Eigen::Matrix<float,2,6> WarpJacobian;
  typedef Eigen::Matrix<float,1,2> ImageGradient;

  /**
   */
  DataExtractor(TemplateData& data, const IndicesAndDisparity& inds)
      : _data(data), _stride(data._input_data->_data.gradientAbsMag.cols), _inds(inds)
  {
    _data.resize(_inds.size());

    float fx = _data._K(0,0), fy = _data._K(1,1),
          cx = _data._K(0,2), cy = _data._K(1,2);
    float Bf = _data._baseline * fx;

    // set the points
    for(size_t i = 0; i < _inds.size(); ++i) {
      int x=0, y=0;
      ind2sub(_stride, _inds[i].first, y, x);

      _data._points[i].z() = Bf * (1.0 / _inds[i].second);
      _data._points[i].x() = (x - cx) * _data._points[i].z() / fx;
      _data._points[i].y() = (y - cy) * _data._points[i].z() / fy;
      _data._points[i].w() = 1.0f;
    }

    assert( _data.numPoints() == (int) _inds.size() );
  }

  FORCE_INLINE void operator()(const tbb::blocked_range<int>& range) const
  {
    auto fx = _data._K(0,0), fy = _data._K(1,1);

    const auto& bitplanes = _data._input_data->_data;
    for(int j = range.begin(); j != range.end(); ++j) {
      // the j-th channel
      const float* B_ptr = bitplanes.cn[j].ptr<const float>();

      // pointer to jacobians and pixels for channel 'j'
      int offset = j * _data.numPoints();
      auto J = _data._jacobians.data() + offset;
      auto P = _data._pixels.data() + offset;

      for(size_t i = 0; i < _inds.size(); ++i) {
        int ii = _inds[i].first;

        P[i] = B_ptr[ii];

        ImageGradient IxIy;
        IxIy[0] = 0.5f * fx * ( B_ptr[ii+1] - B_ptr[ii-1] );
        IxIy[1] = 0.5f * fy * ( B_ptr[ii+_stride] - B_ptr[ii-_stride] );

        const auto& pt = _data.X(i);
        auto x = pt.x(), y = pt.y(), z = pt.z(),
             xx = x*x, yy = y*y, xy = x*y, zz = z*z;

        // TODO inline or try pre-computing this as well
        J[i] = IxIy * (
            WarpJacobian() <<
            -xy/zz, (1.0f + xx/zz), -y/z, 1.0f/z, 0.0f, -x/zz,
            -(1.0f + yy/zz), xy/zz, x/z, 0.0f, 1.0/z, -y/zz).finished();
      }
    }
  }

  TemplateData& _data;
  int _stride;

  const IndicesAndDisparity& _inds;
};

TemplateData::TemplateData(const AlgorithmParameters& p, const Matrix33& K,
                           const float& baseline, int pyr_level)
    : _K(K), _baseline(baseline), _pyr_level(pyr_level)
    , _input_data(make_unique<InputData>(p.sigmaPriorToCensusTransform, p.sigmaBitPlanes)) {}

TemplateData::~TemplateData() {}

void TemplateData::reserve(size_t n)
{
  _jacobians.reserve(8*n);
  _points.reserve(n);
  _pixels.reserve(8*n);
}

void TemplateData::resize(size_t npts)
{
  _points.resize(npts);

  _jacobians.resize(8 * npts);
  _pixels.resize(8 * npts);
}

void TemplateData::clear()
{
  _jacobians.clear();
  _points.clear();
  _pixels.clear();
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
template <class T>
struct IsLocalMax
{
  inline IsLocalMax(const T* ptr, int stride, int radius)
      : _ptr(ptr), _stride(stride), _radius(radius)
  {
    assert( _radius > 0 && _stride > 0);
  }

  inline void setStride(int s) { _stride = s; }
  inline void setRadius(int r) { _radius = r; }

  /**
   * \return true if element at location (row,col) is a strict local maxima
   * within the specified radius
   */
  FORCE_INLINE
  bool operator()(int row, int col) const
  {
    auto v = _ptr[row*_stride + col];
    for(int r = -_radius; r <= _radius; ++r)
      for(int c = -_radius; c <= _radius; ++c)
        if(!(!r && !c) && _ptr[(row+r)*_stride + col + c] >= v)
          return false;

    return true;
  }

 private:
  const T* _ptr;
  int _stride, _radius;
}; // IsLocalMax


/**
 * allows to subsample the disparities using a pyramid level
 */
struct DisparityPyramidLevel
{
  DisparityPyramidLevel(const cv::Mat& D, int pyr_level)
      : _D_ptr(D.ptr<float>()), _stride(D.cols), _scale(1 << pyr_level) {}

  FORCE_INLINE float operator()(int r, int c) const
  {
    return *(_D_ptr + index(r,c));
  }

  FORCE_INLINE int index(int r, int c) const {
    return (r*_scale) * _stride + c*_scale;
  }

  const float* _D_ptr;
  int _stride;
  int _scale;
}; // DisparityPyramidLevel

/**
 * \return {pixel_index, disparity}
 */
static inline std::vector< std::pair<int,float> >
getValidPixelsLocations(const DisparityPyramidLevel& dmap, const cv::Mat_<float>& gmag,
                        int nms_radius, bool do_nonmax_supp)
{
  std::vector<std::pair<int,float>> inds;
  inds.reserve( 0.5 * gmag.rows * gmag.cols );
  auto gmag_ptr = gmag.ptr<const float>();
  const IsLocalMax<float> is_local_max(gmag_ptr, gmag.cols, nms_radius);

  static constexpr int Border = 2;

  for(int y = Border; y < gmag.rows - Border - 1; ++y)
    for(int x = Border; x < gmag.cols - Border - 1; ++x) {
      float d = dmap(y,x);
      int ii = y*gmag.cols + x;;
      if(d > 0.01f && gmag_ptr[ii] > 0.0f) {
        if(!do_nonmax_supp || is_local_max(y, x)) {
          inds.push_back( {ii, d} );
        }
      }
    }

  return inds;
}


void TemplateData::compute(const cv::Mat& image, const cv::Mat& disparity)
{
  assert( disparity.type() == cv::DataType<float>::type );
  assert( image.type() == cv::DataType<uint8_t>::type );

  _input_data->setImage(image);

  auto& bitplanes = _input_data->_data;
  bitplanes.computeGradientAbsMag();

  auto num_pixels = image.rows * image.cols;
  auto do_nonmax_supp = num_pixels >= AlgorithmParameters::MIN_NUM_FOR_PIXEL_PSELECTION;

  // we'll do this in two passes, first find out the good locations, then
  // extract the data. This way we can pre-allocate the right amount of memory
  // and improve cache locality
  int nms_radius = 1;
  auto inds = getValidPixelsLocations(DisparityPyramidLevel(disparity, _pyr_level),
                                      bitplanes.gradientAbsMag, nms_radius, do_nonmax_supp);

  DataExtractor de(*this, inds);

  tbb::blocked_range<int> range(0, 8);
#if TEMPLATE_DATA_EXTRACT_SERIAL
  tbb::serial::parallel_for(range, de);
#else
  tbb::parallel_for(range, de);
#endif

}

static FORCE_INLINE int Floor(double v)
{
  int i = static_cast<int>(v);
  return i - (i > v);
}

void TemplateData::setInputImage(const cv::Mat& image)
{
  _input_data->setImage(image);
}


struct ResidualComputer
{
  typedef EigenAlignedContainer<Eigen::Vector4f>::value_type XyCoeffVector;
  typedef EigenAlignedContainer<Eigen::Vector2i>::value_type UvVector;

  /**
   * \param bitplanes the bitplanes input
   */
  ResidualComputer(const BitPlanesData& bitplanes,
                   const std::vector<float>& pixels,
                   const XyCoeffVector& xy_coeff,
                   const UvVector& uv,
                   const std::vector<uint8_t>& valid,
                   std::vector<float>& residuals)
      : _bitplanes(bitplanes), _pixels(pixels), _xy_coeff(xy_coeff), _uv(uv)
      , _valid(valid), _residuals(residuals), _stride(_bitplanes.cn.front().cols)
  {
    assert( _residuals.size() == 8*_valid.size() );
    assert( _residuals.size() == _pixels.size() );
  }

  FORCE_INLINE void operator()(const tbb::blocked_range<int>& range) const
  {
    auto n_pts = _valid.size();
    auto residuals_ptr = _residuals.data() + n_pts*range.begin();
    auto I0_ptr = _pixels.data() + n_pts*range.begin();

    using Eigen::Vector4f;

    for(int c = range.begin(); c != range.end(); ++c, residuals_ptr += n_pts, I0_ptr += n_pts) {

      auto I1_ptr = _bitplanes.cn[c].ptr<const float>();
      for(size_t i = 0; i < n_pts; ++i) {
        if(_valid[i]) {
          auto ii = _uv[i].y()*_stride + _uv[i].x();
          auto Iw = _xy_coeff[i].dot(Vector4f(I1_ptr[ii], I1_ptr[ii+1],
                                              I1_ptr[ii+_stride], I1_ptr[ii+_stride+1]));
          residuals_ptr[i] = Iw - I0_ptr[i];
        } else {
          residuals_ptr[i] = 0.0f;
        }
      }
    }
  }

  const BitPlanesData& _bitplanes;
  const std::vector<float>& _pixels;
  const XyCoeffVector& _xy_coeff;
  const UvVector& _uv;
  const std::vector<uint8_t>& _valid;
  std::vector<float>& _residuals;

  int _stride = 0;
}; // ResidualComputer

void TemplateData::computeResiduals(const Matrix44& pose, std::vector<float>& residuals,
                                    std::vector<uint8_t>& valid) const
{
  const auto& bitplanes = _input_data->_data;
  int max_rows = bitplanes.cn.front().rows - 1,
      max_cols = bitplanes.cn.front().cols - 1;

  auto n_pts = _points.size();

  typename EigenAlignedContainer<Eigen::Vector4f>::value_type xy_coeff(n_pts);
  typename EigenAlignedContainer<Eigen::Vector2i>::value_type uv(n_pts);

  valid.resize(n_pts);

  const Matrix34 KT = _K * pose.block<3,4>(0,0);

#if 1
  computeInterpolationData(KT, _points, max_rows + 1, max_cols + 1, xy_coeff, uv, valid);
#else


  for(size_t i = 0; i < n_pts; ++i) {
    Eigen::Vector3f x = KT * _points[i];
    x.head<2>() /= x[2];

    int xi = Floor(x[0]);
    int yi = Floor(x[1]);

    valid[i] = xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows;
    uv[i] = Eigen::Vector2i(xi, yi);
    // NOTE: we do not need to compute the coefficients for invalid points, but
    // it is more efficient to eliminate the branch
    float xf = x[0] - xi;
    float yf = x[1] - yi;
    xy_coeff[i] = Eigen::Vector4f(
        (1.0f - yf) * (1.0f - xf),
        (1.0f - yf) * xf,
        yf * (1.0f - xf),
        yf * xf);
  }
#endif

  residuals.resize(_pixels.size());
  ResidualComputer rc(bitplanes, _pixels, xy_coeff, uv, valid, residuals);

  tbb::blocked_range<int> range(0, bitplanes.cn.size());
  tbb::parallel_for(range, rc);
}

} // bpvo

#if TEMPLATE_DATA_EXTRACT_SERIAL
#undef TBB_PREVIEW_SERIAL_SUBSET
#endif

#undef TEMPLATE_DATA_EXTRACT_SERIAL

