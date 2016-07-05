#include "bpvo/photo_error.h"
#include "bpvo/imwarp.h"
#include "bpvo/project_points.h"
#include "bpvo/simd.h"
#include "bpvo/eigen.h"
#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>
#include <vector>

namespace bpvo {

//
// doing the interpolation with opencv's optimization is really fast. But, the
// reduction in accuracy due to the use of fixed floating point affects
// performance severely. We recommend not enabling this options here
//
#define PHOTO_ERROR_WITH_OPENCV 0
#define PHOTO_ERROR_OPT         0 // optimized version

#if PHOTO_ERROR_WITH_OPENCV
#include <opencv2/imgproc/imgproc.hpp>

struct PhotoError::Impl
{
  typedef Eigen::Map<const Point, Eigen::Aligned> PointMap;

  void init(const Matrix34& P, const PointVector& X, ValidVector& valid, int rows, int cols)
  {
    int N = X.size();
    resize(N);
    valid.resize(N);

    if(N > 0)
    {
      Eigen::Vector3f p;
      for(int i = 0; i < N; ++i)
      {
        p = P * X[i];
        float w_i = 1.f / p[2];
        _x[i] = w_i * p[0];
        _y[i] = w_i * p[1];

        int xi = static_cast<int>(_x[i]),
            yi = static_cast<int>(_y[i]);

        valid[i] = xi>=0 && xi<cols-1 && yi>=0 && yi<rows-1;
      }

      cv::convertMaps(_x, _y, _map1, _map2, CV_16SC2);
    }
  }

  void run(const float* I0_ptr, const cv::Mat_<float>& I1, float* r_ptr)
  {
    if(_x.empty())
      return;

    cv::Mat dst;
    cv::remap(I1, dst, _map1, _map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    auto I1_ptr = dst.ptr<float>();
    for(size_t i = 0; i < _x.size(); ++i)
      r_ptr[i] = I1_ptr[i] - I0_ptr[i];
  }

  void resize(size_t n)
  {
    _x.resize(n);
    _y.resize(n);
  }

  std::vector<float> _x;
  std::vector<float> _y;
  cv::Mat _map1, _map2;
}; // PhotoError

#elif PHOTO_ERROR_OPT

struct PhotoError::Impl
{
  typedef Eigen::Matrix<float,4,1> Vector4;
  typedef typename EigenAlignedContainer<Vector4>::type CoeffsVector;

  Impl(InterpolationType interp_type)
      : _interp_type(interp_type)
  {
    THROW_ERROR_IF( _interp_type != InterpolationType::kLinear ||
                    _interp_type != InterpolationType::kCubic,
                    "Unkown interpolation type" );
  }

  void init(const Matrix34& P, const PointVector& X, ValidVector& valid, int rows, int cols)
  {
    resize(X.size());
    valid.resize(X.size());
    _valid_ptr = valid.data();
    _stride = cols;

    if(X.empty())
      return;

    switch( _interp_type )
    {
      case InterpolationType::kLinear:
        {
          projectPoints(P, X[0].data(), X.size(), ImageSize(rows, cols),
                        valid.data(), _inds.data(), _interp_coeffs[0].data());
        } break;
      case InterpolationType::kCubic:
        {
          //computeInterpCoeffsCubic(P, X, ImageSize(rows, cols), valid.data(),
           //                        _inds.size(), _interp_coeffs[0].data());
        } break;
    }

  }

  void run(const float* I0_ptr, const float* I1_ptr, float* r_ptr) const
  {
    int num_points = _interp_coeffs.size();
    if(num_points == 0)
      return;

    int i = 0;

    // TODO clean up this loop
    if(simd::isAligned<DefaultAlignment>(r_ptr) && simd::isAligned<DefaultAlignment>(I0_ptr))
    {
      for(i = 0; i <= num_points - 8; i += 8)
      {
#if defined(__AVX__)
        _mm256_store_ps(r_ptr + i,
                        _mm256_sub_ps(_mm256_setr_ps(
                                this->operator()(I1_ptr, i + 0),
                                this->operator()(I1_ptr, i + 1),
                                this->operator()(I1_ptr, i + 2),
                                this->operator()(I1_ptr, i + 3),
                                this->operator()(I1_ptr, i + 4),
                                this->operator()(I1_ptr, i + 5),
                                this->operator()(I1_ptr, i + 6),
                                this->operator()(I1_ptr, i + 7)), _mm256_load_ps(I0_ptr + i)));
#else
        _mm_store_ps(r_ptr + i,
                     _mm_sub_ps(_mm_setr_ps(
                             this->operator()(I1_ptr, i + 0),
                             this->operator()(I1_ptr, i + 1),
                             this->operator()(I1_ptr, i + 2),
                             this->operator()(I1_ptr, i + 3)), _mm_load_ps(I0_ptr + i)));
        _mm_store_ps(r_ptr + i + 4,
                     _mm_sub_ps(_mm_setr_ps(
                             this->operator()(I1_ptr, i + 4),
                             this->operator()(I1_ptr, i + 5),
                             this->operator()(I1_ptr, i + 6),
                             this->operator()(I1_ptr, i + 7)), _mm_load_ps(I0_ptr + i + 4)));
#endif
      }
    } else {
      for(i = 0; i <= num_points - 8; i += 8)
      {
#if defined(__AVX__)
        _mm256_storeu_ps(r_ptr + i,
                        _mm256_sub_ps(_mm256_setr_ps(
                                this->operator()(I1_ptr, i + 0),
                                this->operator()(I1_ptr, i + 1),
                                this->operator()(I1_ptr, i + 2),
                                this->operator()(I1_ptr, i + 3),
                                this->operator()(I1_ptr, i + 4),
                                this->operator()(I1_ptr, i + 5),
                                this->operator()(I1_ptr, i + 6),
                                this->operator()(I1_ptr, i + 7)), _mm256_loadu_ps(I0_ptr + i)));
#else
        _mm_storeu_ps(r_ptr + i,
                      _mm_sub_ps(_mm_setr_ps(
                              this->operator()(I1_ptr, i + 0),
                              this->operator()(I1_ptr, i + 1),
                              this->operator()(I1_ptr, i + 2),
                              this->operator()(I1_ptr, i + 3)), _mm_loadu_ps(I0_ptr + i)));
        _mm_storeu_ps(r_ptr + i + 4,
                      _mm_sub_ps(_mm_setr_ps(
                              this->operator()(I1_ptr, i + 4),
                              this->operator()(I1_ptr, i + 5),
                              this->operator()(I1_ptr, i + 6),
                              this->operator()(I1_ptr, i + 7)), _mm_loadu_ps(I0_ptr + i + 4)));
#endif
      }
    }


    for( ; i < num_points; ++i)
      r_ptr[i] = this->operator()(I1_ptr, i) - I0_ptr[i];

#if defined(__AVX__)
    _mm256_zeroupper();
#endif

  }

  void resize(size_t N)
  {
    _interp_coeffs.resize(N);
    _inds.resize(N);
  }

  inline Vector4 load_data(const float* ptr, int i) const
  {
    auto p = ptr + _inds[i];
    return Vector4(*p, *(p + 1), *(p + _stride), *(p + _stride + 1));
  }

  inline float operator()(const float* ptr, int i) const
  {
#if defined(WITH_SIMD)
    return _valid_ptr[i] ? simd::dot(simd::load<true>(_interp_coeffs[i].data()),
                                     load_data_simd(ptr + _inds[i]) ) : 0.0f;
#else
    return _valid_ptr[i] ? dot_(_interp_coeffs[i], load_data(ptr, i)) : 0.0f;
#endif
  }

#if defined(WITH_SIMD)
  inline __m128 load_data_simd(const float* p) const
  {
    //
    // NOTE this might segfault for points near the boundary, we should adjust
    // the border/valid mask when selecting points
    //
    return _mm_shuffle_ps(simd::load<false>(p), simd::load<false>(p +  _stride),
                          _MM_SHUFFLE(1,0,1,0));
  }
#endif

  inline float dot_(const Vector4& a, const Vector4& b) const
  {
#if defined(__SSE4_1__)
    return simd::dot(_mm_load_ps(a.data()), _mm_load_ps(b.data()));
#else
    // EIGEN uses 2 applications of hadd after mul, dp seems faster if we have sse4
    return a.dot(b);
#endif
  }

  int _stride;
  CoeffsVector _interp_coeffs;
  std::vector<int> _inds;
  const typename ValidVector::value_type* _valid_ptr = nullptr;

  InterpolationType _interp_type;
}; // PhotoError::Impl
#else


// Standard versionC:w

inline int Floor(float v)
{
  int i = static_cast<int>(v);
  return i - (i > v);
}

inline int Floor(double v)
{
  int i = static_cast<int>(v);
  return i - (i > v);
}

template <typename T>
static inline void interpolateCubic(T x, T* coeffs )
{
  assert_is_floating_point<T>();

  // from opencv, but A is changed to -0.5
  const T A = T(-0.5);

  coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
  coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
  coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
  coeffs[3] = T(1) - coeffs[0] - coeffs[1] - coeffs[2];
}

template <typename T>
static inline void interpolateCosine(T x, T* coeffs)
{
  assert_is_floating_point<T>();

  auto m = (T(1) - std::cos(x*M_PI)) / 2.0;
  coeffs[0] = T(1) - m;
  coeffs[1] = m;
}

template <typename T>
static inline void interpolateCubicHermite(T x, T* coeffs, const T* y,
                                           T bias = 0, T tension = 0)
{
  assert_is_floating_point<T>();

  auto mu2 = x*x;
  auto mu3 = mu2*x;

  auto m0 = (y[1] - y[0]) * (1 + bias) * (1 - tension) / 2.0;
  m0 += (y[2] - y[1]) * (1 - bias) * (1 - tension) / 2.0;

  auto m1 = (y[2] - y[1])*(1 + bias) * (1 - tension) / 2.0;
  m1 += (y[3] - y[2])*(1 - bias) * (1 - tension) / 2.0;

  coeffs[0] = 2*mu3 - 3*mu2 + T(1);
  coeffs[1] = mu3 - 2*mu2 + x;
  coeffs[2] = mu3 - mu2;
  coeffs[3] = -2*mu3 + 3*mu2;
}

template <typename T>
static inline T interpolateCubicHermite(const T* y, T mu, T bias = 0, T tension = 0)
{
  // based on http://paulbourke.net/miscellaneous/interpolation/
  auto mu2 = mu*mu;
  auto mu3 = mu*mu2;

  T m0, m1;
  T a0, a1, a2, a3;

  m0 = ((y[1] - y[0]) * (1 + bias) * ( 1 - tension ) / 2.0) +
       ((y[2] - y[1]) * (1 - bias) * ( 1 - tension ) / 2.0);

  m1 = ((y[2] - y[1]) * (1 + bias) * ( 1 - tension ) / 2.0) +
       ((y[3] - y[2]) * (1 - bias) * ( 1 - tension ) / 2.0);

  a0 = 2*mu3 - 3*mu2 + 1;
  a1 = mu3 - 2*mu2 + mu;
  a2 = mu3 - mu2;
  a3 = -2*mu3 + 3*mu2;

  return a0*y[1] + a1*m0 + a2*m1 + a3*y[2];
}

struct PhotoError::Impl
{
  typedef Eigen::Matrix<double,2,1> Point2;
  typedef typename EigenAlignedContainer<Point2>::type Point2Vector;

  Impl(InterpolationType t)
      : _interp_type(t) {}

  inline void init(const Matrix34& P_, const PointVector& X, ValidVector& valid,
                   int rows, int cols)
  {
    int border_lo = (_interp_type == kLinear || _interp_type == kCosine) ? 0 : 1;
    int border_hi = (_interp_type == kLinear || _interp_type == kCosine) ? 1 : 3;

    _x.resize(X.size());
    valid.resize(X.size());
    const Eigen::Matrix<double,3,4> P = P_.cast<double>();
    for(size_t i = 0; i < X.size(); ++i)
    {
      _x[i] = normHomog( (P * X[i].cast<double>()) );
      int xi = Floor(_x[i].x());
      int yi = Floor(_x[i].y());
      valid[i] = xi >= border_lo && xi < cols-border_hi && yi >= border_lo && yi < rows-1;
    }

    _valid_ptr = valid.data();
    _stride = cols;
  }

  inline void run(const float* I0_ptr, const float* I1_ptr, float* r_ptr) const
  {
    for(size_t i = 0; i < _x.size(); ++i)
    {
      if(_valid_ptr[i])
      {
        double xf = _x[i].x();
        double yf = _x[i].y();

        int xi = Floor(xf);
        int yi = Floor(yf);

        xf -= (double) xi;
        yf -= (double) yi;

        switch(_interp_type)
        {
          case kLinear:
            {
              int ii = yi*_stride + xi;
              double wx = (1.0 - xf);
              double Iw = (1.0 - yf) * (I1_ptr[ii        ]*wx + I1_ptr[ii+1]*xf) +
                  yf  * (I1_ptr[ii+_stride]*wx + I1_ptr[ii+_stride+1]*xf);
              r_ptr[i] = float( Iw - (double) I0_ptr[i] );
            } break;

          case kCosine:
            {
              Eigen::Matrix<float,2,1> Cx, Cy;
              typedef Eigen::Map<const Eigen::Matrix<float,2,1>> MapType;

              auto* p1 = I1_ptr + (yi + 0)*_stride + xi;
              auto* p2 = I1_ptr + (yi + 1)*_stride + xi;

              interpolateCosine((float) xf, Cx.data());
              interpolateCosine((float) yf, Cy.data());

              float Iw = Cy.dot(Eigen::Matrix<float,2,1>(
                      MapType(p1).dot(Cx),
                      MapType(p2).dot(Cx)));
              r_ptr[i] = Iw - I0_ptr[i];
            } break;

          case kCubic:
            {
              Eigen::Matrix<float,4,1> Cx, Cy;
              typedef Eigen::Map<const Eigen::Matrix<float,4,1>> MapType;

              interpolateCubic((float) xf, Cx.data());
              interpolateCubic((float) yf, Cy.data());

              auto* p1 = I1_ptr + (yi - 1)*_stride + xi;
              auto* p2 = I1_ptr + (yi + 0)*_stride + xi;
              auto* p3 = I1_ptr + (yi + 1)*_stride + xi;
              auto* p4 = I1_ptr + (yi + 2)*_stride + xi;

              float Iw = Cy.dot(Eigen::Matrix<float,4,1>(
                      MapType(p1).dot(Cx),
                      MapType(p2).dot(Cx),
                      MapType(p3).dot(Cx),
                      MapType(p4).dot(Cx)));
              r_ptr[i] = Iw - I0_ptr[i];
            } break;

          case kCubicHermite:
            {
              auto* p1 = I1_ptr + (yi - 1)*_stride + xi;
              auto* p2 = I1_ptr + (yi + 0)*_stride + xi;
              auto* p3 = I1_ptr + (yi + 1)*_stride + xi;
              auto* p4 = I1_ptr + (yi + 2)*_stride + xi;

              Eigen::Matrix<float, 4, 1> V;
              V[0] = interpolateCubicHermite(p1, (float) xf),
              V[1] = interpolateCubicHermite(p2, (float) xf);
              V[2] = interpolateCubicHermite(p3, (float) xf);
              V[3] = interpolateCubicHermite(p4, (float) xf);

              float Iw = interpolateCubicHermite(V.data(), (float) yf);
              r_ptr[i] = Iw - I0_ptr[i];
            } break;
        }
      } else
      {
        r_ptr[i] = 0.0f;
      }
    }
  }

 protected:
  int _stride;
  const typename ValidVector::value_type* _valid_ptr = NULL;
  Point2Vector _x;

  InterpolationType _interp_type;
}; // PhotoError::Impl

#endif

PhotoError::PhotoError(InterpolationType t)
  : _impl(new PhotoError::Impl(t)) {}

  PhotoError::~PhotoError() {}

void PhotoError::init(const Matrix34& P, const PointVector& X, ValidVector& valid, int rows, int cols)
{
  _impl->init(P, X, valid, rows, cols);
}

void PhotoError::run(const float* I0_ptr, const float* I1_ptr, float* r_ptr) const
{
  _impl->run(I0_ptr, I1_ptr, r_ptr);
}

#undef PHOTO_ERROR_WITH_OPENCV
#undef PHOTO_ERROR_OPT

}; // bpvo

