#include "bpvo/vo.h"
#include "bpvo/template_data.h"

#include <opencv2/core/core.hpp>

#include <memory>
#include <cmath>

namespace bpvo {

static int getNumberOfPyramidLevels(int min_image_dim, int min_allowed_res)
{
  return 1 + std::round(std::log2(min_image_dim / (double) min_allowed_res));
}

template <typename T> static inline
cv::Mat ToOpenCV(const T* data, int rows, int cols)
{
  return cv::Mat(rows, cols, data, cv::DataType<T>::type);
}

struct VisualOdometry::Impl
{
  /** size of the image at the highest resolution */
  ImageSize _image_size;

  /** parameters */
  AlgorithmParameters _params;

  /** pyramid of template data */
  std::vector<UniquePointer<TemplateData>> _template_data_pyr;

  /**
   * \param K the intrinsic matrix at the highest resolution
   * \param b the stereo baseline at the highest resolution
   * \param params AlgorithmParameters
   */
  Impl(const Matrix33& KK, float b, ImageSize image_size, AlgorithmParameters params)
      : _image_size(image_size), _params(params)
  {
    assert( _image_size.rows > 0 && _image_size.cols > 0 );

    // auto decide the number of levels of params.numPyramidLevels <= 0
    int num_levels = params.numPyramidLevels <= 0 ?
        getNumberOfPyramidLevels(std::min(_image_size.rows, _image_size.cols), 40) :
        params.numPyramidLevels;

    AlgorithmParameters params_low_res(params);

    // relatex the tolerance for low pyramid levels (for speed)
    if(params.relaxTolerancesForCoarseLevels) {
      params_low_res.parameterTolerance *= 10;
      params_low_res.functionTolerance *= 10;
      params_low_res.gradientTolerance *= 10;
      params_low_res.maxIterations = 42;
    }

    // create the data per pyramid level
    Matrix33 K(KK);
    for(int i = 0; i < num_levels; ++i) {
      auto d = make_unique<TemplateData>(i != 0 ? params_low_res : params, K, b, i);
      _template_data_pyr.push_back(std::move(d));
      K *= 0.5; K(2,2) = 1.0; // K is cut by half
      b *= 2.0; // b *2, this was pose does not need rescaling across levels
    }
  }

  /**
   * \param I pointer to the image
   * \param D poitner to the disparity
   */
  Result addFrame(const uint8_t*, const float*);
}; // Impl

VisualOdometry::VisualOdometry(const Matrix33& K, float baseline,
                               ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(K, baseline, image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  return _impl->addFrame(image, disparity);
}

}; // bpvo

