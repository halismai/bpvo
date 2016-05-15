#include "bpvo/central_difference_descriptor.h"
#include "bpvo/math_utils.h"
#include "bpvo/utils.h"
#include "bpvo/imgproc.h"
#include "bpvo/parallel.h"

namespace bpvo {

CentralDifferenceDescriptor::
CentralDifferenceDescriptor(int radius, float sigma_before, float sigma_after)
  : DenseDescriptor()
    , _radius(radius)
    , _sigma_before(sigma_before)
    , _sigma_after(sigma_after)
    , _rows(0)
    , _cols(0)
    , _channels(math::sq(2*radius+1) - 1)
{
  THROW_ERROR_IF(_radius <= 0, "invalid radius");
}

CentralDifferenceDescriptor::
CentralDifferenceDescriptor(const CentralDifferenceDescriptor& other)
    : DenseDescriptor(other),
    _radius(other._radius),
    _sigma_before(other._sigma_before),
    _sigma_after(other._sigma_after),
    _rows(other._rows),
    _cols(other._cols),
    _channels(other._channels) {}

CentralDifferenceDescriptor::~CentralDifferenceDescriptor() {}


class CentralDifferenceDescriptorBody : public ParallelForBody
{
 public:
  CentralDifferenceDescriptorBody(const cv::Mat& src, float sigma,
                                   const std::vector<cv::Point2i>& offsets,
                                   std::vector<cv::Mat>& channels)
      : _src(src), _sigma(sigma), _offset(offsets), _channels(channels)
  {
    THROW_ERROR_IF(_channels.size() != _offset.size(),
                   "number of channels mismatches number of offsets");
  }

  virtual ~CentralDifferenceDescriptorBody() {}

  inline void operator()(const Range& range) const
      //__attribute__((optimize("unroll-loops")))
  {
    for(int i = range.begin(); i != range.end(); ++i) {
      _channels[i].create(_src.size(), cv::DataType<float>::type);
      int y_off = _offset[i].y;
      int x_off = _offset[i].x;

      for(int y = 0; y < _src.rows; ++y) {
        int y_i = std::min(std::max(y + y_off, 0), _src.rows - 1);
        auto* srow_shift = _src.ptr<uint8_t>(y_i);
        auto* srow = _src.ptr<uint8_t>(y);
        auto* drow = _channels[i].ptr<float>(y);

#if defined(WITH_OPENMP)
#pragma omp simd aligned(srow, drow : 16)
#endif
        for(int x = 0; x < _src.cols; ++x) {
          int x_i = std::min(std::max(x + x_off, 0), _src.cols - 1);
          drow[x] = (float) srow[x] - (float) srow_shift[x_i];
        }
      }

      if(_sigma > 0.0f)
        imsmooth(_channels[i], _channels[i], _sigma);
    }
  }

 private:
  const cv::Mat& _src;
  float _sigma;
  const std::vector<cv::Point2i>& _offset;
  std::vector<cv::Mat>& _channels;
}; // CentralDifferenceDescriptorBody

void CentralDifferenceDescriptor::compute(const cv::Mat& image)
{
  _rows = image.rows;
  _cols = image.cols;
  cv::Mat I = _sigma_before > 0.0f ? imsmooth(image, _sigma_before) : image;

  cv::Point2i off;
  std::vector<cv::Point2i> neighbord_offsets;
  for(int r = -_radius; r <= _radius; ++r) {
    for(int c = -_radius; c <= _radius; ++c) {
      if( !(r == 0 && c == 0) )
        neighbord_offsets.push_back( cv::Point2i(c, r) );
    }
  }

  CentralDifferenceDescriptorBody func(I, _sigma_after, neighbord_offsets, _channels);
  parallel_for(Range(0, _channels.size()), func);
}

}; // bpvo
