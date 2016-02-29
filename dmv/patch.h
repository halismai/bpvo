#ifndef BPVO_DMV_PATCH_H
#define BPVO_DMV_PATCH_H

#include <bpvo/utils.h>
#include <bpvo/debug.h>
#include <dmv/descriptor.h>
#include <dmv/patch_util.h>

#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

template <size_t R>
class Patch : public DescriptorBase< Patch<R> >
{
  static_assert(R > 0, "Patch radius must be positive");

  typedef Patch<R> Self;
  typedef DescriptorBase<Self> BaseType;
  typedef typename BaseType::DataType DataType;

 public:
  static constexpr size_t Radius = R;
  static constexpr int Length = GetPatchLength<Radius>();
  static constexpr size_t NumBytes = RoundUpTo<Length, 16>();
  static constexpr int Dimension = Length;

 public:

  inline Patch() {}
  inline ~Patch() {}

  inline void set(const cv::Mat& I, const ImagePoint& p)
  {
    assert( I.type() == cv::DataType<uint8_t>::type && "image must be uint8_t");

    extractPatch(I.ptr<uint8_t>(), I.step/I.elemSize1(), I.rows, I.cols,
                 p.y(), p.x(), Radius, _data);

    for(size_t i = Length; i < sizeof(_data) / sizeof(DataType); ++i)
      _data[i] = DataType(0); // zero the padding
  }

  inline const DataType* data() const { return _data; }

 protected:
  alignas(16) DataType _data[NumBytes];
}; // Patch

typedef Patch<1> Patch3x3;
typedef Patch<2> Patch4x4;


} // dmv
} // bpvo

#endif // BPVO_DMV_PATCH_H

