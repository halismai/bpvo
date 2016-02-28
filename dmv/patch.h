#ifndef BPVO_DMV_PATCH_H
#define BPVO_DMV_PATCH_H

#include <dmv/descriptor.h>

namespace cv {
class Mat;
} // cv

namespace bpvo {
namespace dmv {

template <typename TSrc, typename TDst> inline
void extractPatch(const TSrc* ptr, int stride, int row, int col, int radius, TDst* dst)
{
  for(int r = -radius; r <= radius; ++r)
  {
    auto* p = ptr + (row+r)*stride;
    for(int c = -radius; c <= radius; ++c)
      *dst++ = p[col + c];
  }
}

class Patch3x3 : public DescriptorBase<Patch3x3>
{
  typedef DescriptorBase<Patch3x3> BaseType;
  typedef typename BaseType::DataType DataType;

 public:
  static constexpr int Dimension = 9;

  inline Patch3x3() {}
  inline ~Patch3x3() {}

 protected:

  void set(const cv::Mat& I, const ImagePoint&);
  inline const DataType* data() const { return _data; }

  alignas(16) DataType _data[16]; // padded to 16
}; // Patch3x3

} // dmv
} // bpvo

#endif // BPVO_DMV_PATCH_H

