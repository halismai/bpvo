#ifndef BPVO_DMV_PATCH_H
#define BPVO_DMV_PATCH_H

#include <bpvo/utils.h>
#include <bpvo/debug.h>
#include <dmv/descriptor.h>

#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

template <typename TSrc, typename TDst> inline
void extractPatch(const TSrc* ptr, int stride, int row, int col, int radius, TDst* dst)
{
  for(int r = -radius, i = 0; r <= radius; ++r)
  {
    auto* p = ptr + (row+r)*stride;
    for(int c = -radius; c <= radius; ++c, ++i)
      dst[i] = static_cast<TDst>( p[col + c] );
  }
}

template <size_t R> static inline constexpr size_t GetPatchLength()
{
  return (R*2 + 1) * (R*2 + 1);
}

template <int Radius, typename TSrc, typename TDst> inline
bool extractPatch(const TSrc* ptr, int stride, int rows, int cols, const ImagePoint& p, TDst* dst,
                  bool replicate_border = true)
{
  static_assert( Radius > 0, "Radius must be > 0" );

  int y = static_cast<int>( p.y() ),
      x = static_cast<int>( p.x() );

  bool ret = true;

  // we'll ignore patches that fall outside the image for now
  if(y < Radius || y > rows - Radius - 1 || x < Radius || x > cols - Radius - 1)
    ret = false;

  if(ret || replicate_border) {
    int max_rows = rows - Radius - 1,
        max_cols = cols - Radius - 1;

    for(int r = -Radius, i = 0; r <= Radius; ++r)
    {
      int row = std::min(max_rows, std::max((int) Radius, y + r));
      auto* p = ptr + row*stride;

      for(int c = -Radius; c <= Radius; ++c, ++i)
      {
        int col = std::min(max_cols, std::max((int) Radius, x + c));

        dst[i] = static_cast<TDst>( *(p + col) );
      }
    }
  } else {
    memset(dst, 0, GetPatchLength<Radius>() * sizeof(TDst));
  }

  return ret;
}

template <int N, int Mul> inline constexpr int RoundUpTo()
{
  return Mul ? (( N % Mul ) ? N + Mul - (N % Mul) : N) : N;
}

template <size_t Radius>
class Patch : public DescriptorBase< Patch<Radius> >
{
  typedef Patch<Radius> Self;
  typedef DescriptorBase<Self> BaseType;
  typedef typename BaseType::DataType DataType;

 public:
  static constexpr int Dimension = (2*Radius + 1) * (2*Radius + 1);

  inline Patch() {}
  inline ~Patch() {}

  inline void set(const cv::Mat& I, const ImagePoint& p)
  {
    int stride = I.step / I.elemSize1();
    if(!extractPatch<Radius>(I.ptr(), stride, I.rows, I.cols, p, _data))
    {
      Warn("point is outside image bounds %d,%d\n", (int) p.y(), (int) p.x());
    }

    // set the padding to zero
    for(size_t i = GetPatchLength<Radius>(); i < sizeof(_data)/sizeof(DataType); ++i)
      _data[i] = DataType(0);
  }

  inline const DataType* data() const { return _data; }

 protected:
  alignas(16) DataType _data[ RoundUpTo<Radius,16>() ];
}; // Patch

typedef Patch<1> Patch3x3;



/*
class Patch3x3 : public DescriptorBase<Patch3x3>
{
  typedef DescriptorBase<Patch3x3> BaseType;
  typedef typename BaseType::DataType DataType;

 public:
  static constexpr int Dimension = 9;

  inline Patch3x3() {}
  inline ~Patch3x3() {}

  void set(const cv::Mat& I, const ImagePoint&);
  inline const DataType* data() const { return _data; }

 protected:
  alignas(16) DataType _data[16]; // padded to 16
}; // Patch3x3
*/


} // dmv
} // bpvo

#endif // BPVO_DMV_PATCH_H

