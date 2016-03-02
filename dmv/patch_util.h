#ifndef BPVO_DMV_PATCH_UTIL_H
#define BPVO_DMV_PATCH_UTIL_H

#include <cstddef>

namespace bpvo {
namespace dmv {

template <size_t N> static inline constexpr size_t Square() { return N*N; }

template <size_t R> static inline constexpr size_t GetPatchLength()
{
  return Square<2*R + 1>();
}

static inline size_t GetPatchLength(size_t r)
{
  return (2*r + 1) * (2*r + 1);
}

/**
 * does not check if the (y,x) are within the image bounds
 */
template <typename TSrc, typename TDst> inline
void extractPatch(const TSrc* ptr, size_t stride, int y, int x, int radius, TDst* dst)
{
  for(int r = -radius; r <= radius; ++r)
    for(int c = -radius; c <= radius; ++c)
      *dst++ = static_cast<TDst>( *(ptr + (r + y)*stride + (c + x)) );
}

static inline int clip_(int v, int min_val, int max_val)
{
  return (v < min_val ? min_val : v > max_val ? max_val : v);
}

/**
 * clips the border
 */
template <typename TSrc, typename TDst> inline
void extractPatch(const TSrc* ptr, size_t stride, int rows, int cols, int y, int x, int radius, TDst* dst)
{
  for(int r = -radius; r <= radius; ++r)
  {
    auto p = ptr + stride*clip_(r + y, 0, rows-1);
    for(int c = -radius; c <= radius; ++c)
      *dst++ = static_cast<TDst>( *(p + clip_(c + x, 0, cols-1)) );
  }
}

};  // dmv
};  // bpvo

#endif // BPVO_DMV_PATCH_UTIL_H
