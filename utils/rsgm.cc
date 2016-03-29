// This stuff is from Rober Spangenberg
// Has been modified slightly by halismai to work on my system and compile with
// the BPVO code base


// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

// The license txt is re-producted here
/*
 License Information

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
The authors allow the downloaders of this package to use and modify the source code for their own research. Any commercial application, redistribution, etc. has to be arranged between users and authors individually.

Please use the software provided in this package at your own risk. The executable is provided only for the purpose of evaluation of the algorithm presented in the paper "Large Scale Semi-Global Matching on the CPU" (IV 2014). The authors of the paper cannot be held responsible for any damages resulting from use of this software. There are no warranties associated with the code or executables.
*/

#include "utils/rsgm.h"
#include "bpvo/utils.h"

#if !defined(WITH_GPL_CODE)
struct RSGM::Impl {};

RSGM::RSGM(Config conf) : _config(conf), _impl(bpvo::make_unique<Impl>()) {}
RSGM::~RSGM() {}

void RSGM::compute(const cv::Mat&, const cv::Mat&, cv::Mat&)
{
  THROW_ERROR("compile WITH_GPL_CODE\n");
}

#else
#include "bpvo/debug.h"

#include <opencv2/core/core.hpp>

#include <list>

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <cmath>

#include <algorithm>

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include <nmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>

typedef float float32;
typedef double float64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int8_t sint8;
typedef int16_t sint16;
typedef int32_t sint32;

static constexpr sint32 SINT32_MAX = std::numeric_limits<sint32>::max();
static constexpr sint16 SINT16_MAX = std::numeric_limits<sint16>::max();

#define ALIGN16 __attribute__((aligned(16)))
#define ALIGN32 __attribute__((aligned(32)))
#define FORCEINLINE inline __attribute__((always_inline))

#ifdef MAX
    #undef MAX
#endif

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#ifndef  MIN
    #define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define ROUND(x)    ( ((x) >= 0.0f) ? ((sint32)((x) + 0.5f)) : ((sint32)((x) - 0.5f)) )

#define SIGN(u,v)   ( ((v)>=0.0) ? ABS(u) : -ABS(u) )
#define ABS(x) std::abs((x))

// saturate casts
template<typename _Tp> static inline _Tp saturate_cast(uint8 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(sint8 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint16 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(sint16 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(uint32 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(sint32 v) { return _Tp(v); }
template<typename _Tp> static inline _Tp saturate_cast(float32 v) { return _Tp(v); }

template<> inline uint16 saturate_cast<uint16>(sint8 v)
{
    return (uint16)std::max((int)v, 0);
}
template<> inline uint16 saturate_cast<uint16>(sint16 v)
{
    return (uint16)std::max((int)v, 0);
}
template<> inline uint16 saturate_cast<uint16>(sint32 v)
{
    return (uint16)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0);
}
template<> inline uint16 saturate_cast<uint16>(uint32 v)
{
    return (uint16)std::min(v, (unsigned)USHRT_MAX);
}

template<> inline uint16 saturate_cast<uint16>(float v)
{
    int iv = ROUND(v); return saturate_cast<uint16>(iv);
}



/* hamming costs and population counts */
uint16 m_popcount16LUT[UINT16_MAX+1];

// unsigned to 32bit
inline uint16 hamDist32(const uint32& x, const uint32& y)
{
    uint16 dist = 0, val = (uint16)(x ^ y);

    // Count the number of set bits
    while(val)
    {
        ++dist;
        val &= val - 1;
    }

    return dist;
}

inline void fillPopCount16LUT()
{
  // popCount LUT
    for (int i=0; i < UINT16_MAX+1; i++) {
      m_popcount16LUT[i] = hamDist32(i,0);
    }
}

#define HW_POPCNT
#ifdef HW_POPCNT
    #define POPCOUNT32 _mm_popcnt_u32
    #if defined(_M_X64) || defined(__amd64__) || defined(__amd64)
        #define POPCOUNT64 (uint16)_mm_popcnt_u64
    #else
        #define POPCOUNT64 popcount64LUT
    #endif
#else
    #define POPCOUNT32 popcount32
    #define POPCOUNT64 popcount64LUT
#endif

inline uint16 popcount32(const uint32& i)
{
    return (m_popcount16LUT[i&0xFFFF] + m_popcount16LUT[i>>16]);
}

// pop count for 4 32bit values
FORCEINLINE __m128i popcount32_4(const __m128i& a)
{
    __m128i b = _mm_srli_epi16(a,4); // psrlw       $4, %%xmm1

    ALIGN16 const uint32 _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
    const __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);

    const __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F); //_mm_set1_epi8(0xf);

    __m128i a2 = _mm_and_si128(a, xmm6); // pand    %%xmm6, %%xmm0  ; xmm0 - lower nibbles
    b = _mm_and_si128(b, xmm6); // pand    %%xmm6, %%xmm1  ; xmm1 - higher nibbles

    __m128i popA = _mm_shuffle_epi8(xmm7,a2); // pshufb  %%xmm0, %%xmm2  ; xmm2 = vector of popcount for lower nibbles
    __m128i popB = _mm_shuffle_epi8(xmm7,b); //  pshufb  %%xmm1, %%xmm3  ; xmm3 = vector of popcount for higher nibbles

    __m128i popByte = _mm_add_epi8(popA, popB); // paddb   %%xmm3, %%xmm2  ; xmm2 += xmm3 -- vector of popcount for bytes;

    // How to get to added quadwords?
    const __m128i ZERO = _mm_setzero_si128();

    // with horizontal adds

    __m128i upper = _mm_unpackhi_epi8(popByte, ZERO);
    __m128i lower = _mm_unpacklo_epi8(popByte, ZERO);
    __m128i popUInt16 = _mm_hadd_epi16(lower,upper); // uint16 pop count
    __m128i popUInt32 = _mm_hadd_epi16(popUInt16,ZERO); // uint32 pop count

    return popUInt32;
}

// pop count for 4 32bit values
FORCEINLINE __m128i popcount32_4(const __m128i& a, const __m128i& lut,const __m128i& mask)
{

    __m128i b = _mm_srli_epi16(a,4); // psrlw       $4, %%xmm1

    __m128i a2 = _mm_and_si128(a, mask); // pand    %%xmm6, %%xmm0  ; xmm0 - lower nibbles
    b = _mm_and_si128(b, mask); // pand    %%xmm6, %%xmm1  ; xmm1 - higher nibbles

    __m128i popA = _mm_shuffle_epi8(lut,a2); // pshufb  %%xmm0, %%xmm2  ; xmm2 = vector of popcount for lower nibbles
    __m128i popB = _mm_shuffle_epi8(lut,b); //  pshufb  %%xmm1, %%xmm3  ; xmm3 = vector of popcount for higher nibbles

    __m128i popByte = _mm_add_epi8(popA, popB); // paddb   %%xmm3, %%xmm2  ; xmm2 += xmm3 -- vector of popcount for bytes;

    // How to get to added quadwords?
    const __m128i ZERO = _mm_setzero_si128();

    // Version 1 - with horizontal adds

    __m128i upper = _mm_unpackhi_epi8(popByte, ZERO);
    __m128i lower = _mm_unpacklo_epi8(popByte, ZERO);
    __m128i popUInt16 = _mm_hadd_epi16(lower,upper); // uint16 pop count
    // the lower four 16 bit values contain the uint32 pop count
    __m128i popUInt32 = _mm_hadd_epi16(popUInt16,ZERO);

    return popUInt32;
}


//uint16 hamDist64(uint64 x, uint64 y);

constexpr uint64 m1  = 0x5555555555555555; //binary: 0101...
constexpr uint64 m2  = 0x3333333333333333; //binary: 00110011..
constexpr uint64 m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
constexpr uint64 m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
constexpr uint64 m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
constexpr uint64 m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
constexpr uint64 hff = 0xffffffffffffffff; //binary: all ones
constexpr uint64 h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...

inline uint16 popcount64(uint64 x) {
    x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
    return (x * h01)>>56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

inline uint16 popcount64LUT(const uint64& i)
{
    return (m_popcount16LUT[i&0xFFFF] + m_popcount16LUT[(i>>16) & 0xFFFF]
    + m_popcount16LUT[(i>>32) & 0xFFFF]) + m_popcount16LUT[i>>48];
}

inline uint16* getDispAddr_xyd(uint16* dsi, sint32 width, sint32 disp, sint32 i, sint32 j, sint32 k)
{
    return dsi + i*(disp*width) + j*disp + k;
}


void costMeasureCensus5x5Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2,
                                      const int width,const int dispCount, const uint16 invalidDispValue,
                                      uint16* dsi, const int lineStart,const int lineEnd)
{
    ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
    const __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
    const __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

    for (sint32 i=lineStart;i < lineEnd;i++) {
        uint32* pBase = intermediate1+i*width;
        uint32* pMatchRow = intermediate2+i*width;
        for (uint32 j=0; j < (uint32)width; j++) {
            uint32* pBaseJ = pBase + j;
            uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

            int d=dispCount - 1;

            for (; d >(sint32)j && d >= 0;d--) {
                *getDispAddr_xyd(dsi, width, dispCount, i, j, d) = invalidDispValue;
                pMatchRowJmD++;
            }

            int dShift4m1 = ((d-1) >> 2)*4;
            int diff = d - dShift4m1;
            // rest
            if (diff != 0) {
                for (; diff >= 0 && d >= 0;d--,diff--) {
                    uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                    *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = cost;
                    pMatchRowJmD++;
                }
            }

            // 4 costs at once
            __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);
            d -= 3;

            uint16* baseAddr = getDispAddr_xyd(dsi,width, dispCount, i,j,0);
            for (; d >= 0;d-=4) {
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
                _mm_storel_pi((__m64*)(baseAddr+d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
                pMatchRowJmD+=4;
            }

        }
    }
}

void costMeasureCensus5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2
    , const int height,const int width, const int dispCount, const uint16 invalidDispValue, uint16* dsi,
    sint32 numThreads)
{
    // first 2 lines are empty
    for (int i=0;i<2;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCount-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = invalidDispValue;
            }
        }
    }

    if (numThreads != 4)
    {
#pragma omp parallel num_threads(2)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dsi, 2, height/2);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height/2, height-2);
                }
            }
        }
    } else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dsi, 2, height/4);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height/4, height/2);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height/2, height-height/4);
                }
#pragma omp section
                {
                    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dsi, height-height/4, height-2);
                }
            }
        }
    }
    /* last 2 lines are empty*/
    for (int i=height-2;i<height;i++) {
        for (int j=0; j < width; j++) {
            for (int d=0; d <= dispCount-1;d++) {
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = invalidDispValue;
            }
        }
    }
}

void costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2,const int width,
    const int dispCount, const uint16 invalidDispValue, const sint32 dispCountCompressed, const sint32 dispSubSample, const sint32 dispCountLow,
    uint16* dsi, const sint32 lineStart,const sint32 lineEnd)
{
  for (sint32 i=lineStart;i < lineEnd;i++) {
    uint32* pBase = intermediate1+i*width;
    uint32* pMatchRow = intermediate2+i*width;
    for (uint32 j=0; j < (uint32)width; j++) {
      uint32* pBaseJ = pBase + j;
      uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

      sint32 d=dispCount - 1;

      for (; d >(sint32)j && d >= 0 && d>=dispCountLow; d--) {
        sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = invalidDispValue;
        pMatchRowJmD++;
      }

      for (; d >(sint32)j && d >= 0; d--) {
        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
        pMatchRowJmD++;
      }

      // fill valid disparities

      if (d > dispCountLow - 1) {
        // align to sub-sampled disparities

        // get multiple of sub-sampling
        sint32 shift = (d-dispCountLow+1) % 8;

        // rest
        if (shift > 0) {
          sint32 modRes = d % dispSubSample;
          d -= modRes;
          pMatchRowJmD+= modRes;
          shift = (shift+1) / 2;

          // advance through sub-sampled disparities
          for (; shift > 0 && d >= dispCountLow;d-= dispSubSample, shift-=1) {
            uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
            sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
            *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = cost;
            pMatchRowJmD += dispSubSample;
          }
          d++;
          pMatchRowJmD -= 1;
        }
      } else {
        // align to full-sampled disparities
        sint32 dShift4m1 = ((d-1) >> 2)*4;
        sint32 diff = d - dShift4m1;
        // rest
        if (diff > 0) {
          // advance through full-sampled disparities
          for (; diff >= 0 && d >= 0;d--,diff--) {
            uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
            *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = cost;
            pMatchRowJmD++;
          }
        }
      }

      // 4 costs at once
      ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
      __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
      __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

      __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);


      // sub-sampled disparities
      for (; d >= dispCountLow;d-=8) {
        // flip the values
        __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), _MM_SHUFFLE(1,3,1,3)); //mask = 00 01 10 11
        __m128i rPoint4n = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+4)), _MM_SHUFFLE(1,3,1,3)); //mask = 00 01 10 11
        const sint32 mask = 1<<0 | 1<<1 | 1<<2 | 1<<3;
        __m128i rPoint4Sub = _mm_blend_epi16(rPoint4, rPoint4n, mask); // first and second of first, third and fourth of second
        sint32 compressedD = dispCountLow+(d-7-dispCountLow)/dispSubSample;
        _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4Sub),xmm7,xmm6)));
        pMatchRowJmD+=8;
      }

      // full sampled disparities
      d -=3;
      for (; d >= 0;d-=4) {
        // flip the values
        __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
        _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
        pMatchRowJmD+=4;
      }
    }
  }
}

void costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2,const int width,
                                                          const int dispCount, const uint16 invalidDispValue, const sint32 dispCountCompressed, const sint32 dispSubSample, const sint32 dispCountLow,
                                                          uint16* dsi, const sint32 lineStart,const sint32 lineEnd)
{
  for (sint32 i=lineStart;i < lineEnd;i++) {
    uint32* pBase = intermediate1+i*width;
    uint32* pMatchRow = intermediate2+i*width;
    for (uint32 j=0; j < (uint32)width; j++) {
      uint32* pBaseJ = pBase + j;
      uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

      sint32 d=dispCount - 1;

      for (; d >(sint32)j && d >= 0 && d>=dispCountLow; d--) {
        sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = invalidDispValue;
        pMatchRowJmD++;
      }

      for (; d >(sint32)j && d >= 0; d--) {
        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
        pMatchRowJmD++;
      }

      // fill valid disparities

      if (d > dispCountLow - 1) {
        // align to sub-sampled disparities

        // get multiple of sub-sampling times 4
        sint32 shift = (d-dispCountLow+1) % 16;

        // rest
        if (shift > 0) {
          sint32 modRes = d % dispSubSample;
          d -= modRes;
          pMatchRowJmD+= modRes;
          shift -= modRes;

          // advance through sub-sampled disparities
          for (; shift > 0 && d >= dispCountLow;d-= dispSubSample, shift-=4) {
            uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
            sint32 compressedD = dispCountLow+(d-dispCountLow)/dispSubSample;
            *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD) = cost;
            pMatchRowJmD += dispSubSample;
          }
          d+=3;
          pMatchRowJmD -= 3;
        }
      } else {
        // align to full-sampled disparities
        sint32 dShift4m1 = ((d-1) >> 2)*4;
        sint32 diff = d - dShift4m1;
        // rest
        if (diff > 0) {
          // advance through full-sampled disparities
          for (; diff >= 0 && d >= 0;d--,diff--) {
            uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
            *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = cost;
            pMatchRowJmD++;
          }
        }
      }

      // 4 costs at once
      ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
      __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
      __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

      __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);


      // sub-sampled disparities
      bool entered = false;
      for (; d >= dispCountLow;d-=16) {
        // flip the values
        __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), _MM_SHUFFLE(3,1,2,0)); //mask = 00 01 10 11
        __m128i rPoint4n = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+4)), _MM_SHUFFLE(2,3,1,0)); //mask = 00 01 10 11
        __m128i rPoint4n2 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+8)), _MM_SHUFFLE(1,2,3,0)); //mask = 00 01 10 11
        __m128i rPoint4n3 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)(pMatchRowJmD+12)), _MM_SHUFFLE(0,1,2,3)); //mask = 00 01 10 11

        const sint32 mask = 1<<0 | 1<<1 | 1<<4 | 1<<5;
        // first of first, second of second, third of third, fourth of fourth
        __m128i rPoint4Sub1 = _mm_blend_epi16(rPoint4n2, rPoint4n3, mask);
        __m128i rPoint4Sub2 = _mm_blend_epi16(rPoint4, rPoint4n, mask);
        const sint32 mask2 = 1<<0 | 1<<1 | 1<<2 | 1<<3;
        __m128i rPoint4Sub = _mm_blend_epi16(rPoint4Sub2, rPoint4Sub1, mask2);

        sint32 compressedD = dispCountLow+(d-15-dispCountLow)/dispSubSample;
        _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,compressedD), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4Sub),xmm7,xmm6)));
        pMatchRowJmD+=16;
        entered = true;
      }
      if (entered) {
        d = 63;
      }
      // full sampled disparities
      d -=3;
      for (; d >= 0;d-=4) {
        // flip the values
        __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
        _mm_storel_pi((__m64*)getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
        pMatchRowJmD+=4;
      }
    }
  }
}

void costMeasureCensusCompressed5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2
                                            , sint32 height, sint32 width, sint32 dispCount, const uint16 invalidDispValue, sint32 dispSubSample, uint16* dsi, sint32 numThreads)
{
  sint32 dispCountCompressed=64;
  sint32 dispCountLow = 64;

  if (dispCount > 64) {
    if (dispCount == 128) {
      dispCountCompressed = 64+(dispCount-64)/dispSubSample; // sample every dispSubSample disparity
      dispCountLow = 64;
    } else {
      assert(false);
    }
  } else {
    dispCountCompressed = 64;
    dispCountLow = 64;
  }

  // first 2 lines are empty
  for (int i=0;i<2;i++) {
    for (int j=0; j < width; j++) {
      for (int d=0; d <= dispCountCompressed-1;d++) {
        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
      }
    }
  }

  if (dispSubSample == 2) {

    if (numThreads != 4)
    {
#pragma omp parallel num_threads(2)
      {
#pragma omp sections nowait
        {
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/2);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-2);
          }
        }
      }
    } else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
      {
#pragma omp sections nowait
        {
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/4);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/4, height/2);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-height/4);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample2Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height-height/4, height-2);
          }
        }
      }
    }
  } else if (dispSubSample == 4) {
    if (numThreads != 4)
    {
#pragma omp parallel num_threads(2)
      {
#pragma omp sections nowait
        {
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/2);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-2);
          }
        }
      }
    } else if (numThreads == 4) {
#pragma omp parallel num_threads(4)
      {
#pragma omp sections nowait
        {
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, 2, height/4);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/4, height/2);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height/2, height-height/4);
          }
#pragma omp section
          {
            costMeasureCensusCompressed5x5Subsample4Line_xyd_SSE(intermediate1, intermediate2,width, dispCount, invalidDispValue, dispCountCompressed, dispSubSample, dispCountLow, dsi, height-height/4, height-2);
          }
        }
      }
    }

  }
  /* last 2 lines are empty*/
  for (int i=height-2;i<height;i++) {
    for (int j=0; j < width; j++) {
      for (int d=0; d <= dispCountCompressed-1;d++) {
        *getDispAddr_xyd(dsi,width, dispCountCompressed, i,j,d) = invalidDispValue;
      }
    }
  }
}

void costMeasureCensus9x7Line_xyd(uint64* intermediate1, uint64* intermediate2,int width, int dispCount, uint16* dsi, int startLine, int endLine)
{
  // content
  for (int i=startLine;i < endLine;i++) {
    uint64* baseRow = intermediate1+ width*i;
    uint64* matchRow = intermediate2+width*i;

    for (int j=0;j < width;j++) {
      uint64* pBaseJ = baseRow + j;
      uint64* pMatchRowJmD = matchRow + j - dispCount +1;

      sint32 d = dispCount -1;
      for (; d >(sint32)j && d >= 0;d--) {
        *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = 255;
        pMatchRowJmD++;
      }
      while (d >= 0) {
        uint16 cost = POPCOUNT64(*pBaseJ ^ *pMatchRowJmD);
        pMatchRowJmD++;
        *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = cost;
        d--;
      }
    }
  }
}


void costMeasureCensus9x7_xyd_parallel(uint64* intermediate1, uint64* intermediate2,int height, int width, int dispCount, uint16* dsi,
                                       sint32 numThreads)
{
  // first 3 lines are empty
  for (int i=0;i<3;i++) {
    for (int j=0; j < width; j++) {
      for (int d=0; d <= dispCount-1;d++) {
        *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = 255;
      }
    }
  }

  if (numThreads != 4)
  {
#pragma omp parallel /*shared(dsi,height, width, dispCount, baseImg, matchImg),*/ num_threads(2)
    {
#pragma omp sections nowait
      {
#pragma omp section
        {
          costMeasureCensus9x7Line_xyd(intermediate1, intermediate2, width, dispCount, dsi, 2, height/2);
        }
#pragma omp section
        {
          costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height/2, height-2);
        }
      }
    }
  } else if (numThreads == 4) {
#pragma omp parallel /*shared(dsi,height, width, dispCount, baseImg, matchImg),*/ num_threads(4)
    {
#pragma omp sections nowait
      {
#pragma omp section
        {
          costMeasureCensus9x7Line_xyd(intermediate1, intermediate2, width, dispCount, dsi, 2, height/4);
        }
#pragma omp section
        {
          costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height/4, height/2);
        }
#pragma omp section
        {
          costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height/2, height-height/4);
        }
#pragma omp section
        {
          costMeasureCensus9x7Line_xyd(intermediate1, intermediate2,width, dispCount, dsi, height-height/4, height-2);
        }
      }
    }
  }
  /* last 3 lines are empty*/
  for (int i=height-3;i<height;i++) {
    for (int j=0; j < width; j++) {
      for (int d=0; d <= dispCount-1;d++) {
        *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = 255;
      }
    }
  }
}

void costMeasureCensusCompressed9x7_xyd(uint64* intermediate1, uint64* intermediate2
                                        , sint32 height, sint32 width, sint32 dispCount, sint32 dispSubSample, uint16* dsi)
{
  sint32 dispCountCompressed = dispCount;
  sint32 dispCountLow = dispCount;
  if (dispCount > 64) {
    if (dispCount == 128) {
      dispCountLow = 64;
      if (dispSubSample == 2) {
        dispCountCompressed = 96;
      }
      else if (dispSubSample == 4) {
        dispCountCompressed = 80;
      }
      else {
        assert(false);
      }

    }
    else {
      assert(false);
    }

  }
  // first 3 lines are empty
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < width; j++) {
      for (int d = 0; d < dispCountCompressed; d++) {
        *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
      }
    }
  }

  // content
  for (int i = 3; i < height - 3; i++) {
    uint64* baseRow = intermediate1 + width*i;
    uint64* matchRow = intermediate2 + width*i;

    // stepSize 1
    for (int d = 0; d < dispCountLow; d++) {
      uint64* pBaseJ = baseRow + d;
      uint64* pMatchRowJmD = matchRow;
      for (int j = 0; j < d; j++) {
        *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
      }
      for (int j = d; j < width; j++) {
        uint16 cost = POPCOUNT64(*pBaseJ ^ *pMatchRowJmD);
        pBaseJ++; pMatchRowJmD++;
        *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = cost;
      }
    }

    // stepSize dispSubSample
    for (int d = dispCountLow; d < dispCountCompressed; d++) {
      sint32 realDisp = dispCountLow + (d - dispCountLow)*dispSubSample;
      uint64* pBaseJ = baseRow + realDisp;
      uint64* pMatchRowJmD = matchRow;
      for (int j = 0; j < realDisp; j++) {
        *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
      }
      for (int j = realDisp; j < width; j++) {
        uint16 cost = POPCOUNT64(*pBaseJ ^ *pMatchRowJmD);
        pBaseJ++; pMatchRowJmD++;
        *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = cost;
      }
    }
  }
  // last 3 lines are empty
  for (int i = height - 3; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int d = 0; d < dispCountCompressed; d++) {
        *getDispAddr_xyd(dsi, width, dispCountCompressed, i, j, d) = 255;
      }
    }
  }
}

void matchWTA_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
  const uint32 factorUniq = (uint32)(1024*uniqueness);
  const sint32 disp = maxDisp+1;

  // find best by WTA
  float32* pDestDisp = dispImg;
  for (sint32 i=0;i < height; i++) {
    for (sint32 j=0;j < width; j++) {
      // WTA on disparity values

      uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
      uint16* pCostBase = pCost;
      uint32 minCost = *pCost;
      uint32 secMinCost = minCost;
      int secBestDisp = 0;

      const uint32 end = MIN(disp-1,j);
      if (end == (uint32)disp-1) {
        uint32 bestDisp = 0;

        for (uint32 loop =0; loop < end;loop+= 8) {
          // load costs
          const __m128i costs = _mm_load_si128((__m128i*)pCost);
          // get minimum for 8 values
          const __m128i b = _mm_minpos_epu16(costs);
          const int minValue = _mm_extract_epi16(b,0);

          if ((uint32)minValue < minCost) {
            minCost = (uint32)minValue;
            bestDisp = _mm_extract_epi16(b,1)+loop;
          }
          pCost+=8;
        }

        // get value of second minimum
        pCost = pCostBase;
        pCost[bestDisp]=65535;

#ifdef USE_AVX2
        __m256i secMinVector = _mm256_set1_epi16(-1);
        const uint16* pCostEnd = pCost+disp;
        for (; pCost < pCostEnd;pCost+= 16) {
          // load costs
          __m256i costs = _mm256_load_si256((__m256i*)pCost);
          // get minimum for 8 values
          secMinVector = _mm256_min_epu16(secMinVector,costs);
        }
        secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector,0)),0);
        uint32 secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
        if (secMinCost2 < secMinCost)
          secMinCost = secMinCost2;
#else
        __m128i secMinVector = _mm_set1_epi16(-1);
        const uint16* pCostEnd = pCost+disp;
        for (; pCost < pCostEnd;pCost+= 8) {
          // load costs
          __m128i costs = _mm_load_si128((__m128i*)pCost);
          // get minimum for 8 values
          secMinVector = _mm_min_epu16(secMinVector,costs);
        }
        secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
#endif
        pCostBase[bestDisp]=(uint16)minCost;

        // assign disparity
        if (1024*minCost <=  secMinCost*factorUniq) {
          *pDestDisp = (float)bestDisp;
        } else {
          bool check = false;
          if (bestDisp < (uint32)maxDisp-1 && pCostBase[bestDisp+1] == secMinCost) {
            check=true;
          }
          if (bestDisp>0 && pCostBase[bestDisp-1] == secMinCost) {
            check=true;
          }
          if (!check) {
            *pDestDisp = -10;
          } else {
            *pDestDisp = (float)bestDisp;
          }
        }

      } else {
        int bestDisp = 0;
        // for start
        for (uint32 k=1; k <= end; k++) {
          pCost += 1;
          const uint16 cost = *pCost;
          if (cost < secMinCost) {
            if (cost < minCost) {
              secMinCost = minCost;
              secBestDisp = bestDisp;
              minCost = cost;
              bestDisp = k;
            } else  {
              secMinCost = cost;
              secBestDisp = k;
            }
          }
        }
        // assign disparity
        if (1024*minCost <=  secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
          *pDestDisp = (float)bestDisp;
        } else {
          *pDestDisp = -10;
        }
      }
      pDestDisp++;
    }
  }
}

FORCEINLINE __m128 rcp_nz_ss(__m128 input) {
  __m128 mask = _mm_cmpeq_ss(_mm_set1_ps(0.0), input);
  __m128 recip = _mm_rcp_ss(input);
  return _mm_andnot_ps(mask, recip);
}

FORCEINLINE void setSubpixelValue(float32* dest, uint32 bestDisp, const sint32& c0,const sint32& c1,const sint32& c2)
{
  __m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(),c2 - c0);
  __m128 left = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c0);
  __m128 right = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c2);
  __m128 lowerMin = _mm_min_ss(left, right);
  __m128 d_offset = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f),lowerMin)));
  __m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(),bestDisp);
  __m128 result = _mm_add_ss(baseDisp, d_offset);
  _mm_store_ss(dest,result);
}

inline void matchWTAAndSubPixel_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
  const uint32 factorUniq = (uint32)(1024*uniqueness);
  const sint32 disp = maxDisp+1;

  // find best by WTA
  float32* pDestDisp = dispImg;
  for (sint32 i=0;i < height; i++) {
    for (sint32 j=0;j < width; j++) {
      // WTA on disparity values

      uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
      uint16* pCostBase = pCost;
      uint32 minCost = *pCost;
      uint32 secMinCost = minCost;
      int secBestDisp = 0;

      const uint32 end = MIN(disp-1,j);
      if (end == (uint32)disp-1) {
        uint32 bestDisp = 0;

        for (uint32 loop =0; loop < end;loop+= 8) {
          // load costs
          const __m128i costs = _mm_load_si128((__m128i*)pCost);
          // get minimum for 8 values
          const __m128i b = _mm_minpos_epu16(costs);
          const int minValue = _mm_extract_epi16(b,0);

          if ((uint32)minValue < minCost) {
            minCost = (uint32)minValue;
            bestDisp = _mm_extract_epi16(b,1)+loop;
          }
          pCost+=8;
        }

        // get value of second minimum
        pCost = pCostBase;
        pCost[bestDisp]=65535;

#ifndef USE_AVX2
        __m128i secMinVector = _mm_set1_epi16(-1);
        const uint16* pCostEnd = pCost+disp;
        for (; pCost < pCostEnd;pCost+= 8) {
          // load costs
          __m128i costs = _mm_load_si128((__m128i*)pCost);
          // get minimum for 8 values
          secMinVector = _mm_min_epu16(secMinVector,costs);
        }
        secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
        pCostBase[bestDisp] = (uint16)minCost;
#else
        __m256i secMinVector = _mm256_set1_epi16(-1);
        const uint16* pCostEnd = pCost + disp;
        for (; pCost < pCostEnd; pCost += 16) {
          // load costs
          __m256i costs = _mm256_load_si256((__m256i*)pCost);
          // get minimum for 8 values
          secMinVector = _mm256_min_epu16(secMinVector, costs);
        }
        secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 0)), 0);
        uint32 secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
        if (secMinCost2 < secMinCost)
          secMinCost = secMinCost2;
        pCostBase[bestDisp] = (uint16)minCost;
#endif

        // assign disparity
        if (1024*minCost <=  secMinCost*factorUniq) {
          *pDestDisp = (float)bestDisp;
        } else {
          bool check = false;
          if (bestDisp < (uint32)maxDisp-1 && pCostBase[bestDisp+1] == secMinCost) {
            check=true;
          }
          if (bestDisp>0 && pCostBase[bestDisp-1] == secMinCost) {
            check=true;
          }
          if (!check) {
            *pDestDisp = -10;
          } else {
            if (0 < bestDisp && bestDisp < (uint32)maxDisp-1) {
              setSubpixelValue(pDestDisp, bestDisp, pCostBase[bestDisp-1],minCost, pCostBase[bestDisp+1]);
            } else {
              *pDestDisp = (float)bestDisp;
            }

          }
        }

      } else {
        int bestDisp = 0;
        // for start
        for (uint32 k=1; k <= end; k++) {
          pCost += 1;
          const uint16 cost = *pCost;
          if (cost < secMinCost) {
            if (cost < minCost) {
              secMinCost = minCost;
              secBestDisp = bestDisp;
              minCost = cost;
              bestDisp = k;
            } else  {
              secMinCost = cost;
              secBestDisp = k;
            }
          }
        }
        // assign disparity
        if (1024*minCost <=  secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) {
          if (0 < bestDisp && bestDisp < maxDisp-1) {
            setSubpixelValue(pDestDisp, bestDisp, pCostBase[bestDisp-1],minCost, pCostBase[bestDisp+1]);
          } else {
            *pDestDisp = (float)bestDisp;
          }
        } else {
          *pDestDisp = -10;
        }
      }
      pDestDisp++;
    }
  }
}

inline void matchWTARight_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
  const uint32 factorUniq = (uint32)(1024*uniqueness);

  const uint32 disp = maxDisp+1;
  //_ASSERT(disp <= 256);
  assert( disp <= 256 );
  ALIGN32 uint16 store[256+32];
  store[15] = UINT16_MAX-1;
  store[disp+16] = UINT16_MAX-1;

  // find best by WTA
  float32* pDestDisp = dispImg;
  for (uint32 i=0;i < (uint32)height; i++) {
    for (uint32 j=0;j < (uint32)width;j++) {
      // WTA on disparity values
      int bestDisp = 0;
      uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
      sint32 minCost = *pCost;
      sint32 secMinCost = minCost;
      int secBestDisp = 0;
      const uint32 maxCurrDisp = MIN(disp-1, width-1-j);

      if (maxCurrDisp == disp-1) {

        // transfer to linear storage, slightly unrolled
        for (uint32 k=0; k <= maxCurrDisp; k+=4) {
          store[k+16]=*pCost;
          store[k+16+1]=pCost[disp+1];
          store[k+16+2]=pCost[2*disp+2];
          store[k+16+3]=pCost[3*disp+3];
          pCost += 4*disp+4;
        }
        // search in there
        uint16* pStore = &store[16];
        const uint16* pStoreEnd = pStore+disp;
        for (; pStore < pStoreEnd; pStore+=8) {
          // load costs
          const __m128i costs = _mm_load_si128((__m128i*)pStore);
          // get minimum for 8 values
          const __m128i b = _mm_minpos_epu16(costs);
          const int minValue = _mm_extract_epi16(b,0);

          if (minValue < minCost) {
            minCost = minValue;
            bestDisp = _mm_extract_epi16(b,1)+(int)(pStore-&store[16]);
          }

        }

        // get value of second minimum
        pStore = &store[16];
        store[16+bestDisp]=65535;
#ifndef USE_AVX2
        __m128i secMinVector = _mm_set1_epi16(-1);
        for (; pStore < pStoreEnd;pStore+= 8) {
          // load costs
          __m128i costs = _mm_load_si128((__m128i*)pStore);
          // get minimum for 8 values
          secMinVector = _mm_min_epu16(secMinVector,costs);
        }
        secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);
#else
        __m256i secMinVector = _mm256_set1_epi16(-1);
        for (; pStore < pStoreEnd; pStore += 16) {
          // load costs
          __m256i costs = _mm256_load_si256((__m256i*)pStore);
          // get minimum for 8 values
          secMinVector = _mm256_min_epu16(secMinVector, costs);
        }
        secMinCost = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 0)), 0);
        int secMinCost2 = _mm_extract_epi16(_mm_minpos_epu16(_mm256_extractf128_si256(secMinVector, 1)), 0);
        if (secMinCost2 < secMinCost)
          secMinCost = secMinCost2;
#endif

        // assign disparity
        if (1024U*minCost <=  secMinCost*factorUniq) {
          *pDestDisp = (float)bestDisp;
        } else {
          bool check = (store[16+bestDisp+1] == secMinCost);
          check = check  | (store[16+bestDisp-1] == secMinCost);
          if (!check) {
            *pDestDisp = -10;
          } else {
            *pDestDisp = (float)bestDisp;
          }
        }
        pDestDisp++;
      }
      else {
        // border case handling
        for (uint32 k=1; k <= maxCurrDisp; k++) {
          pCost += disp+1;
          const sint32 cost = (sint32)*pCost;
          if (cost < secMinCost) {
            if (cost < minCost) {
              secMinCost = minCost;
              secBestDisp = bestDisp;
              minCost = cost;
              bestDisp = k;
            } else {
              secMinCost = cost;
              secBestDisp = k;
            }
          }
        }
        // assign disparity
        if (1024U*minCost <= factorUniq*secMinCost|| abs(bestDisp - secBestDisp) < 2  ) {
          *pDestDisp = (float)bestDisp;
        } else {
          *pDestDisp = -10;
        }
        pDestDisp++;
      }
    }
  }
}

inline void doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold)
{
  float* dispRow = dispImg;
  float* dispCheckRow = dispCheckImg;
  for (sint32 i=0;i < height;i++) {
    for (sint32 j=0;j < width;j++) {
      const float32 baseDisp = dispRow[j];
      if (baseDisp >= 0 && baseDisp <= j) {
        const float matchDisp = dispCheckRow[(int)(j-baseDisp)];

        sint32 diff = (sint32)(baseDisp - matchDisp);
        if (abs(diff) > lrThreshold) {
          dispRow[j] = -10; // occluded or false match
        }
      } else {
        dispRow[j] = -10;
      }
    }
    dispRow += width;
    dispCheckRow += width;
  }
}

inline void doRLCheck(float32* dispRightImg, float32* dispCheckImg, const sint32 width, const sint32 height, const sint32 lrThreshold)
{
  float* dispRow = dispRightImg;
  float* dispCheckRow = dispCheckImg;
  for (sint32 i=0;i < height;i++) {
    for (sint32 j=0;j < width;j++) {
      const float32 baseDisp = dispRow[j];
      if (baseDisp >= 0 && j+baseDisp <= width) {
        const float matchDisp = dispCheckRow[(int)(j+baseDisp)];

        sint32 diff = (sint32)(baseDisp - matchDisp);
        if (abs(diff) > lrThreshold) {
          dispRow[j] = -10; // occluded or false match
        }
      } else {
        dispRow[j] = -10;
      }
    }
    dispRow += width;
    dispCheckRow += width;
  }
}


/*  do a sub pixel refinement by a parabola fit to the winning pixel and its neighbors */
inline void subPixelRefine(float32* dispImg, uint16* dsiImg, const sint32 width, const sint32 height, const sint32 maxDisp, sint32 method)
{
  const sint32 disp_n = maxDisp+1;

  /* equiangular */
  if (method == 0) {

    for (sint32 y = 0; y < height; y++)
    {
      uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
      float* disp = (float*)dispImg+y*width;

      for (sint32 x = 1; x < width-1; x++, cost += disp_n)
      {
        if (disp[x] > 0.0) {

          // Get minimum
          int d_min = (int)disp[x];

          // Compute the equations of the parabolic fit
          uint16* costDmin = cost+d_min;
          sint32 c0 = costDmin[-1], c1 = *costDmin, c2 = costDmin[1];

          __m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(),c2 - c0);
          __m128 left = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c0);
          __m128 right = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c2);
          __m128 lowerMin = _mm_min_ss(left, right);
          __m128 result = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f),lowerMin)));

          __m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(),d_min);
          result = _mm_add_ss(baseDisp, result);
          _mm_store_ss(disp+x,result);

        } else {
          disp[x] = -10;
        }
      }
    }
    /* 1: parabolic */
  } else if (method == 1){
    for (sint32 y = 0; y < height; y++)
    {
      uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
      float32* disp = dispImg+y*width;

      for (sint32 x = 1; x < width-1; x++, cost += disp_n)
      {
        if (disp[x] > 0.0) {

          // Get minimum, but offset by 1 from ends
          int d_min = (int)disp[x];

          // Compute the equations of the parabolic fit
          sint32 c0 = cost[d_min-1], c1 = cost[d_min], c2 = cost[d_min+1];
          sint32 a = c0+c0 - 4 * c1 + c2+c2;
          sint32 b =  (c0 - c2);

          // Solve for minimum, which is a correction term to d_min
          disp[x] = d_min + b /(float32) a;

        } else {
          disp[x] = -10;
        }
      }
    }
  } else {
    //        assert("subpixel interpolation method nonexisting");
  }
}

inline void uncompressDisparities_SSE(float32* dispImg, const sint32 width, const sint32 height, uint32 stepwidth)
{
  float32* disp = dispImg;
  const uint32 size = height*width;
  float32* dispEnd = disp + size;

  __m128 maskAll64 = _mm_set1_ps(64.f);

  if (stepwidth == 2) {
    while( disp < dispEnd)
    {
      __m128 valuesComp = _mm_load_ps(disp);

      __m128 result = _mm_add_ps(valuesComp, _mm_max_ps(_mm_sub_ps(valuesComp, maskAll64),_mm_setzero_ps()));
      _mm_store_ps(disp, result);
      disp+=4;
    }
  } else if (stepwidth == 4) {
    while( disp < dispEnd)
    {
      __m128 valuesComp = _mm_load_ps(disp);

      __m128 result = _mm_add_ps(valuesComp, _mm_mul_ps(_mm_max_ps(_mm_sub_ps(valuesComp, maskAll64),_mm_setzero_ps()), _mm_set1_ps(3.0f)));
      _mm_store_ps(disp, result);
      disp+=4;
    }
  }
}

inline uint8* getPixel8(uint8* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint16* getPixel16(uint16* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint32* getPixel32(uint32* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint64* getPixel64(uint64* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

FORCEINLINE void testpixel(uint8* source, uint32 width, sint32 i, sint32 j,uint64& value, sint32 x, sint32 y)
{
    if (*getPixel8(source, width,j+x,i+y) - *getPixel8(source, width,j-x,i-y)>0) {
        value += 1;
    }
}

FORCEINLINE void testpixel2(uint8* source, uint32 width, sint32 i, sint32 j,uint64& value, sint32 x, sint32 y)
{
    value *= 2;
    uint8 result = *getPixel8(source, width,j+x,i+y) - *getPixel8(source, width,j-x,i-y)>0;
    value += result;
}

inline void census9x7_mode8(uint8* source, uint64* dest, uint32 width, uint32 height)
{
    memset(dest, 0, width*height*sizeof(uint64));

    // HWCS central symmetric with 3 central rows
    const int vert = 3;
    const int hor = 4;
    for (sint32 i=vert; i < (sint32)height-vert; i++) {
        for (sint32 j=hor; j < (sint32)width-hor; j++) {
            uint64 value = 0;
            testpixel(source, width, i,j, value,-4,-3);
            testpixel2(source, width, i,j, value,-3,-3);
            testpixel2(source, width, i,j, value,-2,-3);
            testpixel2(source, width, i,j, value,-1,-3);
            testpixel2(source, width, i,j, value,0,-3 );
            testpixel2(source, width, i,j, value,-4,-2);
            testpixel2(source, width, i,j, value,-3,-2);
            testpixel2(source, width, i,j, value,-2,-2);
            testpixel2(source, width, i,j, value,-1,-2);
            testpixel2(source, width, i,j, value,0,-2 );
            testpixel2(source, width, i,j, value,-4,-1);
            testpixel2(source, width, i,j, value,-3,-1);
            testpixel2(source, width, i,j, value,-2,-1);
            testpixel2(source, width, i,j, value,-1,-1);
            testpixel2(source, width, i,j, value,0,-1 );
            testpixel2(source, width, i,j, value,-4, 0);
            testpixel2(source, width, i,j, value,-3, 0);
            testpixel2(source, width, i,j, value,-2, 0);
            testpixel2(source, width, i,j, value,-1, 0);
            testpixel2(source, width, i,j, value,1,-3 );
            testpixel2(source, width, i,j, value,2,-3 );
            testpixel2(source, width, i,j, value,3,-3 );
            testpixel2(source, width, i,j, value,4,-3 );
            testpixel2(source, width, i,j, value,1,-2 );
            testpixel2(source, width, i,j, value,2,-2 );
            testpixel2(source, width, i,j, value,3,-2 );
            testpixel2(source, width, i,j, value,4,-2 );
            testpixel2(source, width, i,j, value,1,-1 );
            testpixel2(source, width, i,j, value,2,-1 );
            testpixel2(source, width, i,j, value,3,-1 );
            testpixel2(source, width, i,j, value,4,-1 );
            testpixel2(source, width, i,j, value,-4,-1);
            testpixel2(source, width, i,j, value,-3,-1);
            testpixel2(source, width, i,j, value,-2,-1);
            testpixel2(source, width, i,j, value,-1,-1);
            testpixel2(source, width, i,j, value,-4, 0);
            testpixel2(source, width, i,j, value,-3, 0);
            testpixel2(source, width, i,j, value,-2, 0);
            testpixel2(source, width, i,j, value,-1, 0);
            testpixel2(source, width, i,j, value,-4, 1);
            testpixel2(source, width, i,j, value,-3, 1);
            testpixel2(source, width, i,j, value,-2, 1);
            testpixel2(source, width, i,j, value,-1, 1);
            testpixel2(source, width, i,j, value,0,-1 );
            *getPixel64(dest,width,j,i) = (uint64)value;
        }
    }
}

FORCEINLINE void testpixel_16bit(uint16* source, uint32 width, sint32 i, sint32 j, uint64& value, sint32 x, sint32 y)
{
    if (*getPixel16(source, width, j + x, i + y) - *getPixel16(source, width, j - x, i - y) > 0) {
        value += 1;
    }
}

FORCEINLINE void testpixel2_16bit(uint16* source, uint32 width, sint32 i, sint32 j, uint64& value, sint32 x, sint32 y)
{
    value *= 2;
    uint8 result = *getPixel16(source, width, j + x, i + y) - *getPixel16(source, width, j - x, i - y) > 0;
    value += result;
}

void census9x7_mode8_16bit(uint16* source, uint64* dest, uint32 width, uint32 height)
{
    memset(dest, 0, width*height*sizeof(uint64));

    // HWCS central symmetric with 3 central rows
    const int vert = 3;
    const int hor = 4;
    for (sint32 i = vert; i < (sint32)height - vert; i++) {
        for (sint32 j = hor; j < (sint32)width - hor; j++) {
            uint64 value = 0;
            testpixel_16bit(source, width, i, j, value, -4, -3);
            testpixel2_16bit(source, width, i, j, value, -3, -3);
            testpixel2_16bit(source, width, i, j, value, -2, -3);
            testpixel2_16bit(source, width, i, j, value, -1, -3);
            testpixel2_16bit(source, width, i, j, value, 0, -3);
            testpixel2_16bit(source, width, i, j, value, -4, -2);
            testpixel2_16bit(source, width, i, j, value, -3, -2);
            testpixel2_16bit(source, width, i, j, value, -2, -2);
            testpixel2_16bit(source, width, i, j, value, -1, -2);
            testpixel2_16bit(source, width, i, j, value, 0, -2);
            testpixel2_16bit(source, width, i, j, value, -4, -1);
            testpixel2_16bit(source, width, i, j, value, -3, -1);
            testpixel2_16bit(source, width, i, j, value, -2, -1);
            testpixel2_16bit(source, width, i, j, value, -1, -1);
            testpixel2_16bit(source, width, i, j, value, 0, -1);
            testpixel2_16bit(source, width, i, j, value, -4, 0);
            testpixel2_16bit(source, width, i, j, value, -3, 0);
            testpixel2_16bit(source, width, i, j, value, -2, 0);
            testpixel2_16bit(source, width, i, j, value, -1, 0);
            testpixel2_16bit(source, width, i, j, value, 1, -3);
            testpixel2_16bit(source, width, i, j, value, 2, -3);
            testpixel2_16bit(source, width, i, j, value, 3, -3);
            testpixel2_16bit(source, width, i, j, value, 4, -3);
            testpixel2_16bit(source, width, i, j, value, 1, -2);
            testpixel2_16bit(source, width, i, j, value, 2, -2);
            testpixel2_16bit(source, width, i, j, value, 3, -2);
            testpixel2_16bit(source, width, i, j, value, 4, -2);
            testpixel2_16bit(source, width, i, j, value, 1, -1);
            testpixel2_16bit(source, width, i, j, value, 2, -1);
            testpixel2_16bit(source, width, i, j, value, 3, -1);
            testpixel2_16bit(source, width, i, j, value, 4, -1);
            testpixel2_16bit(source, width, i, j, value, -4, -1);
            testpixel2_16bit(source, width, i, j, value, -3, -1);
            testpixel2_16bit(source, width, i, j, value, -2, -1);
            testpixel2_16bit(source, width, i, j, value, -1, -1);
            testpixel2_16bit(source, width, i, j, value, -4, 0);
            testpixel2_16bit(source, width, i, j, value, -3, 0);
            testpixel2_16bit(source, width, i, j, value, -2, 0);
            testpixel2_16bit(source, width, i, j, value, -1, 0);
            testpixel2_16bit(source, width, i, j, value, -4, 1);
            testpixel2_16bit(source, width, i, j, value, -3, 1);
            testpixel2_16bit(source, width, i, j, value, -2, 1);
            testpixel2_16bit(source, width, i, j, value, -1, 1);
            testpixel2_16bit(source, width, i, j, value, 0, -1);
            *getPixel64(dest, width, j, i) = (uint64)value;
        }
    }
}


// census 5x5
// input uint8 image, output uint32 image
void census5x5_SSE(uint8* source, uint32* dest, uint32 width, uint32 height)
{
    uint32* dst = dest;
    const uint8* src = source;

    // input lines 0,1,2
    const uint8* i0 = src;

    // output at first result
    uint32* result = dst + 2*width;
    const uint8* const end_input = src + width*height;

    /* expand mask */
    __m128i expandByte1_First4 =  _mm_set_epi8(0x80u, 0x80u, 0x80u, 0x03u,0x80u, 0x80u, 0x80u, 0x02u,
                                               0x80u, 0x80u, 0x80u, 0x01u,0x80u, 0x80u, 0x80u, 0x00u);

    __m128i expandByte2_First4 = _mm_set_epi8(0x80u, 0x80u, 0x03u, 0x80u,0x80u, 0x80u, 0x02u, 0x80u,
                                              0x80u, 0x80u, 0x01u, 0x80u,0x80u, 0x80u, 0x00u, 0x80u);

    __m128i expandByte3_First4  = _mm_set_epi8(0x80u, 0x03u, 0x80u, 0x80u,0x80u, 0x02u, 0x80u, 0x80u,
                                               0x80u, 0x01u, 0x80u, 0x80u,0x80u, 0x00u, 0x80u, 0x80u);

    /* xor with 0x80, as it is a signed compare */
    __m128i l2_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+2*width )) ,_mm_set1_epi8('\x80'));
    __m128i l3_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+3*width )) ,_mm_set1_epi8('\x80'));
    __m128i l4_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+4*width )) ,_mm_set1_epi8('\x80'));  
    __m128i l1_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+width )) ,_mm_set1_epi8('\x80'));
    __m128i l0_register = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0 )) ,_mm_set1_epi8('\x80'));

    i0 += 16;

    __m128i lastResult = _mm_setzero_si128();

    for( ; i0+4*width < end_input; i0 += 16 ) {

            /* parallel 16 pixel processing */
            const __m128i l0_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0 )) ,_mm_set1_epi8('\x80'));
            const __m128i l1_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+width )) ,_mm_set1_epi8('\x80'));
            const __m128i l2_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+2*width )) ,_mm_set1_epi8('\x80'));
            const __m128i l3_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+3*width )) ,_mm_set1_epi8('\x80'));
            const __m128i l4_register_next = _mm_xor_si128(_mm_stream_load_si128( (__m128i*)( i0+4*width )) ,_mm_set1_epi8('\x80'));

            /*  0  1  2  3  4
                5  6  7  8  9
               10 11  c 12 13
               14 15 16 17 18
               19 20 21 22 23 */
            /* r/h is result, v is pixelvalue */

            /* pixel c */
            __m128i pixelcv = _mm_alignr_epi8(l2_register_next,l2_register, 2);

            /* pixel 0*/ 
            __m128i pixel0h = _mm_cmplt_epi8(l0_register,pixelcv);
            
            /* pixel 1*/ 
            __m128i pixel1v = _mm_alignr_epi8(l0_register_next,l0_register, 1);
            __m128i pixel1h = _mm_cmplt_epi8(pixel1v,pixelcv);

            /* pixel 2 */
            __m128i pixel2v = _mm_alignr_epi8(l0_register_next,l0_register, 2);
            __m128i pixel2h = _mm_cmplt_epi8(pixel2v,pixelcv);

            /* pixel 3 */
            __m128i pixel3v = _mm_alignr_epi8(l0_register_next,l0_register, 3);
            __m128i pixel3h = _mm_cmplt_epi8(pixel3v,pixelcv);

            /* pixel 4 */
            __m128i pixel4v = _mm_alignr_epi8(l0_register_next,l0_register, 4);
            __m128i pixel4h = _mm_cmplt_epi8(pixel4v,pixelcv);

            /** line  **/
            /* pixel 5 */
            __m128i pixel5h = _mm_cmplt_epi8(l1_register,pixelcv);

            /* pixel 6 */
            __m128i pixel6v = _mm_alignr_epi8(l1_register_next,l1_register, 1);
            __m128i pixel6h = _mm_cmplt_epi8(pixel6v,pixelcv);

            /* pixel 7 */
            __m128i pixel7v = _mm_alignr_epi8(l1_register_next,l1_register, 2);
            __m128i pixel7h = _mm_cmplt_epi8(pixel7v,pixelcv);

            /* pixel 8 */
            __m128i pixel8v = _mm_alignr_epi8(l1_register_next,l1_register, 3);
            __m128i pixel8h = _mm_cmplt_epi8(pixel8v,pixelcv);

            /* pixel 9 */
            __m128i pixel9v = _mm_alignr_epi8(l1_register_next,l1_register, 4);
            __m128i pixel9h = _mm_cmplt_epi8(pixel9v,pixelcv);

            /* create bitfield part 1*/
            __m128i resultByte1 = _mm_and_si128(_mm_set1_epi8(128u),pixel0h);
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(64),pixel1h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(32),pixel2h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(16),pixel3h));
            __m128i resultByte1b = _mm_and_si128(_mm_set1_epi8(8),pixel4h);
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(4),pixel5h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(2),pixel6h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(1),pixel7h));
            resultByte1 = _mm_or_si128(resultByte1, resultByte1b);

            /** line **/
            /* pixel 10 */
            __m128i pixel10h = _mm_cmplt_epi8(l2_register,pixelcv);

            /* pixel 11 */
            __m128i pixel11v = _mm_alignr_epi8(l2_register_next,l2_register, 1);
            __m128i pixel11h = _mm_cmplt_epi8(pixel11v,pixelcv);

            /* pixel 12 */
            __m128i pixel12v = _mm_alignr_epi8(l2_register_next,l2_register, 3);
            __m128i pixel12h = _mm_cmplt_epi8(pixel12v,pixelcv);
            
            /* pixel 13 */
            __m128i pixel13v = _mm_alignr_epi8(l2_register_next,l2_register, 4);
            __m128i pixel13h = _mm_cmplt_epi8(pixel13v,pixelcv);

            /* line */
            /* pixel 14 */
            __m128i pixel14h = _mm_cmplt_epi8(l3_register,pixelcv);

            /* pixel 15 */
            __m128i pixel15v = _mm_alignr_epi8(l3_register_next,l3_register, 1);
            __m128i pixel15h = _mm_cmplt_epi8(pixel15v,pixelcv);

            /* pixel 16 */
            __m128i pixel16v = _mm_alignr_epi8(l3_register_next,l3_register, 2);
            __m128i pixel16h = _mm_cmplt_epi8(pixel16v,pixelcv);

            /* pixel 17 */
            __m128i pixel17v = _mm_alignr_epi8(l3_register_next,l3_register, 3);
            __m128i pixel17h = _mm_cmplt_epi8(pixel17v,pixelcv);

            /* pixel 18 */
            __m128i pixel18v = _mm_alignr_epi8(l3_register_next,l3_register, 4);
            __m128i pixel18h = _mm_cmplt_epi8(pixel18v,pixelcv);

            /* create bitfield part 2 */
            __m128i resultByte2 = _mm_and_si128(_mm_set1_epi8(128u),pixel8h);
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(64),pixel9h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(32),pixel10h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(16),pixel11h));
            __m128i resultByte2b = _mm_and_si128(_mm_set1_epi8(8),pixel12h);
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(4),pixel13h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(2),pixel14h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(1),pixel15h));
            resultByte2 = _mm_or_si128(resultByte2, resultByte2b);

            /* line */
            /* pixel 19 */
            __m128i pixel19h = _mm_cmplt_epi8(l4_register,pixelcv);

            /* pixel 20 */
            __m128i pixel20v = _mm_alignr_epi8(l4_register_next,l4_register, 1);
            __m128i pixel20h = _mm_cmplt_epi8(pixel20v,pixelcv);

            /* pixel 21 */
            __m128i pixel21v = _mm_alignr_epi8(l4_register_next,l4_register, 2);
            __m128i pixel21h = _mm_cmplt_epi8(pixel21v,pixelcv);

            /* pixel 22 */
            __m128i pixel22v = _mm_alignr_epi8(l4_register_next,l4_register, 3);
            __m128i pixel22h = _mm_cmplt_epi8(pixel22v,pixelcv);

            /* pixel 23 */
            __m128i pixel23v = _mm_alignr_epi8(l4_register_next,l4_register, 4);
            __m128i pixel23h = _mm_cmplt_epi8(pixel23v,pixelcv);

            /* create bitfield part 3*/

            __m128i resultByte3 = _mm_and_si128(_mm_set1_epi8(128u),pixel16h);
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(64),pixel17h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(32),pixel18h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(16),pixel19h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(8),pixel20h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(4),pixel21h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(2),pixel22h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(1),pixel23h));

            /* blend the first part together */
            __m128i resultPart1_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart1_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart1 = _mm_or_si128(resultPart1_1,resultPart1_2);
            __m128i resultPart1_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart1 = _mm_or_si128(resultPart1,resultPart1_3);
            
            /* rotate result bytes */
            /* replace by _mm_alignr_epi8 */
            resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0,3,2,1));
            resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0,3,2,1));
            resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0,3,2,1));

            /* blend the second part together */
            __m128i resultPart2_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart2_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart2 = _mm_or_si128(resultPart2_1,resultPart2_2);
            __m128i resultPart2_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart2 = _mm_or_si128(resultPart2,resultPart2_3);

            /* rotate result bytes */
            resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0,3,2,1));
            resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0,3,2,1));
            resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0,3,2,1));

            /* blend the third part together */
            __m128i resultPart3_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart3_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart3 = _mm_or_si128(resultPart3_1,resultPart3_2);
            __m128i resultPart3_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart3 = _mm_or_si128(resultPart3,resultPart3_3);

            /* rotate result bytes */
            resultByte1 = _mm_shuffle_epi32(resultByte1, _MM_SHUFFLE(0,3,2,1));
            resultByte2 = _mm_shuffle_epi32(resultByte2, _MM_SHUFFLE(0,3,2,1));
            resultByte3 = _mm_shuffle_epi32(resultByte3, _MM_SHUFFLE(0,3,2,1));

            /* blend the fourth part together */
            __m128i resultPart4_1 = _mm_shuffle_epi8(resultByte1, expandByte1_First4);
            __m128i resultPart4_2 = _mm_shuffle_epi8(resultByte2, expandByte2_First4);
            __m128i resultPart4 = _mm_or_si128(resultPart4_1,resultPart4_2);
            __m128i resultPart4_3 = _mm_shuffle_epi8(resultByte3, expandByte3_First4);
            resultPart4 = _mm_or_si128(resultPart4,resultPart4_3);

            /* shift because of offsets */
            _mm_store_si128((__m128i*)result, _mm_alignr_epi8(resultPart1, lastResult, 8));
            _mm_store_si128((__m128i*)(result+4), _mm_alignr_epi8(resultPart2, resultPart1, 8) );
            _mm_store_si128((__m128i*)(result+8), _mm_alignr_epi8(resultPart3, resultPart2, 8) ); 
            _mm_store_si128((__m128i*)(result+12), _mm_alignr_epi8(resultPart4, resultPart3, 8) ); 

            result += 16;
            lastResult = resultPart4;

            /*load next */
            l0_register = l0_register_next;
            l1_register = l1_register_next;
            l2_register = l2_register_next;
            l3_register = l3_register_next;
            l4_register = l4_register_next;

    }
    /* last pixels */
    {
        int i = height - 3;
        for (sint32 j=width-16+2; j < (sint32)width-2; j++) {
            const int centerValue = *getPixel8(source, width,j,i);
            uint32 value = 0;
            for (sint32 x=-2; x <= 2; x++) {
                for (sint32 y=-2; y <= 2; y++) {
                    if (x!=0 || y!=0) {
                        value *= 2;
                        if (centerValue >  *getPixel8(source, width,j+y,i+x)) {
                            value += 1;
                        }
                    }
                }     
            } 
            *getPixel32(dest,width,j,i) = value;
        }
    }
}

// census 5x5
// input uint16 image, output uint32 image
// 2.56ms/2, 768x480, 9.2 cycles/pixel
void census5x5_16bit_SSE(uint16* source, uint32* dest, uint32 width, uint32 height)
{
    uint32* dst = dest;
    const uint16* src = source;
    
    // memsets just for the upper and lower two lines, not really necessary
    memset(dest, 0, width*2*sizeof(uint32));
    memset(dest+width*(height-2), 0, width*2*sizeof(uint32));

    // input lines 0,1,2
    const uint16* i0 = src;
    const uint16* i1 = src+width;
    const uint16* i2 = src+2*width;
    const uint16* i3 = src+3*width;
    const uint16* i4 = src+4*width;

    // output at first result
    uint32* result = dst + 2*width;
    const uint16* const end_input = src + width*height;

    /* expand mask */
    __m128i expandLowerMask  = _mm_set_epi8(0x06, 0x06, 0x06, 0x06, 0x04, 0x04, 0x04, 0x04,
                                            0x02, 0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00);

    __m128i expandUpperMask = _mm_set_epi8(0x0E, 0x0E, 0x0E, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C,
                                           0x0A, 0x0A, 0x0A, 0x0A, 0x08, 0x08, 0x08, 0x08);

    __m128i blendB1B2Mask = _mm_set_epi8(0x00u, 0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u,
                                         0x00u, 0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u);

    __m128i blendB1B2B3Mask  = _mm_set_epi8(0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u, 0x00u,
                                            0x00u, 0x80u, 0x00u, 0x00u, 0x00u, 0x80u, 0x00u, 0x00u);

    __m128i l2_register = _mm_stream_load_si128( (__m128i*)( i2 ) );
    __m128i l3_register = _mm_stream_load_si128( (__m128i*)( i3 ) );
    __m128i l4_register = _mm_stream_load_si128( (__m128i*)( i4 ) );  
    __m128i l1_register = _mm_stream_load_si128( (__m128i*)( i1 ) );
    __m128i l0_register = _mm_stream_load_si128( (__m128i*)( i0 ) );

    i0 += 8;
    i1 += 8;
    i2 += 8;
    i3 += 8;
    i4 += 8;

    __m128i lastResultLower = _mm_setzero_si128();

    for( ; i4 < end_input; i0 += 8, i1 += 8, i2 += 8, i3+=8, i4+=8 ) {

            /* parallel 16 pixel processing */
            const __m128i l0_register_next = _mm_stream_load_si128( (__m128i*)( i0 ) );
            const __m128i l1_register_next = _mm_stream_load_si128( (__m128i*)( i1 ) );
            const __m128i l2_register_next = _mm_stream_load_si128( (__m128i*)( i2 ) );
            const __m128i l3_register_next = _mm_stream_load_si128( (__m128i*)( i3 ) );
            const __m128i l4_register_next = _mm_stream_load_si128( (__m128i*)( i4 ) );

            /*  0  1  2  3  4
                5  6  7  8  9
               10 11  c 12 13
               14 15 16 17 18
               19 20 21 22 23 */
            /* r/h is result, v is pixelvalue */

            /* pixel c */
            __m128i pixelcv = _mm_alignr_epi8(l2_register_next,l2_register, 4);

            /* pixel 0*/ 
            __m128i pixel0h = _mm_cmplt_epi16(l0_register,pixelcv);
            
            /* pixel 1*/ 
            __m128i pixel1v = _mm_alignr_epi8(l0_register_next,l0_register, 2);
            __m128i pixel1h = _mm_cmplt_epi16(pixel1v,pixelcv);

            /* pixel 2 */
            __m128i pixel2v = _mm_alignr_epi8(l0_register_next,l0_register, 4);
            __m128i pixel2h = _mm_cmplt_epi16(pixel2v,pixelcv);

            /* pixel 3 */
            __m128i pixel3v = _mm_alignr_epi8(l0_register_next,l0_register, 6);
            __m128i pixel3h = _mm_cmplt_epi16(pixel3v,pixelcv);

            /* pixel 4 */
            __m128i pixel4v = _mm_alignr_epi8(l0_register_next,l0_register, 8);
            __m128i pixel4h = _mm_cmplt_epi16(pixel4v,pixelcv);

            /** line  **/
            /* pixel 5 */
            __m128i pixel5h = _mm_cmplt_epi16(l1_register,pixelcv);

            /* pixel 6 */
            __m128i pixel6v = _mm_alignr_epi8(l1_register_next,l1_register, 2);
            __m128i pixel6h = _mm_cmplt_epi16(pixel6v,pixelcv);

            /* pixel 7 */
            __m128i pixel7v = _mm_alignr_epi8(l1_register_next,l1_register, 4);
            __m128i pixel7h = _mm_cmplt_epi16(pixel7v,pixelcv);

            /* pixel 8 */
            __m128i pixel8v = _mm_alignr_epi8(l1_register_next,l1_register, 6);
            __m128i pixel8h = _mm_cmplt_epi16(pixel8v,pixelcv);

            /* pixel 9 */
            __m128i pixel9v = _mm_alignr_epi8(l1_register_next,l1_register, 8);
            __m128i pixel9h = _mm_cmplt_epi16(pixel9v,pixelcv);

            /* create bitfield part 1*/
            __m128i resultByte1 = _mm_and_si128(_mm_set1_epi8(128u),pixel0h);
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(64),pixel1h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(32),pixel2h));
            resultByte1 = _mm_or_si128(resultByte1,_mm_and_si128(_mm_set1_epi8(16),pixel3h));
            __m128i resultByte1b = _mm_and_si128(_mm_set1_epi8(8),pixel4h);
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(4),pixel5h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(2),pixel6h));
            resultByte1b = _mm_or_si128(resultByte1b,_mm_and_si128(_mm_set1_epi8(1),pixel7h));
            resultByte1 = _mm_or_si128(resultByte1, resultByte1b);

            /** line **/
            /* pixel 10 */
            __m128i pixel10h = _mm_cmplt_epi16(l2_register,pixelcv);

            /* pixel 11 */
            __m128i pixel11v = _mm_alignr_epi8(l2_register_next,l2_register, 2);
            __m128i pixel11h = _mm_cmplt_epi16(pixel11v,pixelcv);

            /* pixel 12 */
            __m128i pixel12v = _mm_alignr_epi8(l2_register_next,l2_register, 6);
            __m128i pixel12h = _mm_cmplt_epi16(pixel12v,pixelcv);
            
            /* pixel 13 */
            __m128i pixel13v = _mm_alignr_epi8(l2_register_next,l2_register, 8);
            __m128i pixel13h = _mm_cmplt_epi16(pixel13v,pixelcv);

            /* line */
            /* pixel 14 */
            __m128i pixel14h = _mm_cmplt_epi16(l3_register,pixelcv);

            /* pixel 15 */
            __m128i pixel15v = _mm_alignr_epi8(l3_register_next,l3_register, 2);
            __m128i pixel15h = _mm_cmplt_epi16(pixel15v,pixelcv);

            /* pixel 16 */
            __m128i pixel16v = _mm_alignr_epi8(l3_register_next,l3_register, 4);
            __m128i pixel16h = _mm_cmplt_epi16(pixel16v,pixelcv);

            /* pixel 17 */
            __m128i pixel17v = _mm_alignr_epi8(l3_register_next,l3_register, 6);
            __m128i pixel17h = _mm_cmplt_epi16(pixel17v,pixelcv);

            /* pixel 18 */
            __m128i pixel18v = _mm_alignr_epi8(l3_register_next,l3_register, 8);
            __m128i pixel18h = _mm_cmplt_epi16(pixel18v,pixelcv);

            /* create bitfield part 2 */
            __m128i resultByte2 = _mm_and_si128(_mm_set1_epi8(128u),pixel8h);
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(64),pixel9h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(32),pixel10h));
            resultByte2 = _mm_or_si128(resultByte2,_mm_and_si128(_mm_set1_epi8(16),pixel11h));
            __m128i resultByte2b = _mm_and_si128(_mm_set1_epi8(8),pixel12h);
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(4),pixel13h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(2),pixel14h));
            resultByte2b = _mm_or_si128(resultByte2b,_mm_and_si128(_mm_set1_epi8(1),pixel15h));
            resultByte2 = _mm_or_si128(resultByte2, resultByte2b);

            /* line */
            /* pixel 19 */
            __m128i pixel19h = _mm_cmplt_epi16(l4_register,pixelcv);

            /* pixel 20 */
            __m128i pixel20v = _mm_alignr_epi8(l4_register_next,l4_register, 2);
            __m128i pixel20h = _mm_cmplt_epi16(pixel20v,pixelcv);

            /* pixel 21 */
            __m128i pixel21v = _mm_alignr_epi8(l4_register_next,l4_register, 4);
            __m128i pixel21h = _mm_cmplt_epi16(pixel21v,pixelcv);

            /* pixel 22 */
            __m128i pixel22v = _mm_alignr_epi8(l4_register_next,l4_register, 6);
            __m128i pixel22h = _mm_cmplt_epi16(pixel22v,pixelcv);

            /* pixel 23 */
            __m128i pixel23v = _mm_alignr_epi8(l4_register_next,l4_register, 8);
            __m128i pixel23h = _mm_cmplt_epi16(pixel23v,pixelcv);

            /* create bitfield part 3*/

            __m128i resultByte3 = _mm_and_si128(_mm_set1_epi8(128u),pixel16h);
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(64),pixel17h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(32),pixel18h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(16),pixel19h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(8),pixel20h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(4),pixel21h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(2),pixel22h));
            resultByte3 = _mm_or_si128(resultByte3,_mm_and_si128(_mm_set1_epi8(1),pixel23h));

            /* blend byte 1 and byte 2,then byte3, lower part */
            __m128i resultByte1Lower = _mm_shuffle_epi8(resultByte1, expandLowerMask);
            __m128i resultByte2Lower = _mm_shuffle_epi8(resultByte2, expandLowerMask);
            __m128i blendB1B2 = _mm_blendv_epi8(resultByte1Lower,resultByte2Lower,blendB1B2Mask);
            blendB1B2 = _mm_and_si128(blendB1B2, _mm_set1_epi32(0x00FFFFFF)); // zero first byte
            __m128i blendB1B2B3L = _mm_blendv_epi8(blendB1B2,_mm_shuffle_epi8(resultByte3, expandLowerMask),blendB1B2B3Mask);

            /* blend byte 1 and byte 2,then byte3, upper part */
            __m128i resultByte1Upper = _mm_shuffle_epi8(resultByte1, expandUpperMask);
            __m128i resultByte2Upper = _mm_shuffle_epi8(resultByte2, expandUpperMask);
            blendB1B2 = _mm_blendv_epi8(resultByte1Upper,resultByte2Upper,blendB1B2Mask);
            blendB1B2 = _mm_and_si128(blendB1B2, _mm_set1_epi32(0x00FFFFFF)); // zero first byte
            __m128i blendB1B2B3H = _mm_blendv_epi8(blendB1B2,_mm_shuffle_epi8(resultByte3, expandUpperMask),blendB1B2B3Mask);

            /* shift because of offsets */
            __m128i c = _mm_alignr_epi8(blendB1B2B3L, lastResultLower, 8);
            _mm_store_si128((__m128i*)result, c);
            _mm_store_si128((__m128i*)(result+4), _mm_alignr_epi8(blendB1B2B3H, blendB1B2B3L, 8) ); 

            result += 8;
            lastResultLower = blendB1B2B3H;

            /*load next */
            l0_register = l0_register_next;
            l1_register = l1_register_next;
            l2_register = l2_register_next;
            l3_register = l3_register_next;
            l4_register = l4_register_next;

    }
    /* last 8 pixels */
    {
        int i = height - 3;
        for (sint32 j=width-8; j < (sint32)width-2; j++) {
            const int centerValue = *getPixel16(source, width,j,i);
            uint32 value = 0;
            for (sint32 x=-2; x <= 2; x++) {
                for (sint32 y=-2; y <= 2; y++) {
                    if (x!=0 || y!=0) {
                        value *= 2;
                        if (centerValue >  *getPixel16(source, width,j+y,i+x)) {
                            value += 1;
                        }
                    }
                }     
            } 
            *getPixel32(dest,width,j,i) = value;
        }
        *getPixel32(dest,width,width-2,i) = 255;
        *getPixel32(dest,width,width-1,i) = 255;
    }
}

inline void vecSortandSwap(__m128& a, __m128& b)
{
    __m128 temp = a;
    a = _mm_min_ps(a,b);
    b = _mm_max_ps(temp,b);
}

void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height)
{
    // check width restriction
    assert(width % 4 == 0);
    
    float32* destStart = dest;
    //  lines
    float32* line1 = source;
    float32* line2 = source + width;
    float32* line3 = source + 2*width;

    float32* end = source + width*height;

    dest += width;
    __m128 lastMedian = _mm_setzero_ps();

    do {
        // fill value
        const __m128 l1_reg = _mm_load_ps(line1);
        const __m128 l1_reg_next = _mm_load_ps(line1+4);
        __m128 v0 = l1_reg;
        __m128 v1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 4));
        __m128 v2 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 8));

        const __m128 l2_reg = _mm_load_ps(line2);
        const __m128 l2_reg_next = _mm_load_ps(line2+4);
        __m128 v3 = l2_reg;
        __m128 v4 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 4));
        __m128 v5 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 8));

        const __m128 l3_reg = _mm_load_ps(line3);
        const __m128 l3_reg_next = _mm_load_ps(line3+4);
        __m128 v6 = l3_reg;
        __m128 v7 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 4));
        __m128 v8 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 8));

        // find median through sorting network
        vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
        vecSortandSwap(v0, v1) ; vecSortandSwap(v3, v4) ; vecSortandSwap(v6, v7) ;
        vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
        vecSortandSwap(v0, v3) ; vecSortandSwap(v5, v8) ; vecSortandSwap(v4, v7) ;
        vecSortandSwap(v3, v6) ; vecSortandSwap(v1, v4) ; vecSortandSwap(v2, v5) ;
        vecSortandSwap(v4, v7) ; vecSortandSwap(v4, v2) ; vecSortandSwap(v6, v4) ;
        vecSortandSwap(v4, v2) ; 

        // comply to alignment restrictions
        const __m128i c = _mm_alignr_epi8(_mm_castps_si128(v4), _mm_castps_si128(lastMedian), 12);
        _mm_store_si128((__m128i*)dest, c);
        lastMedian = v4;

        dest+=4; line1+=4; line2+=4; line3+=4;

    } while (line3+4+4 <= end);

    memcpy(destStart, source, sizeof(float32)*(width+1));
    memcpy(destStart+width*height-width-1-3, source+width*height-width-1-3, sizeof(float32)*(width+1+3));
}


class StereoSGMParams_t {
 public:
  uint16_t P1; // +/-1 discontinuity penalty
  uint16_t InvalidDispCost;  // init value for invalid disparities (half of max value seems ok)
  uint16_t NoPasses; // one or two passes
  uint16_t Paths; // 8, 0-4 gives 1D path, rest undefined
  float Uniqueness; // uniqueness ratio
  bool MedianFilter; // apply median filter
  bool lrCheck; // apply lr-check
  bool rlCheck; // apply rl-check (on right image)
  int lrThreshold; // threshold for lr-check
  int subPixelRefine; // sub pixel refine method

  // varP2 = - alpha * abs(I(x)-I(x-r))+gamma
  float Alpha; // variable P2 alpha
  uint16_t Gamma; // variable P2 gamma
  uint16_t P2min; // varP2 cannot get lower than P2min

  /* param set out of the paper from Banz
     - noiseless (Cones): P1 = 11, P2min = 17, gamma = 35, alpha=0.5 8bit images
     - Cones with noise: P1=20, P2min=24, gamma = 70, alpha=0.5
     */

  StereoSGMParams_t()
      : P1(7)
        ,InvalidDispCost(12)
        ,NoPasses(2)
        ,Paths(8)
        ,Uniqueness(0.95f)
        ,MedianFilter(true)
        ,lrCheck(true)
        ,rlCheck(true)
        ,lrThreshold(1)
        ,subPixelRefine(-1)
        ,Alpha(0.25f)
        ,Gamma(50)
        ,P2min(17)
  {

  }
} ;


// template param is image pixel type (uint8 or uint16)
template <typename T>
class StereoSGM {
 private:
  int m_width;
  int m_height;
  int m_maxDisp;
  StereoSGMParams_t m_params;
  uint16* m_S;

  float32* m_dispLeftImgUnfiltered;
  float32* m_dispRightImgUnfiltered;

  // SSE version, only maximum 8 paths supported
  template <int NPaths> void accumulateVariableParamsSSE(uint16* &dsi, T* img, uint16* &S);

public:
  // SGM
  StereoSGM(int i_width, int i_height, int i_maxDisp, StereoSGMParams_t i_params);
  ~StereoSGM();

  void process(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg);

  void processParallel(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg, int numThreads);

  // accumulation cube
  uint16* getS();
  // dimensions
  int getHeight();
  int getWidth();
  int getMaxDisp();
  // change params
  void setParams(const StereoSGMParams_t& i_params);

  void adaptMemory(int i_width, int i_height, int i_maxDisp);
};

template <typename T>
class StripedStereoSGM {
  int m_width;
  int m_height;
  int m_maxDisp;
  int m_numStrips;
  int m_border;

  std::vector<StereoSGM<T>* > m_sgmVector;
  std::vector<float32*> m_stripeDispImgVector;
  std::vector<float32*> m_stripeDispImgRightVector;

 public:
  StripedStereoSGM(int i_width, int i_height, int i_maxDisp, int numStrips, const int border, StereoSGMParams_t i_params)
  {
    m_width = i_width;
    m_height = i_height;
    m_maxDisp = i_maxDisp;
    m_numStrips = numStrips;
    m_border = border;

    if (numStrips <= 1) {
      m_sgmVector.push_back(new StereoSGM<T>(m_width, m_height, m_maxDisp, i_params));
    }
    else {
      for (int n = 0; n < m_numStrips; n++)
      {
        sint32 startLine = n*(m_height / m_numStrips);
        sint32 endLine = (n + 1)*(m_height / m_numStrips) - 1;

        if (n == m_numStrips - 1) {
          endLine = m_height - 1;
        }

        sint32 startLineWithBorder = MAX(startLine - m_border, 0);
        sint32 endLineWithBorder = MIN(endLine + m_border, m_height - 1);
        sint32 noLinesBorder = endLineWithBorder - startLineWithBorder + 1;

        m_stripeDispImgVector.push_back((float32*)_mm_malloc(noLinesBorder*m_width*sizeof(float32), 16));
        m_stripeDispImgRightVector.push_back((float32*)_mm_malloc(noLinesBorder*m_width*sizeof(float32), 16));

        m_sgmVector.push_back(new StereoSGM<T>(m_width, noLinesBorder, m_maxDisp, i_params));
      }
    }
  }
  ~StripedStereoSGM()
  {
    if (m_numStrips > 1) {
      for (int i = 0; i < m_numStrips; i++)
      {
        _mm_free(m_stripeDispImgVector[i]);
        _mm_free(m_stripeDispImgRightVector[i]);
        delete m_sgmVector[i];
      }
    }
    else {
      delete m_sgmVector[0];
    }
  }

  void process(T* leftImg, float32* output, float32* dispImgRight, uint16* dsi, const int numThreads)
  {
    // no strips
    if (m_numStrips <= 1) {
      // normal processing (no strip)
      m_sgmVector[0]->process(dsi, leftImg, output, dispImgRight);

      return;
    }

    int NUM_THREADS = numThreads;
#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp for schedule(static, m_numStrips/NUM_THREADS)
      for (int n = 0; n < m_numStrips; n++){

        sint32 startLine = n*(m_height / m_numStrips);
        sint32 endLine = (n + 1)*(m_height / m_numStrips) - 1;

        if (n == m_numStrips - 1) {
          endLine = m_height - 1;
        }

        sint32 startLineWithBorder = MAX(startLine - m_border, 0);

        int imgOffset = startLineWithBorder * m_width;
        int dsiOffset = startLineWithBorder * m_width * (m_maxDisp + 1);

        m_sgmVector[n]->process(dsi + dsiOffset, leftImg + imgOffset, m_stripeDispImgVector[n], m_stripeDispImgRightVector[n]);

        // copy back
        int upperBorder = startLine - startLineWithBorder;
        memcpy(output + startLine*m_width, m_stripeDispImgVector[n] + upperBorder*m_width, (endLine - startLine + 1)*m_width*sizeof(float32));
        memcpy(dispImgRight + startLine*m_width, m_stripeDispImgRightVector[n] + upperBorder*m_width, (endLine - startLine + 1)*m_width*sizeof(float32));
      }
    }
  }
};

template <typename T>
StereoSGM<T>::StereoSGM(int i_width, int i_height, int i_maxDisp, StereoSGMParams_t i_params)
: m_width(i_width)
, m_height(i_height)
, m_maxDisp(i_maxDisp)
    , m_params(i_params)
{
  m_S = (uint16*) _mm_malloc(m_width*m_height*(i_maxDisp+1)*sizeof(uint16),16);

  m_dispLeftImgUnfiltered = (float*)_mm_malloc(m_width*m_height*sizeof(float), 16);
  m_dispRightImgUnfiltered = (float*)_mm_malloc(m_width*m_height*sizeof(float), 16);
}

template <typename T>
StereoSGM<T>::~StereoSGM()
{
  if (m_S != NULL)
    _mm_free(m_S);

  if (m_dispLeftImgUnfiltered != NULL)
    _mm_free(m_dispLeftImgUnfiltered);
  if (m_dispRightImgUnfiltered != NULL)
    _mm_free(m_dispRightImgUnfiltered);
}

template <typename T>
void StereoSGM<T>::adaptMemory(int i_width, int i_height, int i_maxDisp)
{
  if (i_width*i_height*i_maxDisp > m_width * m_height*m_maxDisp) {
    if (m_S != NULL) {
      _mm_free(m_S);
    }
    m_width = i_width;
    m_height = i_height;
    m_maxDisp = i_maxDisp;
    m_S = (uint16*) _mm_malloc(m_width*m_height*(i_maxDisp+1)*sizeof(uint16),16);
  } else {
    m_width = i_width;
    m_height = i_height;
    m_maxDisp = i_maxDisp;
  }
}

template <typename T>
uint16* StereoSGM<T>::getS()
{
  return m_S;
}

template <typename T>
int StereoSGM<T>::getHeight()
{
  return m_height;
}

template <typename T>
int StereoSGM<T>::getWidth()
{
  return m_width;
}

template <typename T>
int StereoSGM<T>::getMaxDisp()
{
  return m_maxDisp;
}

template <typename T>
void StereoSGM<T>::setParams(const StereoSGMParams_t& i_params)
{
  m_params = i_params;
}


inline void swapPointers(uint16*& p1, uint16*& p2)
{
  uint16* temp = p1;
  p1 = p2;
  p2 = temp;
}

inline sint32 adaptP2(const float32& alpha, const uint16& I_p, const uint16& I_pr, const int& gamma, const int& P2min)
{
  sint32 result;
  result = (sint32)(-alpha * abs((sint32)I_p-(sint32)I_pr)+gamma);
  if (result < P2min)
    result = P2min;
  return result;
}

template <typename T>
void StereoSGM<T>::process(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg)
{
  if (m_params.Paths == 0) {
    accumulateVariableParamsSSE<0>(dsi, img, m_S);
  }
  else if (m_params.Paths == 1) {
    accumulateVariableParamsSSE<1>(dsi, img, m_S);
  }
  else if (m_params.Paths == 2) {
    accumulateVariableParamsSSE<2>(dsi, img, m_S);
  }
  else if (m_params.Paths == 3) {
    accumulateVariableParamsSSE<3>(dsi, img, m_S);
  }
  else if (m_params.Paths == 8) {
    accumulateVariableParamsSSE<8>(dsi, img, m_S);
  }

  // median filtering preparation
  float *dispLeftImgUnfiltered;
  float *dispRightImgUnfiltered;

  if (m_params.MedianFilter) {
    dispLeftImgUnfiltered = m_dispLeftImgUnfiltered;
    dispRightImgUnfiltered = m_dispRightImgUnfiltered;
  } else {
    dispLeftImgUnfiltered = dispLeftImg;
    dispRightImgUnfiltered = dispRightImg;
  }

  if (m_params.lrCheck) {
    matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
    matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);

    /* subpixel refine */
    if (m_params.subPixelRefine != -1) {
      subPixelRefine(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.subPixelRefine);
    }

    if (m_params.MedianFilter) {
      median3x3_SSE(dispLeftImgUnfiltered, dispLeftImg, m_width, m_height);
      median3x3_SSE(dispRightImgUnfiltered, dispRightImg, m_width, m_height);
    }
    doLRCheck(dispLeftImg, dispRightImg, m_width, m_height, m_params.lrThreshold);

    if (m_params.rlCheck)
    {
      doRLCheck(dispRightImg, dispLeftImg, m_width, m_height, m_params.lrThreshold);
    }

  } else {
    // find disparities with minimum accumulated costs
    matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);

    /* subpixel refine */
    if (m_params.subPixelRefine != -1) {
      subPixelRefine(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.subPixelRefine);
    }
  }
}

template <typename T>
void StereoSGM<T>::processParallel(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg, sint32 numThreads)
{

  if (m_params.Paths == 0) {
    accumulateVariableParamsSSE<0>(dsi, img, m_S);
  }
  else if (m_params.Paths == 1) {
    accumulateVariableParamsSSE<1>(dsi, img, m_S);
  }
  else if (m_params.Paths == 2) {
    accumulateVariableParamsSSE<2>(dsi, img, m_S);
  }
  else if (m_params.Paths == 3) {
    accumulateVariableParamsSSE<3>(dsi, img, m_S);
  }
  else if (m_params.Paths == 8) {
    accumulateVariableParamsSSE<8>(dsi, img, m_S);
  }

  // median filtering preparation
  float *dispLeftImgUnfiltered;
  float *dispRightImgUnfiltered;

  if (m_params.MedianFilter) {
    dispLeftImgUnfiltered = m_dispLeftImgUnfiltered;
    dispRightImgUnfiltered = m_dispRightImgUnfiltered;
  } else {
    dispLeftImgUnfiltered = dispLeftImg;
    dispRightImgUnfiltered = dispRightImg;
  }

  if (m_params.lrCheck) {
    // find disparities with minimum accumulated costs
    if (numThreads == 1) {
      matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
      matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);
    } else if (numThreads > 1) {
#pragma omp parallel num_threads(2)
      {
#pragma omp sections nowait
        {
#pragma omp section
          {
            if (m_params.subPixelRefine != -1) {
              matchWTAAndSubPixel_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
            } else {
              matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
            }
          }
#pragma omp section
          {
            matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);
          }
        }
      }
    }

    if (m_params.MedianFilter) {
      median3x3_SSE(dispLeftImgUnfiltered, dispLeftImg, m_width, m_height);
      median3x3_SSE(dispRightImgUnfiltered, dispRightImg, m_width, m_height);
    }
    doLRCheck(dispLeftImg, dispRightImg, m_width, m_height, m_params.lrThreshold);

    if (m_params.rlCheck)
    {
      doRLCheck(dispRightImg, dispLeftImg, m_width, m_height, m_params.lrThreshold);
    }
  } else {
    matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);

    /* subpixel refine */
    if (m_params.subPixelRefine != -1) {
      subPixelRefine(dispLeftImg, m_S, m_width, m_height, m_maxDisp, m_params.subPixelRefine);
    }
  }
}

// accumulate along paths
// variable P2 param
template<typename T>
template <int NPaths>
void StereoSGM<T>::accumulateVariableParamsSSE(uint16* &dsi, T* img, uint16* &S)
{
  /* Params */
  const sint32 paramP1 = m_params.P1;
  const uint16 paramInvalidDispCost = m_params.InvalidDispCost;
  const int paramNoPasses = m_params.NoPasses;
  const uint16 MAX_SGM_COST = UINT16_MAX;

  // change params for fixed, if necessary
  const float32 paramAlpha = m_params.Alpha;
  const sint32 paramGamma = m_params.Gamma;
  const sint32 paramP2min =  m_params.P2min;

  const int width = m_width;
  const int width2 = width+2;
  const int maxDisp = m_maxDisp;
  const int height = m_height;
  const int disp = maxDisp+1;
  const int dispP2 = disp+8;

  // accumulated cost along path r
  // two extra elements for -1 and maxDisp+1 disparity
  // current and last line (or better, two buffers)
  uint16* L_r0      = ((uint16*) _mm_malloc(dispP2*sizeof(uint16),16))+1;
  uint16* L_r0_last = ((uint16*) _mm_malloc(dispP2*sizeof(uint16),16))+1;
  uint16* L_r1      = ((uint16*) _mm_malloc(width2*dispP2*sizeof(uint16)+1,16))+dispP2+1;
  uint16* L_r1_last = ((uint16*) _mm_malloc(width2*dispP2*sizeof(uint16)+1,16))+dispP2+1;
  uint16* L_r2_last = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
  uint16* L_r3_last = ((uint16*) _mm_malloc(width2*dispP2*sizeof(uint16)+1,16))+dispP2+1;

  /* image line pointers */
  T* img_line_last = NULL;
  T* img_line = NULL;

  /* left border */
  memset(&L_r1[-dispP2], MAX_SGM_COST, sizeof(uint16)*(dispP2));
  memset(&L_r1_last[-dispP2], MAX_SGM_COST, sizeof(uint16)*(dispP2));
  L_r1[-dispP2 - 1] = MAX_SGM_COST;
  L_r1_last[-dispP2 - 1] = MAX_SGM_COST;
  memset(&L_r3_last[-dispP2], MAX_SGM_COST, sizeof(uint16)*(dispP2));
  L_r3_last[-dispP2 - 1] = MAX_SGM_COST;

  /* right border */
  memset(&L_r1[width*dispP2-1], MAX_SGM_COST, sizeof(uint16)*(dispP2));
  memset(&L_r1_last[width*dispP2-1], MAX_SGM_COST, sizeof(uint16)*(dispP2));
  memset(&L_r3_last[width*dispP2-1], MAX_SGM_COST, sizeof(uint16)*(dispP2));

  // min L_r cache
  uint16 minL_r0_Array[2];
  uint16* minL_r0 = &minL_r0_Array[0];
  uint16* minL_r0_last = &minL_r0_Array[1];
  uint16* minL_r1 = (uint16*) _mm_malloc(width2*sizeof(uint16),16)+1;
  uint16* minL_r1_last = (uint16*) _mm_malloc(width2*sizeof(uint16),16)+1;
  uint16* minL_r2_last = (uint16*) _mm_malloc(width*sizeof(uint16),16);
  uint16* minL_r3_last = (uint16*) _mm_malloc(width2*sizeof(uint16),16)+1;

  minL_r1[-1] =  minL_r1_last[-1] = 0;
  minL_r1[width] = minL_r1_last[width] = 0;
  minL_r3_last[-1] = 0;
  minL_r3_last[width] = 0;

  /*[formula 13 in the paper]
    compute L_r(p, d) = C(p, d) +
    min(L_r(p-r, d),
    L_r(p-r, d-1) + P1,
    L_r(p-r, d+1) + P1,
    min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
    where p = (x,y), r is one of the directions.
    we process all the directions at once:
    ( basic 8 paths )
0: r=(-1, 0) --> left to right
1: r=(-1, -1) --> left to right, top to bottom
2: r=(0, -1) --> top to bottom
3: r=(1, -1) --> top to bottom, right to left
( additional ones for 16 paths )
4: r=(-2, -1) --> two left, one down
5: r=(-1, -1*2) --> one left, two down
6: r=(1, -1*2) --> one right, two down
7: r=(2, -1) --> two right, one down
*/

  // border cases L_r0[0 - disp], L_r1,2,3 is maybe not needed, as done above
  L_r0_last[-1] = L_r1_last[-1] = L_r2_last[-1] = L_r3_last[-1] = MAX_SGM_COST;
  L_r0_last[disp] = L_r1_last[disp] = L_r2_last[disp] = L_r3_last[disp] = MAX_SGM_COST;
  L_r0[-1] = L_r1[-1] = MAX_SGM_COST;
  L_r0[disp] = L_r1[disp] = MAX_SGM_COST;

  for (int pass = 0; pass < paramNoPasses; pass++) {
    int i1; int i2; int di;
    int j1; int j2; int dj;
    if (pass == 0) {
      /* top-down pass */
      i1 = 0; i2 = height; di = 1;
      j1 = 0; j2 = width;  dj = 1;
    } else {
      /* bottom-up pass */
      i1 = height-1; i2 = -1; di = -1;
      j1 = width-1; j2 = -1;  dj = -1;
    }
    img_line = img+i1*width;

    /* first line is simply costs C, except for path L_r0 */
    // first pixel
    uint16 minCost = MAX_SGM_COST;
    if (pass == 0) {
      for (int d=0; d < disp; d++) {
        uint16 cost = *getDispAddr_xyd(dsi, width, disp, i1, j1, d);
        if (cost == 255)
          cost = paramInvalidDispCost;
        L_r0_last[d] = cost;
        L_r1_last[j1*dispP2+d] = cost;
        L_r2_last[j1*dispP2+d] = cost;
        L_r3_last[j1*dispP2+d] = cost;
        if (cost < minCost) {
          minCost = cost;
        }
        *getDispAddr_xyd(S, width, disp, i1,j1, d) = cost;
      }
    } else {
      for (int d=0; d < disp; d++) {
        uint16 cost = *getDispAddr_xyd(dsi, width, disp, i1, j1, d);
        if (cost == 255)
          cost = paramInvalidDispCost;
        L_r0_last[d] = cost;
        L_r1_last[j1*dispP2+d] = cost;
        L_r2_last[j1*dispP2+d] = cost;
        L_r3_last[j1*dispP2+d] = cost;
        if (cost < minCost) {
          minCost = cost;
        }
        *getDispAddr_xyd(S, width, disp, i1,j1, d) += cost;
      }
    }
    *minL_r0_last = minCost;
    minL_r1_last[j1] = minCost;
    minL_r2_last[j1] = minCost;
    minL_r3_last[j1] = minCost;

    // rest of first line
    for (int j=j1+dj; j != j2; j += dj) {
      uint16 minCost = MAX_SGM_COST;
      *minL_r0 = MAX_SGM_COST;
      for (int d=0; d < disp; d++) {
        uint16 cost = *getDispAddr_xyd(dsi, width, disp, i1, j, d);
        if (cost == 255)
          cost = paramInvalidDispCost;
        if (NPaths != 0) {
          L_r1_last[j*dispP2+d] = cost;
          L_r2_last[j*dispP2+d] = cost;
          L_r3_last[j*dispP2+d] = cost;
        }

        if (cost < minCost ) {
          minCost = cost;
        }

        // minimum along L_r0
        sint32 minPropCost = L_r0_last[d]; // same disparity cost
        // P1 costs
        sint32 costP1m = L_r0_last[d-1]+paramP1;
        if (minPropCost > costP1m)
          minPropCost = costP1m;
        sint32 costP1p = L_r0_last[d+1]+paramP1;
        if (minPropCost > costP1p) {
          minPropCost =costP1p;
        }
        // P2 costs
        sint32 minCostP2 = *minL_r0_last;
        sint32 varP2 = adaptP2(paramAlpha, img_line[j],img_line[j-dj],paramGamma, paramP2min);
        if (minPropCost > minCostP2+varP2)
          minPropCost = minCostP2+varP2;
        // add offset
        minPropCost -= minCostP2;

        const uint16 newCost = saturate_cast<uint16>(cost + minPropCost);
        L_r0[d] = newCost;

        if (*minL_r0 > newCost) {
          *minL_r0 = newCost;
        }

        // cost sum
        if (pass == 0) {
          *getDispAddr_xyd(S, width, disp, i1,j, d) = saturate_cast<uint16>(cost + minPropCost);
        } else {
          *getDispAddr_xyd(S, width, disp, i1,j, d) += saturate_cast<uint16>(cost + minPropCost);
        }

      }
      if (NPaths != 0) {
        minL_r1_last[j] = minCost;
        minL_r2_last[j] = minCost;
        minL_r3_last[j] = minCost;
      }

      // swap L0 buffers
      swapPointers(L_r0, L_r0_last);
      swapPointers(minL_r0, minL_r0_last);

      // border cases: disparities -1 and disp
      L_r1_last[j*dispP2-1] = L_r2_last[j*dispP2-1] = L_r3_last[j*dispP2-1] = MAX_SGM_COST;
      L_r1_last[j*dispP2+disp] = L_r2_last[j*dispP2+disp] = L_r3_last[j*dispP2+disp] = MAX_SGM_COST;

      L_r1[j*dispP2-1] = MAX_SGM_COST;
      L_r1[j*dispP2+disp] = MAX_SGM_COST;
    }

    // same as img_line in first iteration, because of boundaries!
    img_line_last = img+(i1+di)*width;

    // remaining lines
    for (int i=i1+di; i != i2; i+=di) {

      memset(L_r0_last, 0, sizeof(uint16)*disp);
      *minL_r0_last = 0;

      img_line = img+i*width;

      for (int j=j1; j != j2; j+=dj) {
        *minL_r0 = MAX_SGM_COST;
        __m128i minLr_08 = _mm_set1_epi16(MAX_SGM_COST);
        __m128i minLr_18 = _mm_set1_epi16(MAX_SGM_COST);
        __m128i minLr_28 = _mm_set1_epi16(MAX_SGM_COST);
        __m128i minLr_38 = _mm_set1_epi16(MAX_SGM_COST);

        const sint32 varP2_r0 = adaptP2(paramAlpha, img_line[j], img_line[j-dj],paramGamma, paramP2min);
        sint32 varP2_r1, varP2_r2, varP2_r3;
        if (NPaths != 0) {
          varP2_r1 = adaptP2(paramAlpha, img_line[j], img_line_last[j-dj],paramGamma, paramP2min);
          varP2_r2 = adaptP2(paramAlpha, img_line[j], img_line_last[j],paramGamma, paramP2min);
          varP2_r3 = adaptP2(paramAlpha, img_line[j], img_line_last[j+dj],paramGamma, paramP2min);
        }
        else
        {
          varP2_r1 = 0;
          varP2_r2 = 0;
          varP2_r3 = 0;
        }

        //only once per point
        const __m128i varP2_r08 = _mm_set1_epi16((uint16) varP2_r0);
        const __m128i varP2_r18 = _mm_set1_epi16((uint16) varP2_r1);
        const __m128i varP2_r28 = _mm_set1_epi16((uint16) varP2_r2);
        const __m128i varP2_r38 = _mm_set1_epi16((uint16) varP2_r3);

        const __m128i minCostP28_r0 =  _mm_set1_epi16((uint16) (*minL_r0_last));
        const __m128i minCostP28_r1 =  _mm_set1_epi16((uint16) minL_r1_last[j-dj]);
        const __m128i minCostP28_r2 =  _mm_set1_epi16((uint16) minL_r2_last[j]);
        const __m128i minCostP28_r3 =  _mm_set1_epi16((uint16) minL_r3_last[j+dj]);

        const __m128i curP2cost8_r0 = _mm_adds_epu16(varP2_r08, minCostP28_r0);
        const __m128i curP2cost8_r1 = _mm_adds_epu16(varP2_r18, minCostP28_r1);
        const __m128i curP2cost8_r2 = _mm_adds_epu16(varP2_r28, minCostP28_r2);
        const __m128i curP2cost8_r3 = _mm_adds_epu16(varP2_r38, minCostP28_r3);

        int d=0;
        __m128i upper8_r0 = _mm_load_si128( (__m128i*)( L_r0_last+0-1 ) );
        int baseIndex_r2 = ((j)*dispP2)+d;

        const int baseIndex_r1 = ((j - dj)*dispP2) + d;
        __m128i upper8_r1 = _mm_load_si128((__m128i*)(L_r1_last + baseIndex_r1 - 1));
        __m128i upper8_r2 = _mm_load_si128( (__m128i*)( L_r2_last+baseIndex_r2-1 ) );
        const int baseIndex_r3 = ((j+dj)*dispP2)+d;
        __m128i upper8_r3 = _mm_load_si128( (__m128i*)( L_r3_last+baseIndex_r3-1 ) );
        const __m128i paramP18 = _mm_set1_epi16((uint16)paramP1);

        for (; d < disp-7; d+=8) {
          //--------------------------------------------------------------------------------------------------------------------------------------------------------
          //to save sum of all paths
          __m128i newCost8_ges = _mm_setzero_si128();

          __m128i cost8;

          cost8 = _mm_load_si128( (__m128i*) getDispAddr_xyd(dsi, width, disp, i, j, d) );

          //--------------------------------------------------------------------------------------------------------------------------------------------------------
          // minimum along L_r0
          if (NPaths == 0 || NPaths == 8 || NPaths == 16) {
            __m128i minPropCost8;

            const __m128i lower8_r0 = upper8_r0;
            upper8_r0 = _mm_load_si128( (__m128i*)( L_r0_last+d-1+8 ) );

            // P1 costs
            const __m128i costPm8_r0 = _mm_adds_epu16(lower8_r0, paramP18);

            const __m128i costPp8_r0 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r0, lower8_r0, 4), paramP18);

            minPropCost8 = _mm_alignr_epi8(upper8_r0, lower8_r0, 2);
            __m128i temp = _mm_min_epu16(costPp8_r0, costPm8_r0);
            minPropCost8 = _mm_min_epu16(minPropCost8, temp);
            minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r0);
            minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r0);


            const __m128i newCost8_r0 = _mm_adds_epu16(cost8, minPropCost8);

            _mm_storeu_si128((__m128i*) (L_r0_last+d) , newCost8_r0);

            //sum of all Paths
            newCost8_ges = newCost8_r0;

            minLr_08 = _mm_min_epu16(minLr_08, newCost8_r0);
          }
          if (NPaths != 0) {
            //--------------------------------------------------------------------------------------------------------------------------------------------------------
            const int baseIndex_r1 = ((j-dj)*dispP2)+d;

            uint16* lastL = L_r1_last;
            uint16* L = L_r1;

            const __m128i lower8_r1 = upper8_r1;
            upper8_r1 = _mm_load_si128( (__m128i*)( lastL+baseIndex_r1-1+8 ) );
            const __m128i costPm8_r1 = _mm_adds_epu16(lower8_r1, paramP18);

            const __m128i costPp8_r1 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r1, lower8_r1, 4), paramP18);

            __m128i minPropCost8 = _mm_alignr_epi8(upper8_r1, lower8_r1, 2);
            __m128i temp = _mm_min_epu16(costPp8_r1, costPm8_r1);
            minPropCost8 = _mm_min_epu16(minPropCost8, temp);
            minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r1);
            minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r1);

            const __m128i newCost8_r1 = _mm_adds_epu16(cost8, minPropCost8);

            _mm_storeu_si128((__m128i*) (L+(j*dispP2)+d) , newCost8_r1);

            //sum of all Paths
            newCost8_ges = _mm_adds_epu16(newCost8_ges, newCost8_r1);

            minLr_18 = _mm_min_epu16(minLr_18, newCost8_r1);

            //--------------------------------------------------------------------------------------------------------------------------------------------------------
            int baseIndex_r2 = ((j)*dispP2) + d;

            const __m128i lower8_r2 = upper8_r2;
            upper8_r2 = _mm_load_si128( (__m128i*)( L_r2_last+baseIndex_r2-1+8 ) );


            const __m128i costPm8_r2 = _mm_adds_epu16(lower8_r2, paramP18);

            const __m128i costPp8_r2 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r2, lower8_r2, 4), paramP18);

            minPropCost8 = _mm_alignr_epi8(upper8_r2, lower8_r2, 2);
            temp = _mm_min_epu16(costPp8_r2, costPm8_r2);
            minPropCost8 = _mm_min_epu16(temp, minPropCost8);
            minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r2);
            minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r2);

            const __m128i newCost8_r2 = _mm_adds_epu16(cost8, minPropCost8);

            _mm_storeu_si128((__m128i*) (L_r2_last+(j*dispP2)+d) , newCost8_r2);

            //sum of all Paths
            newCost8_ges = _mm_adds_epu16(newCost8_ges, newCost8_r2);

            minLr_28 = _mm_min_epu16(minLr_28,newCost8_r2);

            //--------------------------------------------------------------------------------------------------------------------------------------------------------
            int baseIndex_r3 = ((j+dj)*dispP2)+d;

            const __m128i lower8_r3 = upper8_r3;
            upper8_r3 = _mm_load_si128( (__m128i*)( L_r3_last+baseIndex_r3-1+8 ) );

            const __m128i costPm8_r3 = _mm_adds_epu16(lower8_r3, paramP18);

            const __m128i costPp8_r3 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r3, lower8_r3, 4), paramP18);

            minPropCost8 = _mm_alignr_epi8(upper8_r3, lower8_r3, 2);
            minPropCost8 = _mm_min_epu16(minPropCost8, costPm8_r3);
            minPropCost8 = _mm_min_epu16(minPropCost8, costPp8_r3);
            minPropCost8 = _mm_min_epu16(minPropCost8, curP2cost8_r3);
            minPropCost8 = _mm_subs_epu16(minPropCost8, minCostP28_r3);

            const __m128i newCost8_r3 = _mm_adds_epu16(cost8, minPropCost8);

            _mm_storeu_si128((__m128i*) (L_r3_last+(j*dispP2)+d) , newCost8_r3);

            //sum of all Paths
            newCost8_ges = _mm_adds_epu16(newCost8_ges, newCost8_r3);

            minLr_38 = _mm_min_epu16(minLr_38, newCost8_r3);
            //--------------------------------------------------------------------------------------------------------------------------------------------------------
          }
          if (NPaths == 8) {
            if (pass == 0) {
              _mm_store_si128((__m128i*) getDispAddr_xyd(S, width, disp, i,j, d) , newCost8_ges);
            } else {
              _mm_store_si128((__m128i*) getDispAddr_xyd(S, width, disp, i,j, d) , _mm_adds_epu16(_mm_load_si128( (__m128i*) getDispAddr_xyd(S, width, disp, i, j, d) ), newCost8_ges));
            }
          } else if ((NPaths == 0)) {
            if (pass == 0) {
              *getDispAddr_xyd(S, width, disp, i,j, d)   = L_r0_last[d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) = L_r0_last[d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) = L_r0_last[d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) = L_r0_last[d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) = L_r0_last[d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) = L_r0_last[d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) = L_r0_last[d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) = L_r0_last[d+7];
            } else {
              *getDispAddr_xyd(S, width, disp, i,j, d) +=   L_r0_last[d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) += L_r0_last[d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) += L_r0_last[d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) += L_r0_last[d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) += L_r0_last[d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) += L_r0_last[d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) += L_r0_last[d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) += L_r0_last[d+7];
            }
          } else if (NPaths == 1) {
            if (pass == 0) {
              *getDispAddr_xyd(S, width, disp, i,j, d)   = L_r1[j*dispP2+d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) = L_r1[j*dispP2+d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) = L_r1[j*dispP2+d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) = L_r1[j*dispP2+d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) = L_r1[j*dispP2+d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) = L_r1[j*dispP2+d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) = L_r1[j*dispP2+d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) = L_r1[j*dispP2+d+7];
            } else {
              *getDispAddr_xyd(S, width, disp, i,j, d) += L_r1[j*dispP2+d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) += L_r1[j*dispP2+d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) += L_r1[j*dispP2+d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) += L_r1[j*dispP2+d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) += L_r1[j*dispP2+d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) += L_r1[j*dispP2+d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) += L_r1[j*dispP2+d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) += L_r1[j*dispP2+d+7];
            }
          } else if (NPaths == 2) {
            if (pass == 0) {
              *getDispAddr_xyd(S, width, disp, i,j, d)   = L_r2_last[j*dispP2+d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) = L_r2_last[j*dispP2+d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) = L_r2_last[j*dispP2+d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) = L_r2_last[j*dispP2+d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) = L_r2_last[j*dispP2+d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) = L_r2_last[j*dispP2+d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) = L_r2_last[j*dispP2+d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) = L_r2_last[j*dispP2+d+7];
            } else {
              *getDispAddr_xyd(S, width, disp, i,j, d)   += L_r2_last[j*dispP2+d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) += L_r2_last[j*dispP2+d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) += L_r2_last[j*dispP2+d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) += L_r2_last[j*dispP2+d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) += L_r2_last[j*dispP2+d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) += L_r2_last[j*dispP2+d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) += L_r2_last[j*dispP2+d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) += L_r2_last[j*dispP2+d+7];
            }
          } else if (NPaths == 3) {
            if (pass == 0) {
              *getDispAddr_xyd(S, width, disp, i,j, d)   = L_r3_last[j*dispP2+d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) = L_r3_last[j*dispP2+d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) = L_r3_last[j*dispP2+d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) = L_r3_last[j*dispP2+d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) = L_r3_last[j*dispP2+d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) = L_r3_last[j*dispP2+d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) = L_r3_last[j*dispP2+d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) = L_r3_last[j*dispP2+d+7];
            } else {
              *getDispAddr_xyd(S, width, disp, i,j, d)   += L_r3_last[j*dispP2+d];
              *getDispAddr_xyd(S, width, disp, i,j, d+1) += L_r3_last[j*dispP2+d+1];
              *getDispAddr_xyd(S, width, disp, i,j, d+2) += L_r3_last[j*dispP2+d+2];
              *getDispAddr_xyd(S, width, disp, i,j, d+3) += L_r3_last[j*dispP2+d+3];
              *getDispAddr_xyd(S, width, disp, i,j, d+4) += L_r3_last[j*dispP2+d+4];
              *getDispAddr_xyd(S, width, disp, i,j, d+5) += L_r3_last[j*dispP2+d+5];
              *getDispAddr_xyd(S, width, disp, i,j, d+6) += L_r3_last[j*dispP2+d+6];
              *getDispAddr_xyd(S, width, disp, i,j, d+7) += L_r3_last[j*dispP2+d+7];
            }
          }
        }
        *minL_r0_last = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_08),0);
        minL_r1[j] = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_18), 0);
        minL_r2_last[j] = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_28), 0);
        minL_r3_last[j] = (uint16)_mm_extract_epi16(_mm_minpos_epu16(minLr_38), 0);
        //--------------------------------------------------------------------------------------------------------------------------------------------------------
      }

      img_line_last = img_line;
      // exchange buffers - swap line buffers
      {
        // one-liners
        swapPointers(L_r1, L_r1_last);
        swapPointers(minL_r1, minL_r1_last);
      }
    }
  }

  /* free all */
  _mm_free(L_r0-1);
  _mm_free(L_r0_last-1);
  _mm_free(L_r1-dispP2-1);
  _mm_free(L_r1_last-dispP2-1);
  _mm_free(L_r2_last-1);
  _mm_free(L_r3_last-dispP2-1);

  _mm_free(minL_r1-1);
  _mm_free(minL_r1_last-1);
  _mm_free(minL_r2_last);
  _mm_free(minL_r3_last-1);

}

template <typename T>
void census5x5_t_SSE(T* /*source*/, uint32* /*dest*/, uint32 /*width*/, uint32 /*height*/)
{
}

template <>
void census5x5_t_SSE(uint8* source, uint32* dest, uint32 width, uint32 height)
{
    census5x5_SSE(source, dest, width, height);
}

template <>
void census5x5_t_SSE(uint16* source, uint32* dest, uint32 width, uint32 height)
{
    census5x5_16bit_SSE(source, dest, width, height);
}

template <typename T>
void census9x7_t(T* /*source*/, uint64* /*dest*/, uint32 /*width*/, uint32 /*height*/)
{

}

template <>
void census9x7_t(uint8* source, uint64* dest, uint32 width, uint32 height)
{
    census9x7_mode8(source, dest, width, height);
}

template <> inline
void census9x7_t(uint16* source, uint64* dest, uint32 width, uint32 height)
{
    census9x7_mode8_16bit(source, dest, width, height);
}

template <typename T> inline
void costMeasureCensus9x7_xyd_t(T* /*intermediate1*/, T* /*intermediate2*/,
                                int /*height*/, int /*width*/, int /*dispCount*/,
                                uint16* /*dsi*/, int /*threads*/)
{
}

template <> inline
void costMeasureCensus9x7_xyd_t(uint64* intermediate1, uint64* intermediate2,
                                int height, int width, int dispCount, uint16* dsi, int threads)
{
    costMeasureCensus9x7_xyd_parallel(intermediate1, intermediate2,height, width, dispCount, dsi, threads);
}


struct RSGM::Impl
{
  typedef StereoSGM<uint8_t> StereoSGMType;

  Impl()
  {
    fillPopCount16LUT();
  }

  void compute(const RSGM::Config& config, const cv::Mat& left, const cv::Mat& right,
               cv::Mat& disparity)
  {
    _sgm_params.P1 = config.P1;
    _sgm_params.InvalidDispCost = config.invalidDisparityCost;
    _sgm_params.NoPasses = config.numPasses;
    _sgm_params.Paths = config.numPaths;
    _sgm_params.Uniqueness = config.uniquenessRatio;
    _sgm_params.MedianFilter = config.doMedianFilter;
    _sgm_params.lrCheck = config.doLeftRightCheck;
    _sgm_params.rlCheck = config.doRightLeftCheck;

    switch(config.subpixelRefinmentMethod)
    {
      case RSGM::Config::SubpixelRefinmentMethod::None:
        _sgm_params.subPixelRefine = -1;
        break;
      case RSGM::Config::SubpixelRefinmentMethod::Equiangular:
        _sgm_params.subPixelRefine = 0;
        break;
      case RSGM::Config::SubpixelRefinmentMethod::Parabolic:
        _sgm_params.subPixelRefine = 1;
        break;
    }

    _sgm_params.Alpha = config.alpha;
    _sgm_params.Gamma = config.gamma;
    _sgm_params.P2min = config.P2min;

    int extra = left.cols % 16;
    int width = 0;
    int npixels = 0;
    int height = left.rows;
    const int dispCount = 128;
    const int maxDisp = dispCount - 1;

    if(!extra) {
      width = left.cols;
      height = left.rows;
      npixels = width * height;

      _left_image.resize( npixels );
      memcpy(_left_image.data(), left.data, npixels * sizeof(uint8));

      _right_image.resize( npixels );
      memcpy(_right_image.data(), right.data, npixels * sizeof(uint8));
    } else {
      width = left.cols - extra;
      height = left.rows;
      npixels = width * height;

      _left_image.resize( npixels );
      _right_image.resize( npixels );

      auto I0_ptr = _left_image.data();
      auto I1_ptr = _right_image.data();
      for(int r = 0; r < height; ++r)
      {
        auto I0_row = left.ptr<uint8_t>();
        auto I1_row = right.ptr<uint8_t>();
        for(int c = extra; c < width; ++c)
        {
          *I0_ptr++ = I0_row[c];
          *I1_ptr++ = I1_row[c];
        }
      }
    }

    _left_image_census.resize( npixels );
    _right_image_census.resize( npixels );
    _dsi.resize( npixels*(maxDisp + 1) );

    StereoSGMType sgm(width, height, maxDisp, _sgm_params);

    const int numThreads = 2;

    auto leftImg = _left_image.data();
    auto rightImg = _right_image.data();
    auto dsi = _dsi.data();

    switch(config.censusMask)
    {
      case RSGM::Config::CensusMask::C5x5:
        {
          auto leftImgCensus = _left_image_census.data();
          auto rightImgCensus = _right_image_census.data();

          census5x5_t_SSE(leftImg, leftImgCensus, width, height);
          census5x5_t_SSE(rightImg, rightImgCensus, width, height);
          costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width,
                                       dispCount, _sgm_params.InvalidDispCost, dsi, numThreads);
        } break;

      case RSGM::Config::CensusMask::C9x7:
        {
          _left_image_census64.resize( npixels );
          _right_image_census64.resize( npixels );

          auto leftImgCensus = _left_image_census64.data();
          auto rightImgCensus = _right_image_census64.data();

#pragma omp parallel num_threads(2)
          {
#pragma omp sections nowait
            {
#pragma omp section
              {
                census9x7_t<uint8_t>(leftImg, leftImgCensus, width, height);
              }
#pragma omp section
              {
                census9x7_t<uint8_t>(rightImg, rightImgCensus, width, height);
              }
            }
          }
          costMeasureCensus9x7_xyd_t(leftImgCensus, rightImgCensus, height, width, dispCount, dsi, numThreads);
          THROW_ERROR("NOT done yet");
        } break;
    }

    _disp_image_right.resize( npixels );
    auto dispImgRight = _disp_image_right.data();
    float* output = NULL;

    if(extra) {
      _disp_image_left.resize( npixels );
      output = _disp_image_left.data();
    } else {
      disparity.create( height, width, CV_32FC1 );
      output = disparity.ptr<float32>();
    }

    if(numThreads > 1) {
      sgm.processParallel(dsi, leftImg, output, dispImgRight, numThreads);
    } else {
      sgm.process(dsi, leftImg, output, dispImgRight);
    }

    if(extra) {
      cv::Mat d_tmp(height, width, CV_32FC1, _disp_image_left.data());

      disparity.create(left.size(), CV_32FC1);
      for(int r = 0; r < disparity.rows; ++r)
      {
        auto drow = disparity.ptr<float>(r);
        for(int c = 0; c < extra; ++c)
          drow[c] = 0.0f;

        auto D_tmp_row = d_tmp.ptr<float>(r);
        for(int c = extra; c < disparity.cols; ++c)
          drow[c] = D_tmp_row[c - extra];
      }
    }
  }

 private:
  StereoSGMParams_t _sgm_params;

  typename bpvo::AlignedVector<uint8>::type _left_image;
  typename bpvo::AlignedVector<uint8>::type _right_image;
  typename bpvo::AlignedVector<float32>::type _disp_image_left;
  typename bpvo::AlignedVector<float32>::type _disp_image_right;
  typename bpvo::AlignedVector<uint32>::type _left_image_census;
  typename bpvo::AlignedVector<uint32>::type _right_image_census;
  typename bpvo::AlignedVector<uint16>::type _dsi;

  typename bpvo::AlignedVector<uint64>::type _left_image_census64;
  typename bpvo::AlignedVector<uint64>::type _right_image_census64;
}; // Impl


RSGM::RSGM(Config config) :
    _config(config), _impl(bpvo::make_unique<Impl>()) {}

RSGM::~RSGM() {}

void RSGM::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity)
{
  THROW_ERROR_IF( left.size() != right.size(), "stereo pair size mismatch" );
  THROW_ERROR_IF( left.type() != right.type(), "stereo pair type mismatch" );
  THROW_ERROR_IF( left.type() != CV_8UC1, "input images must be CV_8UC1" );

  //_impl->compute(_config, left, right, disparity);

  int extra = left.cols % 16;
  if(!extra) {
    _impl->compute(_config, left, right, disparity);
  } else {
    // we could copy this stuff directly in impl
    int new_cols = left.cols - extra;
    cv::Mat I0( left.rows, new_cols, left.type());
    cv::Mat I1( right.rows, new_cols, right.type() );

    uint8_t* I0_ptr = I0.ptr<uint8_t>();
    uint8_t* I1_ptr = I1.ptr<uint8_t>();

    for(int r = 0; r < left.rows; ++r)
    {
      auto I0_row = left.ptr<uint8_t>(r);
      auto I1_row = right.ptr<uint8_t>(r);
      for(int c = extra; c < left.cols; ++c)
      {
        *I0_ptr++ = I0_row[c];
        *I1_ptr++ = I1_row[c];
      }
    }

    cv::Mat D_tmp;
    _impl->compute(_config, I0, I1, D_tmp);

    disparity.create(left.size(), CV_32FC1);
    for(int r = 0; r < disparity.rows; ++r)
    {
      auto drow = disparity.ptr<float>(r);
      for(int c = 0; c < extra; ++c)
      {
        drow[c] = 0.0f;
      }

      auto D_tmp_row = D_tmp.ptr<float>(r);
      for(int c = extra; c < disparity.cols; ++c)
      {
        drow[c] = D_tmp_row[c - extra];
      }
    }
  }
}
#endif // WITH_GPL_CODE

