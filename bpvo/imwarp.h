#ifndef BPVO_IMWARP_H
#define BPVO_IMWARP_H

#include <cstdint>
#include <bpvo/types.h>

namespace bpvo {

struct ImageSize;

/**
 * \param the image size
 * \param T 4x4 projection matrix (must be ColMajor)           [4x4]
 * \param xyz pointer to N 3D points in homogenous coordinates [4xN]
 * \param N number of points
 * \param inds output indicies                                 [1xN]
 * \param valid true if the i-th point projects onto the image [1xN]
 * \param coeffs interpolation coeffc                          [4xN]
 */
void imwarp_precomp(const ImageSize&, const float* T, const float* xyzw,
                    int N, int* inds, typename ValidVector::value_type* valid,
                    float* coeffs);

}; // bpvo

#endif // BPVO_IMWARP_H
