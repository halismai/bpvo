#ifndef BPVO_IMWARP_H
#define BPVO_IMWARP_H

#include <cstdint>

namespace bpvo {

struct ImageSize;

/**
 * \param the image size
 * \param P 3x4 projection matrix
 * \param xyz pointer to N 3D points in homogenous coordinates
 * \param N number of points
 * \param inds output indicies
 * \param valid true if the i-th point projects onto the image
 * \param coeffs interpolation coeffc
 */
void imwarp_precomp(const ImageSize&, const float* P, const float* xyzw,
                    int N, int* inds, uint8_t* valid, float* coeffs);

}; // bpvo

#endif // BPVO_IMWARP_H
