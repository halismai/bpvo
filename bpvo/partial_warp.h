#ifndef BPVO_PARTIAL_WARP
#define BPVO_PARTIAL_WARP

#include <bpvo/types.h>
#include <bpvo/template_data.h>

namespace bpvo {

/**
 * Transforms the points, determine if they are valid and pre-compute the
 * interpolation weights
 *
 * \param P 3x3 projection matrix = K*[R t]
 * \param xyzw 4xn_pts list of 3D points
 * \param coeffs 4xn interpolation coefficients
 * \param valid true if the point is valid (projects onto the image)
 */
void computeInterpolationData(const Matrix34& P, const typename TemplateData::PointVector& xyzw,
                              int rows, int cols, typename TemplateData::PointVector& interp_coeffs,
                              typename EigenAlignedContainer<Eigen::Vector2i>::value_type& uv,
                              std::vector<uint8_t>& valid);

}; // bpvo

#endif // BPVO_PARTIAL_WARP
