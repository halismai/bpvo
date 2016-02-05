#ifndef LINEAR_SYSTEM_BUILDER_H
#define LINEAR_SYSTEM_BUILDER_H

#include <bpvo/template_data.h>
#include <bpvo/types.h>

namespace bpvo {

class LinearSystemBuilder
{
 public:
  typedef typename TemplateData::JacobianVector JacobianVector;
  typedef typename TemplateData::PixelVector    ResidualsVector;
  typedef typename TemplateData::Jacobian       Jacobian;
  typedef std::vector<uint8_t> ValidVector;

  typedef Eigen::Matrix<float, 6, 1> Gradient;
  typedef Eigen::Matrix<float, 6, 6> Hessian;

 public:

  /**
   * \param J the jacobians per pixel
   * \param R the residuals
   * \param weighted M-estimator weights
   * \param valid incidates which elements are valid. Invalid elements are the
   * ones that project outside the image and we do not need to add them to the
   * optimization. NOTE: their weight will be 0
   *
   * \param H = J'*W*J
   * \param G = J'*W*R
   * \return the norm of the weighted residuals
   */
  static float Run(const JacobianVector& J, const ResidualsVector& R, const ResidualsVector& weights,
                   const ValidVector& valid, Hessian* = nullptr, Gradient* = nullptr);
}; // LinearSystemBuilder

}; // bpvo

#endif // LINEAR_SYSTEM_BUILDER_H
