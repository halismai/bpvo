#ifndef BPVO_TEMPLATE_DATA_H
#define BPVO_TEMPLATE_DATA_H

#include "bpvo/types.h"

namespace cv {
class Mat;
}; // cv

namespace bpvo {


class TemplateData
{
  typedef Eigen::Matrix<float,1,6> Jacobian;
  typedef typename EigenAlignedContainer<Point>::value_type PointVector;
  typedef typename EigenAlignedContainer<Jacobian>::value_type JacobianVector;
  typedef std::vector<float> PixelVector;

 public:
  /**
   * \param parameters for the algorithm
   * \param K the stereo calibration
   * \param baseline the image baseline
   * \parma pyramid level (0 is the finest) Used to sample the disparity from
   * the high resolution
   */
  TemplateData(const AlgorithmParameters& params, const Matrix33& K, const float& baseline, int pyr_level);


  /**
   * sets the template data from an input image and a disparity map
   *
   * The input image should be scaled correctly for the pyramid level
   * The disparity is always at the highest resolution, we sample it using the
   * pyr_level
   */
  void compute(const cv::Mat& image, const cv::Mat& disparity);

  inline const JacobianVector& jacobians()  const { return _jacobians; }
  inline const PointVector&    points()     const { return _points; }
  inline const PixelVector& pixels() const { return _pixels; }

  inline int numPoints() const { return static_cast<int>(_points.size()); }
  inline int numPixels() const { return static_cast<int>(_pixels.size()); }
  inline int pyrLevel() const { return _pyr_level; }

  void reserve(size_t n);

  /**
   * \return the i-th point
   */
  const typename PointVector::value_type& X(size_t i) const;
  typename PointVector::value_type& X(size_t i);

  /**
   * \return the i-th jacobian
   */
  const typename JacobianVector::value_type& J(size_t i) const;
  typename JacobianVector::value_type& J(size_t i);

  /**
   * \return the i-th pixel value
   */
  const typename PixelVector::value_type& I(size_t i) const;
  typename PixelVector::value_type& I(size_t i);

 protected:
  Matrix33 _K;
  float _baseline;
  int _pyr_level;
  float _sigma_ct;
  float _sigma_bp;

  JacobianVector _jacobians;
  PointVector    _points;
  PixelVector    _pixels;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // TemplateData

}; // bpvo

#endif // BPVO_TEMPLATE_DATA_H
