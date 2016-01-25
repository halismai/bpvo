#ifndef BPVO_TEMPLATE_DATA_H
#define BPVO_TEMPLATE_DATA_H

#include "bpvo/types.h"

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class DataExtractor;

class TemplateData
{
 public:
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
   * to prevent compiler auto generated
   */
  ~TemplateData();


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

  /**
   * sets the input image (call this before computeResiduals)
   */
  void setInputImage(const cv::Mat&);

  /**
   * compute the error given an input image that was set using setInputImage
   */
  void computeResiduals(const Matrix44& pose, std::vector<float>& residuals,
                        std::vector<uint8_t>& valid) const;

 protected:
  Matrix33 _K;
  float _baseline;
  int _pyr_level;

  JacobianVector _jacobians;
  PointVector    _points;
  PixelVector    _pixels;

  void clear();
  void resize(size_t n);

  struct InputData;
  UniquePointer<InputData> _input_data;

  friend class DataExtractor;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // TemplateData

}; // bpvo

#endif // BPVO_TEMPLATE_DATA_H
