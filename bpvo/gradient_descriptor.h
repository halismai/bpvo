/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#ifndef BPVO_GRADIENT_DESCRIPTOR_H
#define BPVO_GRADIENT_DESCRIPTOR_H

#include <bpvo/dense_descriptor.h>
#include <opencv2/core/core.hpp>
#include <array>

namespace bpvo {

class GradientDescriptor : public DenseDescriptor
{
 public:
  /**
   * Intensity + gradient information
   *
   * \param sigma an optional smoothing prior to computing gradients
   */
  GradientDescriptor(float sigma = -1.0);

  /**
   * copy ctor
   */
  GradientDescriptor(const GradientDescriptor& other);

  virtual ~GradientDescriptor();

  void compute(const cv::Mat&);

  inline const cv::Mat& getChannel(int i) const { return _channels[i]; }
  inline int numChannels() const { return 3; }
  inline int rows() const { return _rows; }
  inline int cols() const { return _cols; }

  inline void setSigma(float s) { _sigma = s; }

  inline Pointer clone() const
  {
    return Pointer(new GradientDescriptor(*this));
  }

 private:
  int _rows, _cols;
  float _sigma;
  std::array<cv::Mat,3> _channels;
}; // GradientDescriptor

class LaplacianDescriptor : public DenseDescriptor
{
 public:
  inline LaplacianDescriptor(int ks = 1)
      : DenseDescriptor(), _kernel_size(ks) {}

  inline LaplacianDescriptor(const LaplacianDescriptor& other)
      : DenseDescriptor(other)
        , _kernel_size(other._kernel_size)
        , _desc(other._desc) {}

  virtual ~LaplacianDescriptor() {}

  void compute(const cv::Mat&);

  inline int numChannels() const { return 1; }
  inline int rows() const { return _desc.rows; }
  inline int cols() const { return _desc.cols; }
  inline const cv::Mat& getChannel(int) const { return _desc; }

  inline Pointer clone() const
  {
    return Pointer(new LaplacianDescriptor(*this));
  }

 private:
  float _kernel_size;
  cv::Mat _desc;
}; // LaplacianDescriptor

/**
 */
class DescriptorFields : public DenseDescriptor
{
 public:
  /**
   * \param sigma1 applied before computing gradients
   * \param sigma2 applied after creating channels
   */
  DescriptorFields(float sigma1 = 0.5, float sigma2 = 1.5);
  DescriptorFields(const DescriptorFields&);
  virtual ~DescriptorFields();

  void compute(const cv::Mat&);

  inline const cv::Mat& getChannel(int i) const { return _channels[i]; }
  inline int numChannels() const { return 5; }
  inline int rows() const { return _channels[0].rows; }
  inline int cols() const { return _channels[0].cols; }

  inline void setSigma1(float s) { _sigma1 = s; }
  inline void setSigma2(float s) { _sigma2 = s; }

  inline Pointer clone() const { return Pointer(new DescriptorFields(*this)); }

 private:
  int _rows, _cols;
  float _sigma1, _sigma2;
  std::array<cv::Mat, 5> _channels;
}; // DescriptorFields

class DescriptorFields2ndOrder : public DenseDescriptor
{
 public:
  DescriptorFields2ndOrder(float sigma1 = 0.5, float sigma2 = 1.5);
  DescriptorFields2ndOrder(const DescriptorFields2ndOrder&);
  virtual ~DescriptorFields2ndOrder();

  void compute(const cv::Mat&);

  inline const cv::Mat& getChannel(int i) const { return _channels[i]; }

  inline int numChannels() const { return 10; }
  inline int rows() const { return _rows; }
  inline int cols() const { return _cols; }

  inline void setSigma1(float s) { _sigma1 = s; }
  inline void setSigma2(float s) { _sigma2 = s; }

  inline Pointer clone() const { return Pointer(new DescriptorFields2ndOrder(*this)); }

 private:
  int _rows, _cols;
  float _sigma1, _sigma2;
  std::array<cv::Mat, 10> _channels;
}; // DescriptorFields2ndOrder

}; // bpvo

#endif // BPVO_GRADIENT_DESCRIPTOR_H
