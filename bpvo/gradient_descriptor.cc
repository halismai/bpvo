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

#include "bpvo/gradient_descriptor.h"
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <type_traits>
#include <Eigen/Core>

namespace bpvo {

GradientDescriptor::GradientDescriptor(float s)
    : _rows(0), _cols(0), _sigma(s) {}

GradientDescriptor::GradientDescriptor(const GradientDescriptor& other)
    : DenseDescriptor(other), _rows(other._rows), _cols(other._cols),
    _sigma(other._sigma), _channels(other._channels) {}

GradientDescriptor::~GradientDescriptor() {}

void GradientDescriptor::compute(const cv::Mat& image)
{
  _rows = image.rows;
  _cols = image.cols;

  image.convertTo(_channels[0], CV_32F);

  //
  // we will keep the original intennsities unsmoothed, the smoothing will
  // affect the gradient computation only
  //
  cv::Mat I;
  if(_sigma > 0)
    cv::GaussianBlur(_channels[0], I, cv::Size(), _sigma, _sigma);
  else
    I = _channels[0];

  _channels[1].create(_rows, _cols, CV_32FC1);
  xgradient(I.ptr<float>(), _rows, _cols, _channels[1].ptr<float>());

  _channels[2].create(_rows, _cols, CV_32FC1);
  ygradient(I.ptr<float>(), _rows, _cols, _channels[2].ptr<float>());
}

DescriptorFields::DescriptorFields(float s1, float s2)
  : _sigma1(s1),  _sigma2(s2) {}

DescriptorFields::DescriptorFields(const DescriptorFields& other)
  : DenseDescriptor(other), _rows(other._rows), _cols(other._cols),
    _sigma1(other._sigma1), _sigma2(other._sigma2), _channels(other._channels) {}

DescriptorFields::~DescriptorFields() {}

static void splitPosNeg(const cv::Mat& src, cv::Mat& pos, cv::Mat& neg, float sigma)
{
  pos.create(src.size(), CV_32FC1);
  neg.create(src.size(), CV_32FC1);

  for(int r = 0; r < src.rows; ++r) {
    auto srow = src.ptr<float>(r);
    auto prow = pos.ptr<float>(r), nrow = neg.ptr<float>(r);
    for(int c = 0; c < src.cols; ++c) {
      prow[c] = srow[c] >= 0 ? srow[c] : 0.0f;
      nrow[c] = srow[c] < 0  ? srow[c] : 0.0f;
    }
  }

  if(sigma > 0.0f) {
    cv::GaussianBlur(pos, pos, cv::Size(), sigma, sigma);
    cv::GaussianBlur(neg, neg, cv::Size(), sigma, sigma);
  }
}

void DescriptorFields::compute(const cv::Mat& image)
{
  _rows = image.rows;
  _cols = image.cols;

  image.convertTo(_channels[0], CV_32F);

  cv::Mat I;
  if(_sigma1 > 0.0)
    cv::GaussianBlur(_channels[0], I, cv::Size(), _sigma1, _sigma1);
  else
    I = _channels[0];

  cv::Mat buffer(_channels[0].size(), CV_32FC1);

  xgradient(I.ptr<float>(), _rows, _cols, buffer.ptr<float>());
  splitPosNeg(buffer, _channels[1], _channels[2], _sigma2);

  ygradient(I.ptr<float>(), _rows, _cols, buffer.ptr<float>());
  splitPosNeg(buffer, _channels[3], _channels[4], _sigma2);
}

} // bpvo

