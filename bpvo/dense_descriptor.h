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

#ifndef BPVO_DENSE_DESCRIPTOR_H
#define BPVO_DENSE_DESCRIPTOR_H

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class DenseDescriptor
{
 public:
  DenseDescriptor();
  virtual ~DenseDescriptor();

  /**
   * computes the channels/descriptors
   */
  virtual void compute(const cv::Mat&) = 0;

  /**
   * computes the saliency map
   */
  virtual void computeSaliencyMap(cv::Mat&) const = 0;

  /**
   * \return the i-th channel
   */
  virtual const cv::Mat& getChannel(int i) const = 0;

  /**
   * \return the number of channels
   */
  virtual int numChannels() const = 0;

  virtual int rows() const = 0;
  virtual int cols() const = 0;

}; // DenseDescriptor


}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_H
