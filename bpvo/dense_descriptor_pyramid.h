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

#ifndef BPVO_DENSE_DESCRIPTOR_PYRAMID_H
#define BPVO_DENSE_DESCRIPTOR_PYRAMID_H

#include <bpvo/types.h>
#include <bpvo/image_pyramid.h>

namespace bpvo {

class DenseDescriptor;

DenseDescriptor* MakeDescriptor(DescriptorType dtype, const AlgorithmParameters&);

/**
 */
class DenseDescriptorPyramid
{
 public:
  /**
   * \param pointer to the descriptor
   * \param ImagePyramid pre-computed image pyramid
   */
  DenseDescriptorPyramid(DescriptorType dtype, const ImagePyramid& I_pyr,
                         const AlgorithmParameters& = AlgorithmParameters());

  /**
   * \param pointer to descriptor
   * \param number of levels
   * \param image the input image
   */
  DenseDescriptorPyramid(DescriptorType dtype, int n_levels, const cv::Mat&,
                         const AlgorithmParameters& = AlgorithmParameters());

  /**
   */
  ~DenseDescriptorPyramid();

  /**
   * deep copies the thing
   */
  DenseDescriptorPyramid(const DenseDescriptorPyramid&);

  /**
   * move
   */
  DenseDescriptorPyramid(DenseDescriptorPyramid&&);

  /**
   * Computes the descriptor at pyramid level 'i'
   *
   * \param force if true the descriptor will be re-computed. Otherwise, if it
   * had been computed before the function does nothing
   */
  void compute(size_t i, bool force = false);

  /**
   * sets the image for the pyramid (does not compute descriptors)
   */
  void setImage(const cv::Mat& image);

  /**
   * \return the descriptor at level 'i'
   */
  const DenseDescriptor* operator[](size_t i) const;

  /**
   * \return the number of pyramid levels
   */
  inline int size() const { return static_cast<int>(_desc_pyr.size()); }

  inline const ImagePyramid& getImagePyramid() const { return _image_pyramid; }

 protected:
  typedef UniquePointer<DenseDescriptor> DenseDescriptorPointer;
  std::vector<DenseDescriptorPointer> _desc_pyr;
  ImagePyramid _image_pyramid;

}; // DenseDescriptorPyramid
}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_PYRAMID_H

