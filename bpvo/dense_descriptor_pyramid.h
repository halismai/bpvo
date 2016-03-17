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
#include <bpvo/dense_descriptor.h>
#include <bpvo/intensity_descriptor.h>
#include <bpvo/bitplanes_descriptor.h>

namespace bpvo {


template <class ... Args> static inline
DenseDescriptor* MakeDescriptor(DescriptorType dtype, Args... args);

/**
 */
class DenseDescriptorPyramid
{
 public:
  /**
   * \param pointer to the descriptor
   * \param ImagePyramid pre-computed image pyramid
   */
  template <class ... Args>
  DenseDescriptorPyramid(DescriptorType dtype, const ImagePyramid& I_pyr, Args... args)
    : _image_pyramid(I_pyr)
  {
    for(int i = 0; i < _image_pyramid.size(); ++i)
      _desc_pyr.push_back(MakeDescriptor(dtype, args...));
  }

  /**
   * \param pointer to descriptor
   * \param number of levels
   * \param image the input image
   */
  template <class ... Args>
  DenseDescriptorPyramid(DescriptorType dtype, int n_levels, const cv::Mat& I, Args...  args)
    : _image_pyramid(n_levels)
  {
    _image_pyramid.compute(I);
    for(int i = 0; i < n_levels; ++i)
      _desc_pyr.push_back(MakeDescriptor(dtype, args...));
  }

  /**
   */
  ~DenseDescriptorPyramid() {}

  /**
   * Computes the descriptor at pyramid level 'i'
   *
   * \param force if true the descriptor will be re-computed. Otherwise, if it
   * had been computed before the function does nothing
   */
  inline void compute(size_t i, bool force = false)
  {
    assert( i < _desc_pyr.size() );
    if(force || _desc_pyr[i]->rows() == 0 || _desc_pyr[i]->cols() == 0)
      _desc_pyr[i]->compute(_image_pyramid[i]);
  }

  /**
   * \return the descriptor at level 'i'
   */
  inline const DenseDescriptor* operator[](size_t i) const
  {
    assert( i < _desc_pyr.size() );
    return _desc_pyr[i].get();
  }

 protected:
  typedef UniquePointer<DenseDescriptor> DenseDescriptorPointer;
  std::vector<DenseDescriptorPointer> _desc_pyr;
  ImagePyramid _image_pyramid;
}; // DenseDescriptorPyramid

}; // bpvo

#endif // BPVO_DENSE_DESCRIPTOR_PYRAMID_H
