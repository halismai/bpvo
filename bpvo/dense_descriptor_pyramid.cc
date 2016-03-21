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


#include <bpvo/dense_descriptor_pyramid.h>
#include <bpvo/dense_descriptor.h>
#include <bpvo/intensity_descriptor.h>
#include <bpvo/bitplanes_descriptor.h>
#include <bpvo/utils.h>
#include <bpvo/image_pyramid.h>

namespace bpvo {

struct DenseDescriptorPyramid::Impl
{
  inline Impl(const AlgorithmParameters& p)
      : _max_test_level(p.maxTestLevel)
  {
    THROW_ERROR_IF( p.numPyramidLevels <= 0, "invalid number of pyramid levels" );
    THROW_ERROR_IF( p.maxTestLevel < 0, "invalid maxTestLevel" );

    for(int i = 0; i < p.numPyramidLevels; ++i)
      _desc_pyr.push_back(UniquePointer<DenseDescriptor>(DenseDescriptor::Create(p, i)));
  }

  inline const DenseDescriptor* operator[](size_t i) const
  {
    return _desc_pyr[i].get();
  }

  DenseDescriptor* operator[](size_t i) { return _desc_pyr[i].get(); }

  inline void copy(Impl& other) const
  {
    // copy operations should be called on the pyramids of the same level to
    // minimize memory allocations/frees
    THROW_ERROR_IF( other._max_test_level != _max_test_level, "maxTestLevel mismatch" );
    THROW_ERROR_IF( _desc_pyr.size() != other._desc_pyr.size(), "size mismatch" );

    for(int i = _desc_pyr.size()-1; i >= _max_test_level; --i)
    {
      // by class design, DenseDescriptor pointers in 'other' will be allocated
      // but we'll check anyways
      assert( other._desc_pyr[i] != nullptr );
      _desc_pyr[i]->copyTo( other._desc_pyr[i].get() );
    }
  }

  inline void init(const ImagePyramid& image_pyramid)
  {
    for(int i = image_pyramid.size()-1; i >= _max_test_level; --i)
      _desc_pyr[i]->compute(image_pyramid[i]);
  }

  inline void init(const cv::Mat& image)
  {
    ImagePyramid image_pyramid(_desc_pyr.size());
    image_pyramid.compute(image);
    init(image_pyramid);
  }

  inline int size() const { return static_cast<int>(_desc_pyr.size()); }

  int _max_test_level;
  std::vector<UniquePointer<DenseDescriptor>> _desc_pyr;
}; // DenseDescriptorPyramid::Impl

DenseDescriptorPyramid::DenseDescriptorPyramid(const AlgorithmParameters& p)
  : _impl(make_unique<Impl>(p)) {}

DenseDescriptorPyramid::
DenseDescriptorPyramid(DenseDescriptorPyramid&& o) noexcept
  : _impl(std::move(o._impl)) {}

DenseDescriptorPyramid::~DenseDescriptorPyramid() {}

void DenseDescriptorPyramid::init(const cv::Mat& image)
{
  _impl->init(image);
}

void DenseDescriptorPyramid::init(const ImagePyramid& image_pyramid)
{
  _impl->init(image_pyramid);
}

const DenseDescriptor* DenseDescriptorPyramid::operator[](size_t i) const
{
  assert( i < (size_t) size() );
  return _impl->operator[](i);
}

DenseDescriptor* DenseDescriptorPyramid::operator[](size_t i)
{
  assert( i < (size_t) size() );
  return _impl->operator[](i);
}

void DenseDescriptorPyramid::copyTo(DenseDescriptorPyramid& other) const
{
  _impl->copy(*other._impl);
}

int DenseDescriptorPyramid::size() const { return _impl->size(); }


} // bpvo

