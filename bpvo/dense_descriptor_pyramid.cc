#include <bpvo/dense_descriptor_pyramid.h>
#include <bpvo/dense_descriptor.h>
#include <bpvo/intensity_descriptor.h>
#include <bpvo/bitplanes_descriptor.h>
#include <bpvo/utils.h>

namespace bpvo {

DenseDescriptor*
MakeDescriptor(DescriptorType dtype, const AlgorithmParameters& p, int pyr_level)
{
  switch(dtype)
  {
    case DescriptorType::kIntensity:
      return new IntensityDescriptor();

    case DescriptorType::kBitPlanes:
      {
        float sigma_ct = p.sigmaPriorToCensusTransform;
        // do not smooth the bit-planes at low pyramid level, it destroys all
        // the infrmation and prevents us from handling large motions
        float sigma_bp = pyr_level >= p.maxTestLevel ? p.sigmaBitPlanes : -1.0f;
        return new BitPlanesDescriptor(sigma_ct, sigma_bp);
      } break;

    default:
      THROW_ERROR("unkonwn DescriptorType\n");
  }
}

DenseDescriptorPyramid::
DenseDescriptorPyramid(DescriptorType dtype, const ImagePyramid& I_pyr,
                       const AlgorithmParameters& params)
  : _image_pyramid(I_pyr)
{
  for(int i = 0; i < _image_pyramid.size(); ++i) {
    _desc_pyr.push_back(UniquePointer<DenseDescriptor>(
            MakeDescriptor(dtype, params, i)));
    _desc_pyr[i]->setHasData(false);
  }
}

DenseDescriptorPyramid::
DenseDescriptorPyramid(DescriptorType dtype, int n_levels, const cv::Mat& I,
                       const AlgorithmParameters& p)
  : _image_pyramid(n_levels)
{
  _image_pyramid.compute(I);
  for(int i = 0; i < n_levels; ++i)
    _desc_pyr.push_back(UniquePointer<DenseDescriptor>(
            MakeDescriptor(dtype, p, i)));
}

DenseDescriptorPyramid::
DenseDescriptorPyramid(const DenseDescriptorPyramid& other)
  :  _image_pyramid(other._image_pyramid)
{
  for(size_t i = 0; i < other._desc_pyr.size(); ++i) {
    _desc_pyr.push_back(other._desc_pyr[i]->clone());
  }
}

DenseDescriptorPyramid::
DenseDescriptorPyramid(DenseDescriptorPyramid&& o) noexcept
  : _desc_pyr(std::move(o._desc_pyr))
  , _image_pyramid(std::move(o._image_pyramid)) {}

DenseDescriptorPyramid::~DenseDescriptorPyramid() {}


void DenseDescriptorPyramid::compute(size_t i, bool force)
{
  assert( i < _desc_pyr.size() );
  if(force || !_desc_pyr[i]->hasData()) {
    _desc_pyr[i]->compute(_image_pyramid[i]);
    _desc_pyr[i]->setHasData(true);
  }
}

void DenseDescriptorPyramid::setImage(const cv::Mat& image)
{
  _image_pyramid.compute(image);
  for(size_t i = 0; i < _desc_pyr.size(); ++i)
    _desc_pyr[i]->setHasData(false);
}

const DenseDescriptor* DenseDescriptorPyramid::operator[](size_t i) const
{
  assert( i < _desc_pyr.size() );
  return _desc_pyr[i].get();
}

} // bpvo
