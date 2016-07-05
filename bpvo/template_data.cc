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

#include "bpvo/template_data.h"
#include "bpvo/dense_descriptor.h"
#include "bpvo/imgproc.h"
#include "bpvo/parallel.h"
#include "bpvo/utils.h"

namespace bpvo {

TemplateData::TemplateData(int pyr_level, const Matrix33& K, float b,
                           const AlgorithmParameters& p)
  : _pyr_level(pyr_level), _params(p), _warp(K, b), _photo_error(p.interp)
{
  THROW_ERROR_IF( _pyr_level < 0, "pyramid level must be >= 0" );
}

void TemplateData::setData(const DenseDescriptor* desc, const cv::Mat& D)
{
  cv::Mat saliency_map;
  desc->computeSaliencyMap(saliency_map);

  int rows = desc->rows(), cols = desc->cols();
  IsLocalMax<float> is_local_max(nullptr, cols, -1);
  if(rows*cols >= _params.minNumPixelsForNonMaximaSuppression)
  {
    is_local_max.setPointer( saliency_map.ptr<const float>() );
    is_local_max.setStride( saliency_map.cols );
    is_local_max.setRadius( _params.nonMaxSuppRadius );
  }

  const int border = std::max(_params.nonMaxSuppRadius, 3);

  std::vector<uint16_t> inds;
  inds.reserve( saliency_map.rows * saliency_map.cols * 0.5 );
  for(int y = border; y < saliency_map.rows - border - 1; ++y)
  {
    auto srow = saliency_map.ptr<const float>(y);
    for(int x = border; x < saliency_map.cols - border - 1; ++x)
    {
      if(srow[x] >= _params.minSaliency && is_local_max(y,x))
      {
        inds.push_back(y);
        inds.push_back(x);
      }
    }
  }

  std::vector<int> valid_inds;
  valid_inds.reserve(inds.size()/2);

  _points.resize(0);
  _points.reserve(inds.size()/2);
  auto D_ptr = D.ptr<const float>();
  for(size_t i = 0; i < inds.size(); i += 2)
  {
    int y = inds[i + 0], x = inds[i + 1];
    auto d = D_ptr[ (1 << _pyr_level) * (y*D.cols + x) ];
    if(d >= _params.minValidDisparity && d <= _params.maxValidDisparity)
    {
      _points.push_back( _warp.makePoint(x, y, d) );
      valid_inds.push_back( y*cols + x );
    }
  }

  int extra = _points.size() % 16;
  if(extra) {
    _points.erase( _points.end() - extra, _points.end() );
    valid_inds.erase(valid_inds.end() - extra, valid_inds.end());
  }

  if(_params.withNormalization)
    _warp.setNormalization(_points);

  int num_points = _points.size();
  int num_channels = desc->numChannels();
  _pixels.resize( num_channels * num_points );
  _jacobians.resize( num_channels * num_points );

  dprintf("\nnum_points %d (%d) [level %d] %f\n",
         num_points, (int) inds.size()/2, _pyr_level, _params.minSaliency);

  constexpr float NN = 1.0f / 18.0f;

  typename AlignedVector<float>::type IxIy(2*num_points);
  for(int c = 0; c < num_channels; ++c)
  {
    auto c_ptr = desc->getChannel(c).ptr<const float>();
    auto P_ptr = _pixels.data() + c*num_points;

    for(int i = 0; i < num_points; ++i)
    {
      auto ii = valid_inds[i];
      P_ptr[i] = c_ptr[ii];

      auto* cc = c_ptr + ii;

      switch(_params.gradientEstimation)
      {
        case kCentralDifference_3:
          {
            IxIy[2*i+0] = 0.5f * ( *(cc+1) - *(cc-1) );
            IxIy[2*i+1] = 0.5f * ( *(cc+cols) - *(cc-cols) );
          } break;

        case kCentralDifference_5:
          {
            IxIy[2*i+0] = NN * (1.0f*cc[-2] - 8.0f*cc[-1] + 8.0f*cc[1] - 1.0f*cc[2]);
            IxIy[2*i+1] = NN * (1.0f*cc[-2*cols] - 8.0f*cc[-1*cols] + 8.0f*cc[+1*cols] - 1.0f*cc[2*cols]);
          } break;
      }
    }

    auto J_ptr = _jacobians.data() + c*num_points;
    int i = _warp.computeJacobian(_points, IxIy.data(), J_ptr->data());
    for( ; i < num_points; ++i)
      _warp.jacobian(_points[i], IxIy[2*i+0], IxIy[2*i+1], J_ptr[i].data());
  }

  // NOTE: we push an empty Jacobian at the end because of SSE code loading
  // We won't need to this when switching to Vector6
  _jacobians.push_back(Jacobian::Zero());
}

namespace {

struct ComputeResidualsBody : public ParallelForBody
{
 public:
  ComputeResidualsBody(const DenseDescriptor* desc, const PhotoError& photo_error,
                       int num_points, const float* pixels, float* residuals)
      : ParallelForBody(), _desc(desc), _photo_error(photo_error)
      , _num_points(num_points), _pixels(pixels), _residuals(residuals) {}

  inline void operator()(const Range& range) const
  {
    for(int c = range.begin(); c != range.end(); ++c)
    {
      int off = c*_num_points;
      const float* I1_ptr = _desc->getChannel(c).ptr<const float>();
      _photo_error.run(_pixels + off, I1_ptr, _residuals + off);
    }
  }

 protected:
  const DenseDescriptor* _desc;
  const PhotoError& _photo_error;
  const int _num_points;
  const float* _pixels;
  float* _residuals;
}; // ComputeResidualsBody

}; // namespace

void TemplateData::computeResiduals(const DenseDescriptor* desc, const Matrix44& pose,
                                    ResidualsVector& residuals, ValidVector& valid) const
{
  THROW_ERROR_IF( numPoints() == 0, "you should call setData before calling computeResiduals" );
  _warp.setPose(pose);

  valid.resize(_points.size());
  residuals.resize(_pixels.size());

  _photo_error.init(_warp.P(), _points, valid, desc->rows(), desc->cols());

  ComputeResidualsBody func(desc, _photo_error, _points.size(),
                            _pixels.data(), residuals.data());

  parallel_for(Range(0, desc->numChannels()), func);
}

}; // bpvo

