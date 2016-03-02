#include "dmv/photo_vo_base.h"
#include "bpvo/utils.h"
#include "bpvo/imgproc.h"

#include <Eigen/LU>
#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

PhotoVoBase::PhotoVoBase(const Mat_<double,3,3>& K, double b, Config config)
    : _K(K), _K_inv(_K.inverse()), _baseline(b), _config(config) {}

void PhotoVoBase::setTemplate(const cv::Mat& I_, const cv::Mat& D_)
{
  THROW_ERROR_IF( I_.type() != CV_8UC1, "image must be CV_8UC1" );
  THROW_ERROR_IF( D_.type() != CV_32FC1, "disparity must be CV_32FC1" );

  //
  // clear the old points
  //
  _points.resize(0);

  const cv::Mat_<uint8_t>& I = (const cv::Mat_<uint8_t>&) I_;
  const cv::Mat_<float>& D  = (const cv::Mat_<float>&) D_;

  auto Ix = [&](int y, int x) {
    return static_cast<int16_t>(I(y,x+1)) - static_cast<int16_t>(I(y,x-1));
  }; // Ix

  auto Iy = [&](int y, int x) {
    return static_cast<int16_t>(I(y+1,x)) - static_cast<int16_t>(I(y-1,x));
  }; // Iy

  double bf = _baseline * _K(0,0);

  switch( _config.pixelSelection )
  {
    case PhotoVoBase::Config::PixelSelectorType::None:
      {
        //
        // no pixel selection will happen
        // we will use all points with gradient mag > minSaliency
        //
        for(int y = 1; y < I.rows - 2; ++y)
          for(int x = 1; x < I.cols - 2; ++x)
            if( (std::abs(Ix(y,x)) + std::abs(Iy(y,x))) > _config.minSaliency )
              if(D(y,x) > 0.01f)
                _points.push_back(WorldPoint(_K_inv, Vec_<double,2>(x,y), bf / D(y,x)));
      } break;

    case PhotoVoBase::Config::PixelSelectorType::LocalAbsGradMax:
      {
        //
        // pixel selection based on non-maxima suppresion of the absolute
        // gradient magnitutde
        //
        cv::Mat_<short> G(I.size());
        for(int y = 1; y < I.rows - 2; ++y)
        {
          auto srow0 = I.ptr<uint8_t>(y-1),
               srow1 = I.ptr<uint8_t>(y+1),
               srow  = I.ptr<uint8_t>(y);
          auto grow = G.ptr<short>(y);
          for(int x = 1; x < I.cols - 2; ++x)
          {
            grow[x] = std::abs(srow[x+1] - srow[x-1]) + std::abs(srow1[x] - srow0[x]);
          }
        }

        const int radius = _config.nonMaxSuppRadius;
        THROW_ERROR_IF(radius < 1, "nonMaxSuppRadius must be > 0");

        for(int y = radius; y < I.rows - radius - 1; ++y)
        {
          for(int x = radius; x < I.cols - radius - 1; ++x)
          {
            if(D(y,x) > 0.0f)
            {
              if(G(y,x) > _config.minSaliency)
              {
                bool is_local_max = true;
                for(int r = -radius; r <= radius; ++r)
                  for(int c = -radius; c <= radius; ++c)
                    if(!(!r && !c) && G(y+r,x+c) > G(y,x))
                    {
                      is_local_max = false;
                      break;
                    }

                if(is_local_max)
                {
                  _points.push_back(WorldPoint(_K_inv, Vec_<double,2>(x,y), bf / D(y,x)));
                }
              }
            }
          }
        }

      } break;
  }

  this->setImageData(I, D);
}

} // dmv
} // bpvo
