#ifndef BPVO_DMV_PHOTO_VO_WITH_DEPTH_H
#define BPVO_DMV_PHOTO_VO_WITH_DEPTH_H

#include <bpvo/types.h>

namespace cv {
class Mat;
}; // cv

namespace bpvo {
namespace dmv {

/**
 * Estimates the pose of the camera and refines the depth as well
 */
class PhotoVoWithDepth
{
 public:
  struct Config
  {
    int nonMaxSuppRadius = 2;
    int maxIterations = 200;
    int patchRadius = 1; // 3x3 patch
    short minSaliency = 1;
    bool withRobust = true;
  }; // Config
}; // PhotoVoWithDepth

}; // dmv
}; // bpvo

#endif // BPVO_DMV_PHOTO_VO_WITH_DEPTH_H
