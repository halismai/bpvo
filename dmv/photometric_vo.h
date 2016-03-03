#ifndef BPVO_DMV_PHOTOMETRIC_VO_H
#define BPVO_DMV_PHOTOMETRIC_VO_H

#include <bpvo/types.h>
#include <bpvo/eigen.h>
#include <dmv/photometric_vo_config.h>

namespace bpvo {

class VoOutput;

namespace dmv {

/**
 */
class PhotometricVo
{
 public:
  typedef Mat_<double,3,3> Mat33;
  typedef Mat_<double,4,4> Mat44;
  typedef Vec_<double,3> Point3;
  typedef EigenAlignedContainer<Point3>::type  Point3Vector;

 public:
  struct Result
  {
    /**
     * estimated pose
     */
    Mat44 pose;

    /**
     * number of iterations
     */
    int numIterations;

    /**
     * refined 3D points
     */
    Point3Vector points;

    /**
     */
    Point3Vector pointsRaw;
  }; // Result

 public:
  /**
   * K camera matrix
   * b stereo baseline
   * config algorithm configurations
   */
  PhotometricVo(const Mat33& K, double b, PhotometricVoConfig config);

  /**
   */
  ~PhotometricVo();

  Result addFrame(const VoOutput*);

 protected:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // PhotometricVo

}; // dmv
}; // bpvo

#endif // BPVO_DMV_PHOTOMETRIC_VO_H
