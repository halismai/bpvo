#ifndef BPVO_DMV_PHOTO_BUNDLE_H
#define BPVO_DMV_PHOTO_BUNDLE_H

#include <bpvo/types.h>
#include <bpvo/trajectory.h>
#include <dmv/photo_bundle_config.h>

#include <iosfwd>

namespace cv { class Mat; }; // cv

namespace bpvo {

class VoOutput;

namespace dmv {

class PhotoBundle
{
 public:
  typedef typename EigenAlignedContainer<Point>::type PointVector;
  typedef std::vector<float> WeightsVector;

  struct Result
  {
    Trajectory trajectory;  //< optimized camera poses
    PointVector points;     //< optimized point in world frame

    std::string termMessage; //< message of why the solver terminated

    double initialCost = -1.0; //< initial cost
    double finalCost = -1.0;   //< the final cost

    double timeInSeconds = -1.0;  //< time it took in seconds

    int numIterations = -1;   //< number of iterations it took the solver

    friend std::ostream& operator<<(std::ostream&, const Result&);
  }; // Result

 public:
  PhotoBundle(const Matrix33& K, const PhotoBundleConfig& config);
  ~PhotoBundle();

  void addData(const VoOutput* vo_output);

  /**
   * Runs the optimization
   */
  Result optimize();

 private:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // PhotoBundle

}; // dmv
}; // bpvo

#endif // BPVO_DMV_PHOTO_BUNDLE_H
