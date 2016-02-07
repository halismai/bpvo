#include "bpvo/pose_estimator_gn.h"
#include "bpvo/template_data_.h"
#include "bpvo/channels.h"

using namespace bpvo;

typedef RawIntensity ChannelsType;
typedef RigidBodyWarp WarpType;

typedef TemplateData_<ChannelsType, WarpType> TemplateData;
typedef PoseEstimatorGN<TemplateData> PoseEstimatorT;

int main()
{

  TemplateData* tdata = nullptr;
  ChannelsType cn;
  Matrix44 T(Matrix44::Identity());

  PoseEstimatorT pose_estimator;
  pose_estimator.run(tdata, cn, T);

  return 0;
}


