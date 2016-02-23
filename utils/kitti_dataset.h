#ifndef BPVO_UTILS_KITTI_DATASET_H
#define BPVO_UTILS_KITTI_DATASET_H

#include <utils/dataset.h>

namespace bpvo {

class KittiDataset : public StereoDataset
{
 public:
  KittiDataset(std::string conf_fn);
  virtual ~KittiDataset();

  inline std::string name() const { return "kitti"; }
  inline StereoCalibration calibration() const { return _calib; }

 protected:
  StereoCalibration _calib;

  bool init(const ConfigFile&);
  bool loadCalibration(std::string calib_fn);
}; // KittiDataset

}; // bpvo

#endif // BPVO_UTILS_KITTI_DATASET_H
