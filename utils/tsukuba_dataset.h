#ifndef BPVO_UTILS_TSUKUBA_DATASET_H
#define BPVO_UTILS_TSUKUBA_DATASET_H

#include <utils/dataset.h>

namespace bpvo {

/**
 * uses the groundtruth disparities
 */
class TsukubaSyntheticDataset : public DisparityDataset
{
 public:
  TsukubaSyntheticDataset(std::string);
  virtual ~TsukubaSyntheticDataset();

  inline std::string name() const { return "tsukuba_synthetic"; }
  inline StereoCalibration calibration() const { return _calib; }

 protected:
  StereoCalibration _calib;

  virtual bool init(const ConfigFile&);
};

class TsukubaStereoDataset : public StereoDataset
{
 public:
  TsukubaStereoDataset(std::string conf_fn);
  virtual ~TsukubaStereoDataset();

  inline std::string name() const { return "tsukuba_stereo"; }
  inline StereoCalibration calibration() const { return _calib; }

 protected:
  StereoCalibration _calib;

  virtual bool init(const ConfigFile& cf);
}; // TsukubaSyntheticDataset


}; // bpvo

#endif // BPVO_UTILS_TSUKUBA_DATASET_H
