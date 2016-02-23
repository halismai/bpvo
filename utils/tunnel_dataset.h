#ifndef BPVO_TUNNEL_DATASET_H
#define BPVO_TUNNEL_DATASET_H

#include <utils/dataset.h>

namespace bpvo {

class TunnelDataset : public DisparityDataset
{
 public:
  TunnelDataset(std::string conf_fn);
  virtual ~TunnelDataset();

  inline std::string name() const { return "tunnel"; }
  inline StereoCalibration calibration() const { return _calib; }

 protected:
  StereoCalibration _calib;

  bool loadCalibration(std::string calib_fn);
}; // TunnelDataset

}; // bpvo

#endif // BPVO_TUNNEL_DATASET_H
