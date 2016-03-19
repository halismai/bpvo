#include "dataset/dataset.h"
#include "utils/stereo_algorithm.h"
#include "utils/stereo_calibration.h"
#include "utils/file_loader.h"
#include "bpvo/config_file.h"

#include <opencv2/highgui/highgui.hpp>

namespace bpvo {
namespace dataset {

struct Dataset::Impl
{
  Impl(std::string conf_fn)
  {
    init(ConfigFile(conf_fn));
  }

  void init(const ConfigFile& cf);
}; // Impl

Dataset::Dataset(std::string conf_fn)
    : _impl(make_unique<Impl>(conf_fn))
{
  loadCalibration(ConfigFile(conf_fn));
}

Dataset::~Dataset() {}

}; // dataset
}; // bpvo
