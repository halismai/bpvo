#include "utils/dataset.h"
#include "utils/tunnel_dataset.h"
#include "utils/kitti_dataset.h"
#include "utils/tsukuba_dataset.h"

#include "bpvo/config_file.h"
#include "bpvo/utils.h"

namespace bpvo {

UniquePointer<Dataset> Dataset::Create(std::string conf_fn)
{
  ConfigFile cf(conf_fn);

  auto name = cf.get<std::string>("Dataset");

  if(icompare("tunnel", name)) {
    return UniquePointer<Dataset>(new TunnelDataset(conf_fn));
  } else if(icompare("kitti", name)) {
    return UniquePointer<Dataset>(new KittiDataset(conf_fn));
  } else if(icompare("tsukuba_stereo", name)) {
    return UniquePointer<Dataset>(new TsukubaStereoDataset(conf_fn));
  } else if(icompare("tsukuba", name) || icompare("tsukuba_synthetic", name)) {
    return UniquePointer<Dataset>(new TsukubaSyntheticDataset(conf_fn));
  }

  THROW_ERROR(Format("unknown dataset '%s'\n", name.c_str()).c_str());
}

} // bpvo
