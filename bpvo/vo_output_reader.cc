#include "bpvo/debug.h"
#include "bpvo/vo_output_reader.h"
#include "bpvo/vo_output.h"
#include "utils/file_loader.h"

#if defined(WITH_CEREAL)
#include <fstream>
#include <cereal/archives/binary.hpp>
#endif

namespace bpvo {

VoOutputReader::VoOutputReader(std::string dname, std::string pattern, int f_start)
    : _file_loader(make_unique<FileLoader>(dname, pattern, f_start)) {}

VoOutputReader::~VoOutputReader() {}

UniquePointer<VoOutput> VoOutputReader::getFrame(int f_i)
{
  UniquePointer<VoOutput> ret;

#if defined(WITH_CEREAL)
  auto fn = _file_loader->operator[](f_i);

  if(!fn.empty() && fs::exists(fn))
  {
    std::ifstream ifs(fn, std::ios::binary);
    if(ifs.is_open())
    {
      cereal::BinaryInputArchive ar(ifs);
      ar(ret);
    } else {
      Warn("Failed to open %s\n", fn.c_str());
    }
  }
#else
  Warn("compile WITH_CEREAL\n");
#endif

  return ret;
}


} // bpvo
