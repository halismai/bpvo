#include "bpvo/vo_output_writer.h"
#include "bpvo/debug.h"

#if defined(WITH_CEREAL)
#include <fstream>
#include <cereal/archives/binary.hpp>
#endif

namespace bpvo {

void VoOutputWriter::run()
{
  ElementType frame;
  while(!_stop)
  {
    if(_buffer.pop(&frame))
    {
      if(frame.second->pointCloud().empty())
      {
        Warn("got an empty point cloud\n");
      } else
      {
#if defined(WITH_CEREAL)
        std::ofstream ofs(frame.first, std::ios::binary);
        if(ofs.is_open())
        {
          cereal::BinaryOutputArchive ar(ofs);
          ar(*frame.second);
        } else
        {
          Warn("Failed to open %s\n", frame.first.c_str());
        }
#else
        Warn("compile WITH_CEREAL\n");
#endif
      }
    }
  }

  _stop = true;
}

} // bpvo

