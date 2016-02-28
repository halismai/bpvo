#ifndef BPVO_VO_OUTPUT_READER_H
#define BPVO_VO_OUTPUT_READER_H

#include <string>
#include <bpvo/types.h>

namespace bpvo {

class VoOutput;
class FileLoader;

class VoOutputReader
{
 public:
  VoOutputReader(std::string dname, std::string pattern, int frame_start=0);
  ~VoOutputReader();

  UniquePointer<VoOutput> getFrame(int i);

  inline UniquePointer<VoOutput> operator[](int i) { return getFrame(i); }

 protected:
  UniquePointer<FileLoader> _file_loader;
}; // VoOutputReader

}; // bpvo

#endif // BPVO_VO_OUTPUT_READER_H
