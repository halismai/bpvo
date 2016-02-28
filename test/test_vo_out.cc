#include <cstdio>
#if defined(WITH_CEREAL)

#include <bpvo/utils.h>
#include <bpvo/vo_output.h>
#include <utils/file_loader.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <cereal/archives/binary.hpp>

using namespace bpvo;

int main()
{
  FileLoader file_loader(".", "vo_%05d.voout", 1);

  cv::namedWindow("image");

  for(int i = 0; ; ++i)
  {
    auto fn = file_loader[i];
    if(!fs::exists(fn))
      break;
    {
      std::ifstream ifs(fn, std::ios::binary);
      if(!ifs.is_open())
      {
        printf("failed to open %s\n", fn.c_str());
        break;
      }

      {
        UniquePointer<VoOutput> frame;
        cereal::BinaryInputArchive ar(ifs);
        ar(frame);
      }
    }

  }

  return 0;
}

#else
int main() { printf("compile WITH_CEREAL\n"); }
#endif
