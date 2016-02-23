#include "bpvo/debug.h"
#include "utils/glob.h"
#include <glob.h>

namespace bpvo {

std::vector<std::string> glob(const std::string& pattern, bool verbose)
{
  std::vector<std::string> ret;

  ::glob_t globbuf;
  int err = ::glob(pattern.c_str(), GLOB_TILDE, NULL, &globbuf);
  switch(err)
  {
    case GLOB_NOSPACE:
      {
        if(verbose) Warn("glob(): out of  memory\n");
        break;
      }
    case GLOB_ABORTED:
      {
        if(verbose) Warn("glob() : aborted. read error\n");
        break;
      }
    case GLOB_NOMATCH :
      {
        if(verbose) Warn("glob() : no match for: '%s'\n", pattern.c_str());
        break;
    }
  }

  if(!err)
  {
    const int count = globbuf.gl_pathc;
    ret.resize(count);
    for(int i = 0; i < count; ++i)
      ret[i] = std::string(globbuf.gl_pathv[i]);
  }

  globfree(&globbuf);
  return ret;
}

}; // bpvo
