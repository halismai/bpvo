#ifndef BPVO_UTILS_GLOB_H
#define BPVO_UTILS_GLOB_H

#include <string>
#include <vector>

namespace bpvo {

/**
 * globs for files or directories matching the pattern
 * \return the matched files, or an empty vector on any error
 *
 * Example
 *
 *   using namespace std;
 *   vector<string> files = glob("~/data/image*.png");
 *   for(const auto& f : files)
 *      system(("identify " + f).c_str());
 *
 */
std::vector<std::string> glob(const std::string& pattern, bool verbose=true);


}; // bpvo


#endif // BPVO_UTILS_GLOB_H
