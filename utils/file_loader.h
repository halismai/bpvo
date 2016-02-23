#ifndef BPVO_UTILS_FILE_LOADER_H
#define BPVO_UTILS_FILE_LOADER_H

#include <string>
#include <vector>

namespace bpvo {

class FileLoader
{
 public:
  /**
   * \param dir      data directory
   * \param pattern  either a glob (e.g. \.*.png) or a print fmt (e.g.
   *                 "image_%04d.png"). This will be auto detected. If the
   *                 'pattern' contains a '%' then it will be taken as format
   * \param frame_start first frame in the sequence
   */

  FileLoader(std::string dname, std::string pattern, int frame_start = 0);

  /**
   * \return the next file name, or an empty string if not more frames
   */
  std::string next();

  /**
   * \return file at position 'i'
   */
  std::string operator[](size_t i) const;

  /**
   * \return the number of files
   */
  size_t size() const;

 private:
  std::string _fmt;
  std::vector<std::string> _files;
  int _frame_start;
  int _frame_counter;
}; // FileLoader

}; // bpvo


#endif // BPVO_UTILS_FILE_LOADER_H
