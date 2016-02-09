#ifndef BPVO_TEST_PROGRAM_OPTIONS
#define BPVO_TEST_PROGRAM_OPTIONS

#include <string>
#include <boost/program_options.hpp>
#include <iostream>

namespace bpvo {

class ProgramOptions
{
 public:
  ProgramOptions(std::string name = "ProgramOptions");

  ProgramOptions& addOption(std::string name, std::string msg);

  inline ProgramOptions& operator()(std::string name, std::string msg) {
    return addOption(name, msg);
  }

  template <class T> inline
  ProgramOptions& operator()(std::string name, T v, std::string help)
  {
    _desc.add_options()(name.c_str(), boost::program_options::value<T>()->default_value(v),
                        help.c_str());
    return *this;
  }

  ProgramOptions& operator()(std::string name, const char* v, std::string help)
  {
    return this->operator()(name, std::string(v), help);
  }

  template <class T> inline
  T get(std::string name) const {
    try {
      return _vm[name].template as<T>();
    } catch(const std::exception& ex) {
      std::cerr << "Error: " << ex.what() << std::endl;
      throw ex;
    }
  }

  void parse(int argc, char** argv);

  void printHelpAndExit(int exit_code = 0) const;

  bool hasOption(std::string) const;

 private:
  boost::program_options::options_description _desc;
  boost::program_options::variables_map _vm;
}; // ProgramOptions

}; // bpvo

#endif // BPVO_TEST_PROGRAM_OPTIONS
