#include "utils/program_options.h"
#include <iostream>

namespace bpvo {

namespace po = boost::program_options;

ProgramOptions::ProgramOptions(std::string name)
     : _desc(name) { addOption("help,h", "Print help message and exit"); }

void ProgramOptions::parse(int argc, char** argv)
{
  try {
    po::store(po::parse_command_line(argc, argv, _desc), _vm);
  } catch(const std::exception& ex) {
    std::cerr << "error: " << ex.what() << std::endl;
    throw ex;
  }

  po::notify(_vm);

  if(hasOption("help"))
    printHelpAndExit();
}

bool ProgramOptions::hasOption(std::string name) const
{
  return _vm.count(name);
}

void ProgramOptions::printHelpAndExit(int exit_code) const
{
  std::cout << _desc << std::endl;
  exit(exit_code);
}

ProgramOptions& ProgramOptions::addOption(std::string name, std::string help)
{
  _desc.add_options()(name.c_str(), help.c_str());
  return *this;
}

}
