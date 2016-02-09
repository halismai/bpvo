#include <boost/program_options.hpp>

namespace bpo=boost::program_options;

int main(int argc, char** argv)
{
  // NOTE: boost 1.60.0 has a serious bunch of bugs with program_options
  bpo::options_description command_line("Command line options");
  command_line.add_options()("int,i", bpo::value<int>()->default_value(5));
  bpo::variables_map vmap;
  bpo::store(bpo::parse_command_line(argc,argv,command_line),vmap);

  return 0;
}


