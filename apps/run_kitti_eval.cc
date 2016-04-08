#include "utils/kitti_eval.h"
#include "utils/program_options.h"

#include "bpvo/utils.h"

using namespace bpvo;

int main(int argc, char** argv)
{
  ProgramOptions options;
  options
      ("groundtruth,g", "", "groundtruth data directory")
      ("results,r", "", "results directory")
      ("output,o", "kitti_eval", "output prefix").parse(argc, argv);

  auto gt_dir = options.get<std::string>("groundtruth");
  auto results_dir = options.get<std::string>("results");

  DIE_IF( gt_dir.empty(), "need groundtruth directory"  );
  DIE_IF( results_dir.empty(), "need results directory" );

  RunKittiEvaluation(gt_dir, results_dir, options.get<std::string>("output"));

  return 0;
}

