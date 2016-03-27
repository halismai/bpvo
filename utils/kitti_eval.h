#ifndef KITTI_EVAL_H
#define KITTI_EVAL_H

#include <string>

void RunKittiEvaluation(std::string gt_dir, std::string results_dir,
                        std::string output_prefix);

#endif // KITTI_EVAL_H
