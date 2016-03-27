#include "bpvo/types.h"
#include "bpvo/utils.h"
#include "utils/kitti_eval.h"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>
#include <cassert>

#include <array>

//
// most of the code is based on the KITTI devkit
//
static const std::array<float,8> EvalLengths{
    {100, 200, 300, 400, 500, 600, 700, 800}};

typedef Eigen::Matrix<double,4,4> Pose;
typedef typename bpvo::EigenAlignedContainer<Pose>::type PoseList;

static inline
Pose InvertPose(const Pose& T)
{
  Pose ret;

  ret.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
  ret.block<3,1>(0,3) = - ret.block<3,3>(0,0).transpose() * T.block<3,1>(0,3);
  ret.block<1,3>(3,0).setZero();
  ret(3,3) = 1.0;
  return ret;
}

static inline
PoseList LoadPoses(std::string filename)
{
  PoseList ret;

  FILE* fp = fopen(filename.c_str(), "r");
  THROW_ERROR_IF( fp == nullptr,
                 bpvo::Format("failed to open %s\n", filename.c_str()).c_str());

  while( !feof(fp) )
  {
    Pose T(Pose::Identity());
    if(12 == fscanf(fp,
                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &T(0,0), &T(0,1), &T(0,2), &T(0,3),
                    &T(1,0), &T(1,1), &T(1,2), &T(1,3),
                    &T(2,0), &T(2,1), &T(2,2), &T(2,3)))
      ret.push_back(T);
  }

  printf("got %zu poses\n", ret.size());
  fclose(fp);
  return ret;
}

static inline
std::vector<float> TrajectoryDistance(const PoseList& poses)
{
  std::vector<float> ret;
  ret.reserve( poses.size() );
  ret.push_back(0.0f);

  for(size_t i = 1; i < poses.size(); ++i) {
    float d = (poses[i-1].block<3,1>(0,3) - poses[i].block<3,1>(0,3)).norm();
    ret.push_back( ret[i-1]  + d );
  }

  return ret;
}

static inline
int LastFrameFromSegmentLength(const std::vector<float>& dists, int first_frame, float len)
{
  float d = dists[first_frame] + len;
  for(int i = first_frame; i < (int) dists.size(); ++i)
    if(dists[i] > d)
      return i;

  return -1;
}

static inline
float RotationError(const Pose& T_err)
{
  //float d = 0.5 * (T_err.diagonal().array().sum() - 1.0);
  float a = T_err(0,0);
  float b = T_err(1,1);
  float c = T_err(2,2);
  float d = 0.5f * (a + b + c - 1.0f);
  return std::acos(std::max(std::min(d, 1.0f), -1.0f));
}

static inline
float TranslationError(const Pose& T_err) { return T_err.block<3,1>(0,3).norm(); }


struct KittiErrors
{
  int32_t first_frame;
  float r_err;
  float t_err;
  float len;
  float speed;

  KittiErrors(int f, float r, float t, float l, float s)
      : first_frame(f), r_err(r), t_err(t), len(l), speed(s) {}
}; // KittiErrors

static inline
std::vector<KittiErrors>
CalcSequenceErrors(const PoseList& gt_poses, const PoseList& est_poses)
{
  std::vector<KittiErrors> ret;
  auto dists = TrajectoryDistance(gt_poses);


  int step_size = 10;
  for(int f_i = 0; f_i < (int) gt_poses.size(); f_i += step_size)
  {
    const Pose T_gt_inv = InvertPose( gt_poses[f_i] );
    const Pose T_est_inv = InvertPose( est_poses[f_i] );

    for(int i = 0; i < (int) EvalLengths.size(); ++i)
    {
      int f_last = LastFrameFromSegmentLength( dists, f_i, EvalLengths[i] );

      if(f_last != -1)
      {
        const Pose T_d_gt = T_gt_inv * gt_poses[f_last];
        const Pose T_d_est = T_est_inv * est_poses[f_last];
        const Pose T_err = InvertPose( T_d_est ) * T_d_gt;

        ret.push_back(
            KittiErrors(
                f_i,
                RotationError(T_err) / EvalLengths[i],
                TranslationError(T_err) / EvalLengths[i],
                EvalLengths[i],
                EvalLengths[i] / (0.1 * (f_last - f_i + 1)))
            );
      }
    }
  }

  return ret;
}

static inline
void SaveErrorPlotData(const std::vector<KittiErrors>& errors, std::string output_prefix)
{
  std::string fn;

  fn = bpvo::Format("%s_tl.txt", output_prefix.c_str());
  FILE* fp_tl = fopen( fn.c_str(), "w" );
  if(!fp_tl) {
    THROW_ERROR(bpvo::Format("Failed to open %s\n", fn.c_str()).c_str());
  }

  fn = bpvo::Format("%s_rl.txt", output_prefix.c_str());
  FILE* fp_rl = fopen( fn.c_str(), "w" );
  if(!fp_rl) {
    fclose(fp_tl);
    THROW_ERROR(bpvo::Format("Failed to open %s\n", fn.c_str()).c_str());
  }

  fn = bpvo::Format("%s_ts.txt", output_prefix.c_str());
  FILE* fp_ts = fopen( fn.c_str(), "w" );
  if(!fp_ts) {
    fclose(fp_tl);
    fclose(fp_rl);
    THROW_ERROR(bpvo::Format("Failed to open %s\n", fn.c_str()).c_str());
  }

  fn = bpvo::Format("%s_rs.txt", output_prefix.c_str());
  FILE* fp_rs = fopen( fn.c_str(), "w" );
  if(!fp_ts) {
    fclose(fp_tl);
    fclose(fp_rl);
    fclose(fp_ts);
    THROW_ERROR(bpvo::Format("Failed to open %s\n", fn.c_str()).c_str());
  }

  for(int i = 0; i < (int) EvalLengths.size(); ++i)
  {
    float t_err = 0.0f, r_err = 0.0f;
    int count = 0;

    for(const auto& err : errors)
    {
      if( std::fabs( err.len - EvalLengths[i] ) < 1.0f )
      {
        t_err += err.t_err;
        r_err += err.r_err;
        ++count;
      }
    }

    if(count > 2) {
      fprintf(fp_tl, "%f %f\n", EvalLengths[i], t_err / (float) count);
      fprintf(fp_rl, "%f %f\n", EvalLengths[i], r_err / (float) count);
    }
  }

  fclose(fp_tl);
  fclose(fp_rl);

  for(int speed = 2; speed < 25; speed += 2)
  {
    float t_err = 0.0f, r_err = 0.0f;
    int count = 0;

    for(const auto& err : errors)
    {
      if(std::fabs(err.speed - static_cast<float>(speed) ) < 2.0f)
      {
        t_err += err.t_err;
        r_err += err.r_err;
        ++count;
      }
    }

    if(count > 2)
    {
      fprintf(fp_ts, "%d %f\n", speed, t_err / count);
      fprintf(fp_rs, "%d %f\n", speed, r_err / count);
    }
  }

  fclose(fp_ts);
  fclose(fp_rs);
}


void RunKittiEvaluation(std::string gt_dir, std::string results_dir,
                        std::string output_prefix)
{
  std::vector<KittiErrors> total_err;
  for(int i = 0; i <= 10; ++i)
  {
    printf("loading %d\n", i);
    auto gt_poses = LoadPoses( bpvo::Format("%s/%02d.txt", gt_dir.c_str(), i) ),
         est_poses = LoadPoses( bpvo::Format("%s/%02d.txt", results_dir.c_str(), i) );

    THROW_ERROR_IF( gt_poses.empty(), "failed to load ground truth poses" );
    THROW_ERROR_IF( est_poses.empty(), "failed to load user results" );

    auto seq_err = CalcSequenceErrors(gt_poses, est_poses);
    total_err.insert(total_err.end(), seq_err.begin(), seq_err.end());
  }

  SaveErrorPlotData(total_err, output_prefix);
}


