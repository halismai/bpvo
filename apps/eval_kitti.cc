#include "utils/dataset_loader_thread.h"
#include "utils/program_options.h"
#include "utils/kitti_eval.h"

#include "bpvo/config.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"
#include "bpvo/trajectory.h"

#include "apps/vo_app.h"

#include <fstream>

using namespace bpvo;

static inline
UniquePointer<Dataset> GetKittiSequence(int s, int scale_by = 1)
{
  std::string fn("/tmp/kitti_data.cfg");
  FILE* fp = fopen(fn.c_str(), "w");

  fprintf(fp, "Dataset = kitti\n");
  fprintf(fp, "DatasetRootDirectory = ~/data/kitti/dataset\n");
  fprintf(fp, "SequenceNumber = %d\n", s);
  fprintf(fp, "ScaleBy = %d\n", scale_by);

  //fprintf(fp, "StereoAlgorithm = BlockMatching\n");
  fprintf(fp, "StereoAlgorithm = SemiGlobalMatching\n");
  fprintf(fp, "SADWindowSize = 9\n");
  fprintf(fp, "minDisparity = 0\n");
  fprintf(fp, "numberOfDisparities = 96\n");
  fprintf(fp, "textureThreshold = 5\n");
  fprintf(fp, "uniquenessRatio = 15\n");
  fprintf(fp, "trySmallerWindows = 1\n");

  fclose(fp);

  Info("dataset file %s\n", fn.c_str());
  return Dataset::Create(fn);
}

static inline
void WriteTrajectoryKittiFormat(std::string filename, const Trajectory& trajectory)
{
  FILE* fp = fopen(filename.c_str(), "w");
  DIE_IF( fp == nullptr, Format("Failed to open '%s'", filename.c_str()).c_str() );

  for(size_t i = 0; i < trajectory.size(); ++i)
  {
    const auto& T = trajectory[i];
    fprintf(fp,
            "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            T(0,0), T(0,1), T(0,2), T(0,3),
            T(1,0), T(1,1), T(1,2), T(1,3),
            T(2,0), T(2,1), T(2,2), T(2,3));
  }

  fclose(fp);
}

template <typename T> static inline
void WriteVector(std::string filename, const std::vector<T>& data)
{
  std::ofstream ofs(filename);
  DIE_IF( !ofs.is_open(), Format("Failed to open '%s'", filename.c_str()).c_str() );

  for(const auto& v : data)
    ofs << v << "\n";

  ofs.close();
}

int main(int argc, char** argv)
{
  try {
    ProgramOptions options;
    options
        ("config,c", "", "config file")
        ("output,o", ".", "output dir").parse(argc, argv);

    const auto config = options.get<std::string>("config");
    const auto output_dir = options.get<std::string>("output");
    DIE_IF( config.empty(), "need config file" );

    for(int i = 1; i <= 10; ++i)
    {
      Info("Running sequence number %d\n", i);

      VoApp::Options vo_app_options;

      vo_app_options.store_iter_time = true;
      vo_app_options.store_iter_num = true;
      vo_app_options.data_buffer_size = 16;
      vo_app_options.max_num_frames = -1;

      VoApp::ViewerOptions viewer_options;
      viewer_options.image_display_mode = VoApp::ViewerOptions::ImageDisplayMode::None;
      vo_app_options.viewer_options = viewer_options;

      auto dataset = GetKittiSequence(i);

      printf("\n\n\n");
      Info("Starting VO\n");
      VoApp vo_app( vo_app_options, config, std::move(dataset));
      vo_app.run();
      while(vo_app.isRunning())
        Sleep(100);

      Info("done .. writing results\n");
      WriteTrajectoryKittiFormat(
          Format("%s/%02d.txt", output_dir.c_str(), i), vo_app.getTrajectory());
      WriteVector(
          Format("%s/%02d_time.txt", output_dir.c_str(), i), vo_app.getIterationTime());
      WriteVector(
          Format("%s/%02d_iters.txt", output_dir.c_str(), i), vo_app.getNumIterations());
    }

  } catch(const std::exception& ex) {
    fprintf(stderr, "ERROR: %s\n", ex.what());
  }

  return 0;
}

