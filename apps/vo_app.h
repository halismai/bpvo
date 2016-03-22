#ifndef BPVO_VO_APP_H
#define BPVO_VO_APP_H

#include <bpvo/types.h>

namespace bpvo {

class Dataset;
class Viewer;

class VoApp
{
 public:
  struct ViewerOptions
  {
    enum class ImageDisplayMode
    {
      None, //< do not show image
      ShowLeftOnly, //< show the left image only
      ShowLeftAndDisparity, //< show image and disparity in separate windows
      ShowLeftAndDisparityOverlay //< overlay the image and disparity
    }; // ImageDisplayMode

    ImageDisplayMode image_display_mode;

    ViewerOptions();
  }; // ViewerOptions

  struct Options
  {
    /** output filename to store the trajectory and camera path */
    std::string trajectory_prefix;
    /** prefix to store point clouds from reference frames */
    std::string points_prefix;
    /** buffer to size for the data loader thread */
    size_t data_buffer_size;
    /** maximum number of frames to process */
    int max_num_frames;
    /** what to show, if any */
    ViewerOptions viewer_options;

    Options();
  }; // Options

 public:
  /**
   * \parma options
   * \param conf_fn configuration file for vo
   * \param dataset the dataset to load
   */
  VoApp(Options options, std::string conf_fn, UniquePointer<Dataset> datsaet);

  ~VoApp();

  /**
   */
  void run();

  /**
   * stops the code
   */
  void stop();

  /**
   * \return true if VO is still running
   */
  bool isRunning() const;

 protected:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // VoApp

}; // bpvo

#endif // BPVO_VO_APP_H
