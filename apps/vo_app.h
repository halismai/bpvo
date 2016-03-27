/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#ifndef BPVO_VO_APP_H
#define BPVO_VO_APP_H

#include <bpvo/types.h>

namespace bpvo {

class Dataset;
class Trajectory;

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

    /** store the timing per iteration to disk */
    bool store_iter_time;

    /** store the number of iterations per frame */
    bool store_iter_num;

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

  const Trajectory& getTrajectory() const;
  const std::vector<float>& getIterationTime() const;
  const std::vector<int>& getNumIterations() const;

 protected:
  struct Impl;
  UniquePointer<Impl> _impl;
}; // VoApp

}; // bpvo

#endif // BPVO_VO_APP_H
