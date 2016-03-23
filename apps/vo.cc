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

#include "utils/dataset_loader_thread.h"
#include "utils/program_options.h"

#include "bpvo/config.h"
#include "bpvo/utils.h"

#include "apps/vo_app.h"

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;


int main(int argc, char** argv)
{
  fprintf(stdout, "%s\n", BPVO_BUILD_STR);

  ProgramOptions options;
  options
      ("config,c", "/home/halismai/code/bpvo/conf/tsukuba.cfg", "config file")
      ("output,o", "output", "trajectory output file")
      ("numframes,n", int(100), "number of frames to process")
      ("buffersize,b", int(16), "buffer size to load images")
      ("points,p", "", "store the points to files with the given prefix")
      ("store-timing", "store the timing information")
      ("store-iterations", "store the number of iterations")
      ("dontshow,x", "do not show images")
      .parse(argc, argv);

  VoApp::Options vo_app_options;
  vo_app_options.trajectory_prefix = options.get<std::string>("output");
  vo_app_options.points_prefix = options.get<std::string>("points");
  vo_app_options.data_buffer_size = options.get<int>("buffersize");
  vo_app_options.max_num_frames = options.get<int>("numframes");
  vo_app_options.viewer_options.image_display_mode =
      options.hasOption("dontshow") ?
      VoApp::ViewerOptions::ImageDisplayMode::None :
      VoApp::ViewerOptions::ImageDisplayMode::ShowLeftAndDisparityOverlay;
  vo_app_options.store_iter_time = options.hasOption("store-timing");
  vo_app_options.store_iter_num = options.hasOption("store-iterations");

  const auto conf_fn = options.get<std::string>("config");
  VoApp vo_app(vo_app_options, conf_fn, Dataset::Create(conf_fn));
  vo_app.run();

  while(vo_app.isRunning())
    Sleep(100);

  Info("done\n");
  return 0;
}


