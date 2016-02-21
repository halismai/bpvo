# BPVO

A library for (semi-dense) real-time visual odometry. There two modes of
operation that are compiled separately. First, is the _real-time_ mode that not
handle difficult lighting (60+ fps). Second, is an illumination robust mode
handles low light using the Bit-Planes descriptor as described [here][bp].


## Building

### Dependencies

* [Eigen][eigen] version 3.2+
* [OpenCV][opencv] version 2.11 usage of opencv is limited to a few function.
  You need the `core`, `imgproc`, and optionally `highgui` and `contrib`
  modules.

### Real-time mode


USAGE

For examples see ./apps/vo.cc


A minimal example is as follows:

  include <bpvo/vo.h>

  using namespace bpvo;

  int main() {

	  // setup phase
	  // Get the camera intrinsics and baseline
	  Matrix33 K; // this is just an Eigen::Matrix<float,3,3>
	  float b;

	  // you also need the imgae size
	  ImageSize image_size = ...

	  // decide on the parameters you want to use
	  AlgorithmParameters params = ...

	  // create vo
	  VisualOdometry vo(K, b, image_size, params);

	  // keep adding frames
	  while(have_data) {
		auto result = vo.addFrame(image_ptr, disparity_ptr);
	  }

  }



[bp]: http://arxiv.org/abs/1602.00307

[eigen]: http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2

[opencv]: https://github.com/Itseez/opencv/archive/2.4.11.zip


