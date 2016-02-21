# BPVO

A library for (semi-dense) real-time visual odometry. There two modes of operation that are compiled separately. First, is the _real-time_ mode that not handle difficult lighting (60+ fps). Second, is an illumination robust mode handles low light using the Bit-Planes descriptor as described [here][bp].


## Building

### Dependencies

* Compiler with c++11 support. Code is tested with `gcc-4.9`, `clang-3.5`, and `icc 16.0.1`
* [Eigen][eigen] version 3.2+
* [OpenCV][opencv] version 2.11 usage of opencv is limited to a few function. You need the `core`, `imgproc`, and optionally `highgui` and `contrib` modules.
* Optional, but recommended,`tbb` any version.
* Optional: boost `program_options` and `circular_buffer`. You need version 1.58, 1.59 (there seems to be a bug with boost version 1.60). Older version of boost do not have support for move semantics and will not work.

Other optional packages:
* Google Performance tools, `libtcmalloc` which can be turned on with `-DWITH_TCMALLOC`

There are two libraries, the core `bpvo` and some utilities `bpvo_utils`. You do not need to compile the utilities if you want to embed the library in your code. If you are not building the utilities library, you do not need the following dependencies:
* `boost`
* `opencv_highgui`
* `opencv_contrib`

### Real-time mode
```shell
mkdir build && cmake .. && \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_TBB=ON              \
  -DBUILD_STATIC=ON  && make -j3
```

See `CMakeLists.txt` for additional flags and configurations. You may also configure the library using `cmake-gui`

### Illumination robust mode

Same as above, but turn on the BITPLANES options
```shell
mkdir build && cmake .. && \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_STATIC=ON          \
  -DWITH_TBB=ON              \
  -DWITH_BITPLANES=ON && make -j3
```


## Examples
A complete example is provided in `apps/vo.cc`

A minimal example is as follows:
```cpp
#include <bpvo/vo.h>
#include <bpvo/trajectory.h> // if you want the trajectory
#include <bpvo/point_cloud.h> // if you want the point cloud

using namespace bpvo;

int main()
{
  //
  // initialize VO using the calibration and AlgorithmParameters
  //
  Matrix33 K; // your calibration
  float b;    // the streo baseline
  ImageSize image_size(rows, cols); // the image size

  VisualOdometry vo(K, b, image_size, AlgorithmParameters());

  //
  // for every frame
  //  image_ptr is a uint8_t* to the image data
  //  disparity_ptr is a float* to the disparity map
  //
  Result result = vo.addFrame(image_ptr, disparity_ptr);

  //
  // If you want the point cloud, you must check if it is available
  //
  if(result.pointCloud) {
    // do something with the point cloud
  }

  // you can also get the trajectory of the camera at any point by calling
  auto trajectory = vo.trajectory();

}
```





[bp]: http://arxiv.org/abs/1602.00307

[eigen]: http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2

[opencv]: https://github.com/Itseez/opencv/archive/2.4.11.zip


