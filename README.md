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

  // DONE, there is nothing special to do delete the object
  return EXIT_SUCCESS;
}
```

## AlgorithmParameters
The parameters for the algorithm are documented in `bpvo/types.h`. It is important to get the parameters right for the type of data. Below are additional comments

### Common parameters
* `numPyramidLevels` You want this to be as small as possible to handle large motions. If you set it to -1, the code will automatically decide the number of pyramid levels. Sometimes, the lowest resolution image is too small and things might not work. For 640x480 images a value of 4 works ok.

* `lossFunction` this is the type of the robust loss function used in the IRLS optimization. Use `kTukey`, or 'kHuber'. You can also run without weighting with 'kL2', which is much faster but unreliable.

* `goodPointThreshold` Weights assigned to every point are between 0 and 1. Set this value to determine which points should be considered *good*. This will affect the number of 3D points you get in the point cloud. If the data is relatively clean, set this value to something high (e.g. 0.85), but if the data is fairly difficult without much stereo, set it to something lower (e.g. 0.5).

* `minNumPixelsForNonMaximaSuppression` to achieve real-time VO, when the number of pixels in the image exceeds minNumPixelsForNonMaximaSuppression we do non-maxima suppression on a saliency map extracted from the image. This results in semi-dense maps at the highest resolution. If you do not want this, and instead you want as many 3D points at possible, set minNumPixelsForNonMaximaSuppression to a value higher than the number of pixels of your image

* `minSaliency` minimum saliency to use a pixel. If you want to use all pixels irrespective of their saliency, set this to a negative value

### Keyframing

* `minTranslationMagToKeyFrame`
* `minRotationMagToKeyFrame`
* `maxFractionOfGoodPointsToKeyFrame`

If you want to disable keyframing in order to get pose and point clouds for every image you add, set `minTranslationMagToKeyFrame=0.0`

### parameters specific to illumination robust mode

* `sigmaPriorToCensusTransform` this is the standard deviation of a Gaussian to blur the image before computing Bit-Planes. This should have a value less than 1.5, otherwise too much information is lost. A value of 0.5-0.75 is good.

* `sigmaBitPlanes` a std dev of Gaussian to smooth the Bit-Planes descriptor. You should experiment with this as it affects the basin of convergence. It also depends on how much motion there is in the data. A value of 0.75-1.5 is good.


## 3D point clouds
Point clouds are generated from the current keyframe. You should check if result.pointCloud is not NULL prior to accessing it. The point cloud comes with its pose as well in the world coordinate system.

[bp]: http://arxiv.org/abs/1602.00307

[eigen]: http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2

[opencv]: https://github.com/Itseez/opencv/archive/2.4.11.zip

