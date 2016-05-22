# BPVO

[![Coverity Scan Build Status](https://scan.coverity.com/projects/8506/badge.svg)](https://scan.coverity.com/projects/halismai-bpvo)
[![Build Status](https://api.travis-ci.org/halismai/bpvo.svg?branch=master)](https://travis-ci.org/halismai/bpvo)
[![Code Health](https://landscape.io/github/halismai/bpvo/master/landscape.svg?style=flat)](https://landscape.io/github/halismai/bpvo/master)


A library for (semi-dense) real-time visual odometry from stereo data using direct alignment of feature descriptors. There are descriptors implemented. First, is raw intensity (no descriptor), which runs in  _real-time_  or faster. Second, is an implementation of the Bit-Planes descriptor designed for robust performance under challenging illumination conditions as described [here][bp] and [here][bpvo].

If you run into any issues, or have questions contact
```
halismai @ cs . cmu . edu
```

## Building

### Dependencies

* Compiler with c++11 support. Code is tested with `gcc-4.9`, `clang-3.5`, and `icc 16.0.1`
* [Eigen][eigen] version 3.2+
* [OpenCV][opencv] version 2.11 usage of opencv is limited to a few function. You need the `core`, `imgproc`, and optionally `highgui` and `contrib` modules.
* Optional, but recommended,`tbb` to speed up Bit-Planes. The code works with OpenMP as well, if available in the system.
* Optional: boost `program_options` and `circular_buffer`. You need version 1.58, 1.59 (there seems to be a bug with boost version 1.60). Older versions of boost will not work, as they do not have support for move semantics.
Other optional packages:

There are two libraries, the core `bpvo` and some utilities `bpvo_utils`. You do not need to compile the utilities if you want to embed the library in your code. If you are not building the utilities library, you do not need the following dependencies:
* `boost`
* `opencv_highgui`
* `opencv_contrib`

You can drop opencv altogether if you provide your own stereo code (and edit the code slightly).

To build the code base
```shell
mkdir build && cmake .. && make -j2
```

See `CMakeLists.txt` for additional flags and configurations. You may also configure the library using `cmake-gui`


## Building the Matlab interface
Get [mexmat](https://github.com/halismai/mexmat) and install it on your system.
```
git clone https://github.com/halismai/mexmat.git
```
It is a header-only library, so no compilation is needed. Then, until the matlab interface is integrated into the build system,
```
cd matlab && make
```

You might need to modify `matlab/Makefile` to point to the right location of Matlab and the c++ compiler. As of the date of writing (04/2016) Matlab R2015a supports up to g++-4.7. The code will require `g++4.8+`

If you get issues with Matlab GLIB_xxx not found, start matlab as
```
LD_PRELOAD=`g++-4.8+ -print-file-name=libstdc++.so` matlab
```

There is an experimental support for building the Matlab interface directly from the CMake build system. To try it out configure the build as:
```
cmake -DBUILD_MATLAB=ON ..
```
If that does not work, try the manual makefile above.

In either case, you will have to manually edit the location of mexmat and the Matlab path. The process will be fully automated in the future.


## Examples
A complete example is provided in `apps/vo.cc`

A simple example in `apps/vo_example.cc`

For real-time timting looin in `apps/vo_perf.cc` On my machine a dual core i7 from 2011, `vo_perf.cc` runs at 100+ Hz

A minimal example is as follows:
```cpp
#include <bpvo/vo.h>
#include <bpvo/trajectory.h>  // if you want the trajectory
#include <bpvo/point_cloud.h> // if you want the point cloud

using namespace bpvo;

int main()
{
  //
  // initialize VO using the calibration and AlgorithmParameters
  //
  Matrix33 K; // your calibration (Eigen typedef)
  float b;    // the stereo baseline
  ImageSize image_size(rows, cols); // the image size

  VisualOdometry vo(K, b, image_size, AlgorithmParameters());

  //
  // for every frame
  //  image_ptr is a uint8_t* to the image data
  //  disparity_ptr is a float* to the disparity map
  //
  // Both, the image and disparity size must be the same as supplied to
  // VisualOdometry constructor
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

### Matlab Example
```matlab
params = VoMex.DefaultParameters;
vo = VoMex(K, baseline, image_size, params);

T = eye(4); % the accumulated camera poses
% get images and disparities
% the image must be uint8_t and disparity float
% this assertion must hold
% assert( isa(I, 'uint8') && isa(D, 'single') );
result = vo.addFrame(I, D);

% accumulate the pose
T(:,:,end+1) = T(:,:,end) * inv( result.pose );
```

See also the code in side `matlab/`

## AlgorithmParameters
The parameters for the algorithm are documented in `bpvo/types.h`. It is important to get the parameters right for the type of data. Below are additional comments

### Common parameters
* `numPyramidLevels` You want this to be as small as possible to handle large motions. If you set it to -1, the code will automatically decide the number of pyramid levels. Sometimes, the lowest resolution image is too small and things might not work. For 640x480 images a value of 4 seems to works ok.

* `lossFunction` this is the type of the robust loss function used in the IRLS optimization. Use `kTukey`, or `kHuber`. You can also run without weighting with `kL2`, which is much faster but fails too often too often.

* `goodPointThreshold` Weights assigned to every point are between 0 and 1. Set this value to determine which points should be considered *good*. This will affect the number of 3D points you get in the point cloud. If the data is relatively clean, set this value to something high (e.g. 0.85), but if the data is fairly difficult without much stereo, set it to something lower (e.g. 0.6).

* `minNumPixelsForNonMaximaSuppression` to achieve real-time VO, when the number of pixels in the image exceeds minNumPixelsForNonMaximaSuppression we do non-maxima suppression on a saliency map extracted from the descriptor image. This results in semi-dense maps at the highest resolution. If you do not want this, and instead want as many 3D points at possible, set `minNumPixelsForNonMaximaSuppression to` a value higher than the number of pixels of your image. You can also disable this option be setting `nonMaxSuppRadius` to negative value.

* `minSaliency` minimum saliency to use a pixel. If you want to use all pixels irrespective of their saliency, set this to a negative value.


### For Intensity descriptor
If you want to use intensity only, disable parallisim. Compile the code with
```
cmake .. -DWITH_TBB=OFF -DWITH_SIMD=OFF
```

Or in your code
```
bpvo::setNumThreads(1);
```

The overhead of threads with intensity is not worth it for medium resolution images.

### Keyframing

* `minTranslationMagToKeyFrame`
* `minRotationMagToKeyFrame`
* `maxFractionOfGoodPointsToKeyFrame`

If you want to disable keyframing in order to get pose and point clouds for every image you add, set `minTranslationMagToKeyFrame=0.0`

### parameters specific to illumination robust mode

* `sigmaPriorToCensusTransform` this is the standard deviation of a Gaussian to blur the image before computing Bit-Planes. This should have a value less than 1.5, otherwise too much information is lost. A value of 0.5-0.75 is good.

* `sigmaBitPlanes` a standard deviation of Gaussian to smooth the Bit-Planes descriptor. You should experiment with this as it affects the basin of convergence. It also depends on how much motion there is in the data. A value of 0.75-1.5 is good.


## 3D point clouds
Point clouds are generated from the current keyframe. You should check if result.pointCloud is not NULL prior to accessing it. The point cloud comes with its pose as well in the world coordinate system.

## Citation
If you find this work useful please cite either
```
@ARTICLE{2016arXiv160400990A,
   author = {{Alismail}, Hatem and {Browning}, Browning and {Lucey}, Simon},
    title = "{Direct Visual Odometry using Bit-Planes}",
  journal = {ArXiv e-prints arXiv:1064.00990},
  year = {2016}
}
```
or
```
@article{alismail2016bit,
  title={Bit-Planes: Dense Subpixel Alignment of Binary Descriptors},
  author={{Alismail}, Hatem and {Browning}, Brett and {Lucey}, Simon},
  journal={arXiv preprint arXiv:1602.00307},
  year={2016}
}
```

Keep an eye on [http://www.cs.cmu.edu/~halismai/] for additional data

[bp]: http://arxiv.org/abs/1602.00307
[bpvo]: http://arxiv.org/abs/1604.00990
[eigen]: http://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2
[opencv]: https://github.com/Itseez/opencv/archive/2.4.11.zip

