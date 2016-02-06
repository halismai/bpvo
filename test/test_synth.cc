#include "test/data_loader.h"
#include "bpvo/types.h"
#include "bpvo/vo.h"
#include "bpvo/math_utils.h"

#include <cmath>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>

using namespace bpvo;


StereoFrame MakeSynthetic(const StereoCalibration& calib, const Matrix44& T,
                          const StereoFrame& frame)
{
  float fx = calib.K(0,0),
        fy = calib.K(1,1),
        cx = calib.K(0,2),
        cy = calib.K(1,2),
        b = calib.baseline,
        bf = b * fx;

  const cv::Mat& I0 = frame.image();
  const cv::Mat& D = frame.disparity();
  Matrix34 P = calib.K * T.block<3,4>(0,0);

  cv::Mat I(I0.size(), CV_8U);
  for(int r = 0; r < I0.rows; ++r)
  {
    for(int c = 0; c < I0.cols; ++c)
    {
      float Z = bf / D.at<float>(r,c);
      float X = (c - cx) * Z / fx;
      float Y = (r - cy) * Z / fy;
      Eigen::Vector3f Xw = P * Eigen::Vector4f(X, Y, Z, 1.0);

      float xf = Xw[0] / Xw[2];
      float yf = Xw[1] / Xw[2];

      int xi = static_cast<int>( xf + 0.5 );
      int yi = static_cast<int>( yf + 0.5 );

      if(xi >= 0 && xi < I0.cols-1 && yi >= 0 && yi < I0.rows-1)
      {
        yf -= yi;
        xf -= xi;
        float i0 = (float) I0.at<uint8_t>(yi,xi),
              i1 = (float) I0.at<uint8_t>(yi,xi+1),
              i2 = (float) I0.at<uint8_t>(yi+1,xi),
              i3 = (float) I0.at<uint8_t>(yi+1,xi+1),
              Iw = (1.0f-yf) * ((1.0f-xf)*i0 + xf*i1) +
                         yf  * ((1.0f-xf)*i2 + xf*i3);
        I.at<uint8_t>(r,c) = cv::saturate_cast<uint8_t>(Iw);
      } else
      {
        I.at<uint8_t>(r,c) = 0;
      }
    }
  }

  StereoFrame ret;
  ret.setLeft(I);
  ret.setDisparity(D);
  return ret;
}


int main()
{
  Matrix44 T(Matrix44::Identity());
  T.block<3,1>(0,3) = Eigen::Vector3f(0.01, 0.01, 0.01);
  std::cout << T << std::endl;
  TsukubaDataLoader data_loader;
  auto calib = data_loader.calibration();
  auto f1 = data_loader.getFrame(1);

  std::cout << "making frame:\n";
  auto f0 = MakeSynthetic(calib, T, *(StereoFrame*)f1.get());

  AlgorithmParameters params("../conf/tsukuba.cfg");
  std::cout << params << std::endl;
  VisualOdometry vo(calib.K, calib.baseline, data_loader.imageSize(), params);

  cv::imshow("image", f0.image());
  cv::waitKey(10);

  vo.addFrame(f0.image().ptr<uint8_t>(), f0.disparity().ptr<float>());
  auto result = vo.addFrame(f1->image().ptr<uint8_t>(), f1->disparity().ptr<float>());

  auto pose_error = (math::MatrixToTwist(result.pose.inverse() * T));

  std::cout << "\n\n";
  std::cout << result << "\n" << std::endl;
  std::cout << "ERROR: " << pose_error.norm() << std::endl;
  std::cout << pose_error.transpose() << std::endl;

  return 0;
}

