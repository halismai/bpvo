#include "bpvo/types.h"
#include "utils/sps.h"

#include <opencv2/core/core.hpp>

/*
    Copyright (C) 2014  Koichiro Yamaguchi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
// modified by halismai

#include <stack>
class SPSStereo {
 public:
  SPSStereo();

  void setOutputDisparityFactor(const double outputDisparityFactor);
  void setIterationTotal(const int outerIterationTotal, const int innerIterationTotal);
  void setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight);
  void setInlierThreshold(const double inlierThreshold);
  void setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty);

  void compute(const int superpixelTotal, const cv::Mat& leftImage, const cv::Mat& rightImage,
               cv::Mat& segmentImage, cv::Mat& disparityImage,
               std::vector< std::vector<double> >& disparityPlaneParameters,
               std::vector< std::vector<int> >& boundaryLabels);

 private:
  class Segment {
   public:
    Segment() {
      pixelTotal_ = 0;
      colorSum_[0] = 0;  colorSum_[1] = 0;  colorSum_[2] = 0;
      positionSum_[0] = 0;  positionSum_[1] = 0;
      disparityPlane_[0] = 0;  disparityPlane_[1] = 0;  disparityPlane_[2] = -1;
    }

    void addPixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
      pixelTotal_ += 1;
      colorSum_[0] += colorL;  colorSum_[1] += colorA;  colorSum_[2] += colorB;
      positionSum_[0] += x;  positionSum_[1] += y;
    }
    void removePixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
      pixelTotal_ -= 1;
      colorSum_[0] -= colorL;  colorSum_[1] -= colorA;  colorSum_[2] -= colorB;
      positionSum_[0] -= x;  positionSum_[1] -= y;
    }
    void setDisparityPlane(const double planeGradientX, const double planeGradientY, const double planeConstant) {
      disparityPlane_[0] = planeGradientX;
      disparityPlane_[1] = planeGradientY;
      disparityPlane_[2] = planeConstant;
    }

    int pixelTotal() const { return pixelTotal_; }
    double color(const int colorIndex) const { return colorSum_[colorIndex]/pixelTotal_; }
    double position(const int coordinateIndex) const { return positionSum_[coordinateIndex]/pixelTotal_; }
    double estimatedDisparity(const double x, const double y) const
    {
      return disparityPlane_[0]*x + disparityPlane_[1]*y + disparityPlane_[2];
    }

    bool hasDisparityPlane() const
    {
      if (disparityPlane_[0] != 0.0 || disparityPlane_[1] != 0.0 || disparityPlane_[2] != -1.0)
        return true;
      else return false;
    }

    void clearConfiguration()
    {
      neighborSegmentIndices_.clear();
      boundaryIndices_.clear();
      for (int i = 0; i < 9; ++i) polynomialCoefficients_[i] = 0;
      for (int i = 0; i < 6; ++i) polynomialCoefficientsAll_[i] = 0;
    }

    void appendBoundaryIndex(const int boundaryIndex) { boundaryIndices_.push_back(boundaryIndex); }
    void appendSegmentPixel(const int x, const int y) {
      polynomialCoefficientsAll_[0] += x*x;
      polynomialCoefficientsAll_[1] += y*y;
      polynomialCoefficientsAll_[2] += x*y;
      polynomialCoefficientsAll_[3] += x;
      polynomialCoefficientsAll_[4] += y;
      polynomialCoefficientsAll_[5] += 1;
    }
    void appendSegmentPixelWithDisparity(const int x, const int y, const double d) {
      polynomialCoefficients_[0] += x*x;
      polynomialCoefficients_[1] += y*y;
      polynomialCoefficients_[2] += x*y;
      polynomialCoefficients_[3] += x;
      polynomialCoefficients_[4] += y;
      polynomialCoefficients_[5] += x*d;
      polynomialCoefficients_[6] += y*d;
      polynomialCoefficients_[7] += d;
      polynomialCoefficients_[8] += 1;
    }

    int neighborTotal() const { return static_cast<int>(neighborSegmentIndices_.size()); }
    int neighborIndex(const int index) const { return neighborSegmentIndices_[index]; }

    int boundaryTotal() const { return static_cast<int>(boundaryIndices_.size()); }
    int boundaryIndex(const int index) const { return boundaryIndices_[index]; }

    double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }
    double polynomialCoefficientAll(const int index) const { return polynomialCoefficientsAll_[index]; }

    double planeParameter(const int index) const { return disparityPlane_[index]; }

   private:
    int pixelTotal_;
    double colorSum_[3];
    double positionSum_[2];
    double disparityPlane_[3];

    std::vector<int> neighborSegmentIndices_;
    std::vector<int> boundaryIndices_;
    double polynomialCoefficients_[9];
    double polynomialCoefficientsAll_[6];
  };
  class Boundary {
   public:
    Boundary() { segmentIndices_[0] = -1; segmentIndices_[1] = -1; clearCoefficients(); }
    Boundary(const int firstSegmentIndex, const int secondSegmentIndex) {
      if (firstSegmentIndex < secondSegmentIndex) {
        segmentIndices_[0] = firstSegmentIndex; segmentIndices_[1] = secondSegmentIndex;
      } else {
        segmentIndices_[0] = secondSegmentIndex; segmentIndices_[1] = firstSegmentIndex;
      }
      clearCoefficients();
    }

    void clearCoefficients() {
      for (int i = 0; i < 6; ++i) polynomialCoefficients_[i] = 0;
    }

    void setType(const int typeIndex) { type_ = typeIndex; }
    void appendBoundaryPixel(const double x, const double y) {
      boundaryPixelXs_.push_back(x);
      boundaryPixelYs_.push_back(y);
      polynomialCoefficients_[0] += x*x;
      polynomialCoefficients_[1] += y*y;
      polynomialCoefficients_[2] += x*y;
      polynomialCoefficients_[3] += x;
      polynomialCoefficients_[4] += y;
      polynomialCoefficients_[5] += 1;
    }

    int type() const { return type_; }
    int segmentIndex(const int index) const { return segmentIndices_[index]; }
    bool consistOf(const int firstSegmentIndex, const int secondSegmentIndex) const {
      if ((firstSegmentIndex == segmentIndices_[0] && secondSegmentIndex == segmentIndices_[1])
          || (firstSegmentIndex == segmentIndices_[1] && secondSegmentIndex == segmentIndices_[0]))
      {
        return true;
      }
      return false;
    }
    int include(const int segmentIndex) const {
      if (segmentIndex == segmentIndices_[0]) return 0;
      else if (segmentIndex == segmentIndices_[1]) return 1;
      else return -1;
    }
    int boundaryPixelTotal() const { return static_cast<int>(boundaryPixelXs_.size()); }
    double boundaryPixelX(const int index) const { return boundaryPixelXs_[index]; }
    double boundaryPixelY(const int index) const { return boundaryPixelYs_[index]; }

    double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }

   private:
    int type_;
    int segmentIndices_[2];
    std::vector<double> boundaryPixelXs_;
    std::vector<double> boundaryPixelYs_;

    double polynomialCoefficients_[6];
  };


  void allocateBuffer();
  void freeBuffer();
  void setInputData(const cv::Mat& leftImage, const cv::Mat& rightImage);
  void setLabImage(const cv::Mat& leftImage);
  void computeInitialDisparityImage(const cv::Mat& leftImage, const cv::Mat& rightImage);
  void initializeSegment(const int superpixelTotal);
  void makeGridSegment(const int superpixelTotal);
  void assignLabel();
  void extractBoundaryPixel(std::stack<int>& boundaryPixelIndices);
  bool isBoundaryPixel(const int x, const int y) const;
  bool isUnchangeable(const int x, const int y) const;
  int findBestSegmentLabel(const int x, const int y) const;
  std::vector<int> getNeighborSegmentIndices(const int x, const int y) const;
  double computePixelEnergy(const int x, const int y, const int segmentIndex) const;
  double computeBoundaryLengthEnergy(const int x, const int y, const int segmentIndex) const;
  void changeSegmentLabel(const int x, const int y, const int newSegmentIndex);
  void addNeighborBoundaryPixel(const int x, const int y, std::stack<int>& boundaryPixelIndices) const;
  void initialFitDisparityPlane();
  void estimateDisparityPlaneRANSAC(const float* disparityImage);
  void solvePlaneEquations(const double x1, const double y1, const double z1, const double d1,
                           const double x2, const double y2, const double z2, const double d2,
                           const double x3, const double y3, const double z3, const double d3,
                           std::vector<double>& planeParameter) const;
  int computeRequiredSamplingTotal(const int drawTotal, const int inlierTotal, const int pointTotal, const int currentSamplingTotal, const double confidenceLevel) const;
  void interpolateDisparityImage(float* interpolatedDisparityImage) const;
  void initializeOutlierFlagImage();
  void performSmoothingSegmentation();
  void buildSegmentConfiguration();
  bool isHorizontalBoundary(const int x, const int y) const;
  bool isVerticalBoundary(const int x, const int y) const;
  int appendBoundary(const int firstSegmentIndex, const int secondSegmentIndex);
  void planeSmoothing();
  void estimateBoundaryLabel();
  void estimateSmoothFitting();
  void makeOutputImage(cv::Mat& segmentImage, cv::Mat& segmentDisparityImage) const;
  void makeSegmentBoundaryData(std::vector< std::vector<double> >& disparityPlaneParameters,
                               std::vector< std::vector<int> >& boundaryLabels) const;


  // Parameter
  double outputDisparityFactor_;
  int outerIterationTotal_;
  int innerIterationTotal_;
  double positionWeight_;
  double disparityWeight_;
  double boundaryLengthWeight_;
  double smoothRelativeWeight_;
  double inlierThreshold_;
  double hingePenalty_;
  double occlusionPenalty_;
  double impossiblePenalty_;

  // Input data
  int width_;
  int height_;
  typename bpvo::AlignedVector<float>::type inputLabImage_;
  typename bpvo::AlignedVector<float>::type initialDisparityImage_;

  // Superpixel segments
  int segmentTotal_;
  std::vector<Segment> segments_;
  int stepSize_;
  typename bpvo::AlignedVector<int>::type labelImage_;
  typename bpvo::AlignedVector<unsigned char>::type outlierFlagImage_;
  typename bpvo::AlignedVector<unsigned char>::type boundaryFlagImage_;
  std::vector<Boundary> boundaries_;
  std::vector< std::vector<int> > boundaryIndexMatrix_;
};

SpsStereo::Config::Config()
  : numInnerIterations(10),
    numOuterIterations(10),
    outputDisparityFactor(256.0),
    positionWeight(500.0),
    disparityWeight(2000.0),
    boundaryLengthWeight(1500.0),
    smoothnessWeight(400.0),
    inlierThreshold(3.0),
    hingePenalty(5.0),
    occlusionPenalty(15.0),
    impossiblePenalty(30.0) {}

struct SpsStereo::Impl
{
}; // Impl

SpsStereo::SpsStereo(Config config)
  : _config(config), _impl(new Impl) {}

SpsStereo::~SpsStereo() { delete _impl; }

void SpsStereo::compute(const cv::Mat& left, const cv::Mat& right, cv::Mat& D)
{
}


// Default parameters
const double SPSSTEREO_DEFAULT_OUTPUT_DISPARITY_FACTOR = 256.0;
const int SPSSTEREO_DEFAULT_OUTER_ITERATION_COUNT = 10;
const int SPSSTEREO_DEFAULT_INNER_ITERATION_COUNT = 10;
const double SPSSTEREO_DEFAULT_POSITION_WEIGHT = 500.0;
const double SPSSTEREO_DEFAULT_DISPARITY_WEIGHT = 2000.0;
const double SPSSTEREO_DEFAULT_BOUNDARY_LENGTH_WEIGHT = 1500.0;
const double SPSSTEREO_DEFAULT_SMOOTHNESS_WEIGHT = 400.0;
const double SPSSTEREO_DEFAULT_INLIER_THRESHOLD = 3.0;
const double SPSSTEREO_DEFAULT_HINGE_PENALTY = 5.0;
const double SPSSTEREO_DEFAULT_OCCLUSION_PENALTY = 15.0;
const double SPSSTEREO_DEFAULT_IMPOSSIBLE_PENALTY = 30.0;

// Pixel offsets of 4- and 8-neighbors
const int fourNeighborTotal = 4;
const int fourNeighborOffsetX[4] = { -1, 0, 1, 0 };
const int fourNeighborOffsetY[4] = { 0, -1, 0, 1 };
const int eightNeighborTotal = 8;
const int eightNeighborOffsetX[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
const int eightNeighborOffsetY[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };


SPSStereo::SPSStereo() : outputDisparityFactor_(SPSSTEREO_DEFAULT_OUTPUT_DISPARITY_FACTOR),
						 outerIterationTotal_(SPSSTEREO_DEFAULT_OUTER_ITERATION_COUNT),
						 innerIterationTotal_(SPSSTEREO_DEFAULT_INNER_ITERATION_COUNT),
						 positionWeight_(SPSSTEREO_DEFAULT_POSITION_WEIGHT),
						 disparityWeight_(SPSSTEREO_DEFAULT_DISPARITY_WEIGHT),
						 boundaryLengthWeight_(SPSSTEREO_DEFAULT_BOUNDARY_LENGTH_WEIGHT),
						 inlierThreshold_(SPSSTEREO_DEFAULT_INLIER_THRESHOLD),
						 hingePenalty_(SPSSTEREO_DEFAULT_HINGE_PENALTY),
						 occlusionPenalty_(SPSSTEREO_DEFAULT_OCCLUSION_PENALTY),
						 impossiblePenalty_(SPSSTEREO_DEFAULT_IMPOSSIBLE_PENALTY)
{
	smoothRelativeWeight_ = SPSSTEREO_DEFAULT_SMOOTHNESS_WEIGHT/SPSSTEREO_DEFAULT_DISPARITY_WEIGHT;
}

void SPSStereo::setOutputDisparityFactor(const double outputDisparityFactor) {
	if (outputDisparityFactor < 1) {
		throw std::invalid_argument("[SPSStereo::setOutputDisparityFactor] disparity factor is less than 1");
	}

	outputDisparityFactor_ = outputDisparityFactor;
}

void SPSStereo::setIterationTotal(const int outerIterationTotal, const int innerIterationTotal) {
	if (outerIterationTotal < 1 || innerIterationTotal < 1) {
		throw std::invalid_argument("[SPSStereo::setIterationTotal] the number of iterations is less than 1");
	}

	outerIterationTotal_ = outerIterationTotal;
	innerIterationTotal_ = innerIterationTotal;
}

void SPSStereo::setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight) {
	if (positionWeight < 0 || disparityWeight < 0 || boundaryLengthWeight < 0 || smoothnessWeight < 0) {
		throw std::invalid_argument("[SPSStereo::setWeightParameter] weight value is nagative");
	}

	positionWeight_ = positionWeight;
	disparityWeight_ = disparityWeight;
	boundaryLengthWeight_ = boundaryLengthWeight;
	smoothRelativeWeight_ = smoothnessWeight/disparityWeight;
}

void SPSStereo::setInlierThreshold(const double inlierThreshold) {
	if (inlierThreshold <= 0) {
		throw std::invalid_argument("[SPSStereo::setInlierThreshold] threshold of inlier is less than zero");
	}

	inlierThreshold_ = inlierThreshold;
}

void SPSStereo::setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty) {
	if (hingePenalty <= 0 || occlusionPenalty <= 0 || impossiblePenalty < 0) {
		throw std::invalid_argument("[SPSStereo::setPenaltyParameter] penalty value is less than zero");
	}
	if (hingePenalty >= occlusionPenalty) {
		throw std::invalid_argument("[SPSStereo::setPenaltyParameter] hinge penalty is larger than occlusion penalty");
	}

	hingePenalty_ = hingePenalty;
	occlusionPenalty_ = occlusionPenalty;
	impossiblePenalty_ = impossiblePenalty;
}

void SPSStereo::compute(const int superpixelTotal,
                        const cv::Mat& leftImage,
                        const cv::Mat& rightImage,
                        cv::Mat& segmentImage,
                        cv::Mat& disparityImage,
                        std::vector< std::vector<double> >& disparityPlaneParameters,
                        std::vector< std::vector<int> >& boundaryLabels)
{
  if (superpixelTotal < 2) {
    throw std::invalid_argument("[SPSStereo::compute] the number of superpixels is less than 2");
  }

  width_ = leftImage.cols;
  height_ = leftImage.rows;
  if (rightImage.cols != width_ || rightImage.rows != height_) {
    throw std::invalid_argument("[SPSStereo::setInputData] sizes of left and right images are different");
  }

  allocateBuffer();

  setInputData(leftImage, rightImage);
  initializeSegment(superpixelTotal);
  performSmoothingSegmentation();

  makeOutputImage(segmentImage, disparityImage);
  makeSegmentBoundaryData(disparityPlaneParameters, boundaryLabels);

  freeBuffer();
}

void SPSStereo::allocateBuffer()
{
  inputLabImage_.resize( width_ * height_ * 3);
  initialDisparityImage_.resize( width_ * height_ );
  labelImage_.resize( width_ * height_ );
  outlierFlagImage_.resize( width_ * height_ );
  boundaryFlagImage_.resize( width_ * height_ );
}

void SPSStereo::freeBuffer()
{
}

void SPSStereo::setInputData(const cv::Mat& leftImage, const cv::Mat& rightImage)
{
	setLabImage(leftImage);
	computeInitialDisparityImage(leftImage, rightImage);
}

void SPSStereo::setLabImage(const cv::Mat& leftImage)
{
	std::vector<float> sRGBGammaCorrections(256);
	for (int pixelValue = 0; pixelValue < 256; ++pixelValue)
  {
		double normalizedValue = pixelValue/255.0;
		double transformedValue = (normalizedValue <= 0.04045) ? normalizedValue/12.92 : pow((normalizedValue+0.055)/1.055, 2.4);

		sRGBGammaCorrections[pixelValue] = static_cast<float>(transformedValue);
	}

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			png::rgb_pixel rgbPixel = leftImage.get_pixel(x, y);

			float correctedR = sRGBGammaCorrections[rgbPixel.red];
			float correctedG = sRGBGammaCorrections[rgbPixel.green];
			float correctedB = sRGBGammaCorrections[rgbPixel.blue];

			float xyzColor[3];
			xyzColor[0] = correctedR*0.412453f + correctedG*0.357580f + correctedB*0.180423f;
			xyzColor[1] = correctedR*0.212671f + correctedG*0.715160f + correctedB*0.072169f;
			xyzColor[2] = correctedR*0.019334f + correctedG*0.119193f + correctedB*0.950227f;

			const double epsilon = 0.008856;
			const double kappa = 903.3;
			const double referenceWhite[3] = { 0.950456, 1.0, 1.088754 };

			float normalizedX = static_cast<float>(xyzColor[0]/referenceWhite[0]);
			float normalizedY = static_cast<float>(xyzColor[1]/referenceWhite[1]);
			float normalizedZ = static_cast<float>(xyzColor[2]/referenceWhite[2]);
			float fX = (normalizedX > epsilon) ? static_cast<float>(pow(normalizedX, 1.0/3.0)) : static_cast<float>((kappa*normalizedX + 16.0)/116.0);
			float fY = (normalizedY > epsilon) ? static_cast<float>(pow(normalizedY, 1.0/3.0)) : static_cast<float>((kappa*normalizedY + 16.0)/116.0);
			float fZ = (normalizedZ > epsilon) ? static_cast<float>(pow(normalizedZ, 1.0/3.0)) : static_cast<float>((kappa*normalizedZ + 16.0)/116.0);

			inputLabImage_[width_*3*y + 3*x] = static_cast<float>(116.0*fY - 16.0);
			inputLabImage_[width_*3*y + 3*x + 1] = static_cast<float>(500.0*(fX - fY));
			inputLabImage_[width_*3*y + 3*x + 2] = static_cast<float>(200.0*(fY - fZ));
		}
	}
}


