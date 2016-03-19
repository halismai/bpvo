#include "utils/dataset2.h"
#include "utils/file_loader.h"
#include "utils/stereo_algorithm.h"
#include "bpvo/config_file.h"
#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>

namespace bpvo {
namespace utils {

DatasetFrame::~DatasetFrame() {}
std::string DatasetFrame::filename() const { return _filename; }

Dataset::~Dataset() {}

static ImageSize GetImageSize(const DatasetFrame* f)
{
  return ImageSize(f->image().rows, f->image().cols);
}

static inline void removeWhiteSpace(std::string& s)
{
  s.erase(std::remove_if(s.begin(), s.end(),
                 [](char c) { return std::isspace<char>(c, std::locale::classic()); }),
          s.end());
}

static inline bpvo::Matrix34 set_kitti_camera_from_line(std::string line)
{
  auto tokens = bpvo::splitstr(line);
  THROW_ERROR_IF( tokens.empty() || tokens[0].empty() || tokens[0][0] != 'P',
                 "invalid calibration line");
  THROW_ERROR_IF( tokens.size() != 13, "wrong line length" );

  std::vector<float> vals;
  for(size_t i = 1; i < tokens.size(); ++i)
    vals.push_back(bpvo::str2num<float>(tokens[i]));

  bpvo::Matrix34 ret;
  for(int r = 0, i = 0; r < ret.rows(); ++r)
    for(int c = 0; c < ret.cols(); ++c, ++i)
      ret(r,c) = vals[i];

  return ret;
}

static StereoCalibration
LoadKittiCalibration(std::string filename)
{
  std::ifstream ifs(filename);
  THROW_ERROR_IF( !ifs.is_open(), "failed to open calib.txt" );

  StereoCalibration ret;

  Matrix34 P1, P2;
  std::string line;

  // the first camera
  std::getline(ifs, line);
  P1 = set_kitti_camera_from_line(line);

  std::getline(ifs, line);
  P2 = set_kitti_camera_from_line(line);

  ret.K = P1.block<3,3>(0,0);
  ret.baseline =  -P2(0,3) / P2(0,0);

  return ret;
}

static inline
void ToGray(const cv::Mat& src, cv::Mat& dst)
{
  switch(src.type())
  {
    case CV_8UC1:
      dst = src;
      break;
    case CV_8UC3:
      cv::cvtColor(src, dst, CV_BGR2GRAY);
      break;
    case CV_8UC4:
      cv::cvtColor(src, dst, CV_BGRA2GRAY);
      break;
    default:
      THROW_ERROR("unsupported image type");
  }
}

class Dataset::Impl
{
 public:
  Impl(std::string conf_file) { init(conf_file); }

  void init(std::string conf_file)
  {
    try {
      std::cout << "Impl ConfigFile from: " << conf_file << std::endl;

      ConfigFile cf(conf_file);
      auto type = cf.get<std::string>("DatasetType");

      if(icompare("stereo", type))
        _type = Dataset::Type::Stereo;
      else if(icompare("disparity", type))
        _type = Dataset::Type::Disparity;
      else if(icompare("depth", type))
        _type = Dataset::Type::Depth;
      else
        THROW_ERROR(Format("unknown data set type %s\n", type.c_str()).c_str());

      if(_type == Dataset::Type::Stereo) {
        _stereo_alg = make_unique<StereoAlgorithm>(cf);
      } else {
        _disparity_scale = cf.get<double>("DisparityScale");
      }

      // image scaling (down/up) for experiments with resolution
      _scale_by = cf.get<double>("ScaleBy", 1.0);

      std::cout << "ScaleBy: " << _scale_by << "\n";

      auto name = cf.get<std::string>("DatasetName", "Generic");
      if(icompare("tsukuba", name) || icompare("tsukuba_synthetic", name))
      {
        printf("initTsukubaSynthetic\n");
        initTsukubaSynthetic(cf);
      }
      else if(icompare("tsukuba_stereo", name))
      {
        printf("initTsukubaStereo\n");
        initTsukubaStereo(cf);
      }
      else if(icompare("kitti", name))
      {
        printf("initKitti\n");
        initKitti(cf);
      }
      else if(icompare("tunnel", name))
      {
        printf("initTunnel\n");
        initTunnel(cf);
      }
      else if(icompare("Generic", name))
      {
        printf("initGeneric\n");
        initGeneric(cf);
      }
      else
      {
        THROW_ERROR(Format("unknown dataset name %s\n", name.c_str()).c_str());
      }

      _name = name;
    } catch(const std::exception& ex) {
      Warn("failed to init data set %s\n", ex.what());
      throw ex;
    }
  }

  UniquePointer<DatasetFrame> getFrame(int f_i)
  {
    switch(_type)
    {
      case Dataset::Type::Stereo:
        return getStereoFrame(f_i);
      case Dataset::Type::Disparity:
        return getDisparityFrame(f_i);
      case Dataset::Type::Depth:
        return getDepthFrame(f_i);
      default:
        THROW_ERROR("unkonwn type!");
    }
  }

 protected:
  void initTsukubaSynthetic(const ConfigFile& cf)
  {
    auto root_dir = fs::expand_tilde(
        cf.get<std::string>("DataSetRootDirectory", "~/home/data/NewTsukubaStereoDataset"));
    THROW_ERROR_IF( !fs::exists(root_dir), "DataSetRootDirectory does not exist" );

    auto illumination = cf.get<std::string>("Illumination", "fluorescent");
    auto frame_start = cf.get<int>("FirstFrameNumber", 1); // tsukuba starts from 1

    auto img_fmt = Format("illumination/%s/left/tsukuba_%s_L_%s.png",
                          illumination.c_str(), illumination.c_str(), "%05d");
    auto dmap_fmt = Format("groundtruth/disparity_maps/left/tsukuba_disparity_L_%s.png", "%05d");

    this->_left_filenames = make_unique<FileLoader>(root_dir, img_fmt, frame_start);
    this->_disparity_filenames = make_unique<FileLoader>(root_dir, dmap_fmt, frame_start);

    _image_size = ImageSize(480, 640);
    _calib.K <<
        615.0, 0.0, 320.0,
        0.0, 615.0, 240.0,
        0.0, 0.0, 1.0;
    _calib.baseline = 0.1;

    scaleCalibration();
  }

  void initTsukubaStereo(const ConfigFile& cf)
  {
    auto root_dir = fs::expand_tilde(
        cf.get<std::string>("DataSetRootDirectory", "~/home/data/NewTsukubaStereoDataset"));
    THROW_ERROR_IF( !fs::exists(root_dir), "DataSetRootDirectory does not exist" );

    auto illumination = cf.get<std::string>("Illumination", "fluorescent");
    auto frame_start = cf.get<int>("FirstFrameNumber", 1); // tsukuba starts from 1

    auto left_fmt = Format("illumination/%s/left/tsukuba_%s_L_%s.png",
                           illumination.c_str(), illumination.c_str(), "%05d");
    auto right_fmt = Format("illumination/%s/right/tsukuba_%s_R_%s.png",
                           illumination.c_str(), illumination.c_str(), "%05d");

    std::cout << "init with " << left_fmt << std::endl;

    this->_left_filenames = make_unique<FileLoader>(root_dir,  left_fmt, frame_start);
    this->_right_filenames = make_unique<FileLoader>(root_dir, right_fmt, frame_start);

    _image_size = ImageSize(480, 640);
    _calib.K <<
        615.0, 0.0, 320.0,
        0.0, 615.0, 240.0,
        0.0, 0.0, 1.0;
    _calib.baseline = 0.1;
  }

  void initKitti(const ConfigFile& cf)
  {
    THROW_ERROR_IF(_type != Dataset::Type::Stereo, "kitti data is stereo!");

    auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
    auto sequence = cf.get<int>("SequenceNumber");

    auto left_fmt = Format("sequences/%02d/image_0/%s.png", sequence, "%06d");
    auto right_fmt = Format("sequences/%02d/image_1/%s.png", sequence, "%06d");
    auto frame_start = cf.get<int>("FirstFrameNumber", 0);

    this->_left_filenames = make_unique<FileLoader>(root_dir, left_fmt, frame_start);
    this->_right_filenames = make_unique<FileLoader>(root_dir, right_fmt, frame_start);

    auto frame = this->getFrame(0);
    THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
    this->_image_size = GetImageSize(frame.get());

    auto calib_fn = Format("%s/sequences/%02d/calib.txt", root_dir.c_str(), sequence);
    _calib = LoadKittiCalibration(calib_fn);
  }

  void initTunnel(const ConfigFile& cf)
  {
    auto calib_fn = cf.get<std::string>("CalibrationFile");
    std::ifstream ifs(calib_fn);
    THROW_ERROR_IF(!ifs.is_open(), "failed to open calibration file");

    std::string line;
    std::getline(ifs, line); // version
    if(line == "CRL Camera Config") {
      std::getline(ifs, line);
      removeWhiteSpace(line);
      int rows = 0, cols = 0;
      sscanf(line.c_str(), "Width,height:%d,%d", &cols, &rows);
      this->_image_size.rows = rows;
      this->_image_size.cols = cols;
      std::getline(ifs, line); // fps
      std::getline(ifs, line);
      removeWhiteSpace(line);
      float fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
      sscanf(line.c_str(), "fx,fy,cx,cy:%f,%f,%f,%f", &fx, &fy, &cx, &cy);
      _calib.K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
      std::getline(ifs, line);
      removeWhiteSpace(line);
      sscanf(line.c_str(), "xyzrpq:%f", &_calib.baseline);
      if(_calib.baseline < 0.0f)
        _calib.baseline *= -1.0f;
    } else {
      std::getline(ifs, line); // the camera calibration

      int rows = 0, cols = 0;
      float fx = 0.0, fy = 0.0f, cx = 0.0f, cy = 0.0f;
      float dist_coeffs[6];

      removeWhiteSpace(line);
      sscanf(line.c_str(), "CameraIntrinsicsPlumbBob{%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}",
             &cols, &rows, &fx, &fy, &cx, &cy,
             &dist_coeffs[0], &dist_coeffs[1], &dist_coeffs[2],
             &dist_coeffs[3], &dist_coeffs[4], &dist_coeffs[5]);

      this->_image_size.rows = rows;
      this->_image_size.cols = cols;

      _calib.K <<
          fx, 0.0, cx,
          0.0, fy, cy,
          0.0, 0.0, 1.0;

      std::getline(ifs, line); // CameraIntrinsicsPlumbBob second line
      THROW_ERROR_IF(line.empty(), "malformatted line in calibration file");

      for(int i = 0; i < 4; ++i) {
        std::getline(ifs, line);
        THROW_ERROR_IF(line.empty(), "malformatted line in calibration file");
      }

      std::getline(ifs, line);
      THROW_ERROR_IF(line.empty(), "malformatted line in calibration file");
      float dummy = 0.0f;
      removeWhiteSpace(line);
      sscanf(line.c_str(), "Transform3D(%f,%f,%f,%f",
             &dummy, &dummy, &dummy, &_calib.baseline);

      if(_calib.baseline < 0)
        _calib.baseline *= -1;
    }
  }

  void initGeneric(const ConfigFile& /*cf*/)
  {
  }

  void scaleCalibration()
  {
    _calib.scale(_scale_by);
  }

 public:
  inline StereoCalibration stereroCalibration() const { return _calib; }
  inline ImageSize imageSize() const { return _image_size; }
  inline std::string name() const { return _name;  }
  inline Dataset::Type type() const { return _type; }

 private:
  UniquePointer<FileLoader> _left_filenames;
  UniquePointer<FileLoader> _right_filenames;
  UniquePointer<FileLoader> _disparity_filenames;
  UniquePointer<StereoAlgorithm> _stereo_alg;

  Dataset::Type _type;
  std::string _name;
  ImageSize _image_size;

  StereoCalibration _calib;

  double _scale_by = 1.0;
  double _disparity_scale = 1.0 / 16.0;


  struct StereoFrame : public DatasetFrame
  {
    StereoFrame(std::string left, std::string right)
    {
      I_orig[0] = cv::imread(left, cv::IMREAD_UNCHANGED);
      I_orig[1] = cv::imread(right, cv::IMREAD_UNCHANGED);

      THROW_ERROR_IF(I_orig[0].empty() || I_orig[1].empty(),
                     Format("could not read frame from:%s\n%s\n",
                            left.c_str(), right.c_str()).c_str());

      ToGray(I_orig[0], I_gray[0]);
      ToGray(I_orig[1], I_gray[1]);

      this->_filename = left;
    }

    virtual ~StereoFrame() {}

    inline const cv::Mat& image() const { return I_gray[0]; }
    inline const cv::Mat& disparity() const { return dmap; }

    cv::Mat I_orig[2];
    cv::Mat I_gray[2];
    cv::Mat dmap;
  }; // StereoFrame

  UniquePointer<DatasetFrame> getStereoFrame(int f_i)
  {
    StereoFrame ret(_left_filenames->operator[](f_i),
                    _right_filenames->operator[](f_i));

    _stereo_alg->run(ret.I_gray[0], ret.I_gray[1], ret.dmap);
    return UniquePointer<DatasetFrame>(new StereoFrame(ret));
  }

  struct DisparityFrame : public DatasetFrame
  {
    DisparityFrame(std::string image_fn, std::string d_fn, float d_scale)
    {
      this->_filename = image_fn;

      I_orig = cv::imread(image_fn, cv::IMREAD_UNCHANGED);
      THROW_ERROR_IF(I_orig.empty(), Format("failed to raad image from '%s'\n",
                                            image_fn.c_str()).c_str());
      ToGray(I_orig, I_gray);

      D = cv::imread(d_fn, cv::IMREAD_UNCHANGED);
      THROW_ERROR_IF(D.empty(), Format("Failed to read disparity from %s\n",
                                       d_fn.c_str()).c_str());

      THROW_ERROR_IF( D.channels() > 1, "disparity must be a single channel" );
      D.convertTo(D, CV_32FC1, d_scale, 0.0);
    }

    virtual ~DisparityFrame() {}

    inline const cv::Mat& image() const { return I_gray; }
    inline const cv::Mat& disparity() const { return D; }

    cv::Mat I_orig;
    cv::Mat I_gray;
    cv::Mat D;
  }; // DisparityFrame

  UniquePointer<DatasetFrame> getDisparityFrame(int f_i)
  {
    auto image_fn = _left_filenames->operator[](f_i);
    auto d_fn = _disparity_filenames->operator[](f_i);
    return UniquePointer<DatasetFrame>(
        new DisparityFrame(image_fn, d_fn, _disparity_scale));
  }

  UniquePointer<DatasetFrame> getDepthFrame(int f_i)
  {
    // TODO this the same thing as getDisparityFrame, just in case we need to
    // handle depth slightly differently
    auto image_fn = _left_filenames->operator[](f_i);
    auto d_fn = _disparity_filenames->operator[](f_i);
    return UniquePointer<DatasetFrame>(
        new DisparityFrame(image_fn, d_fn, _disparity_scale));
  }
}; // Dataset::Impl


UniquePointer<DatasetFrame> Dataset::getFrame(int f_i)
{
  return _impl->getFrame(f_i);
}

ImageSize Dataset::imageSize() const { return _impl->imageSize(); }

StereoCalibration Dataset::calibration() const { return _impl->stereroCalibration(); }

Dataset::Type Dataset::type() const { return _impl->type(); }

std::string Dataset::name() const { return _impl->name(); }

UniquePointer<Dataset> Dataset::Create(std::string conf_fn)
{
  UniquePointer<Dataset> ret(new Dataset);
  ret->_impl = make_unique<Dataset::Impl>(conf_fn);
  return ret;
}


} // utils
} // bpvo

