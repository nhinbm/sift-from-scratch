#include <iostream>
#include <math.h>
#include <tuple>
#include <algorithm> 
#include <opencv2/opencv.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;


// Helper function: Filter keypoints along edge
bool filter_keypoints_along_edges(KeyPoint &kp, Mat &image) {
  int x = kp.pt.x;
  int y = kp.pt.y;
  // Calculate the Hessian matrix
  double dxx = image.at<float>(y, x + 1) + image.at<float>(y, x - 1) - 2 * image.at<float>(y, x);
  double dyy = image.at<float>(y + 1, x) + image.at<float>(y - 1, x) - 2 * image.at<float>(y, x);
  double dxy = (image.at<float>(y + 1, x + 1) - image.at<float>(y + 1, x - 1) - image.at<float>(y - 1, x + 1) + image.at<float>(y - 1, x - 1)) / 4.0;
  // Compute eigenvalues
  double trace = dxx + dyy;
  double determinant = dxx * dyy - dxy * dxy;
  double discriminant = trace * trace - 4 * determinant;
  // Compute eigenvalues
  double lambda1 = (trace + sqrt(discriminant)) / 2.0;
  double lambda2 = (trace - sqrt(discriminant)) / 2.0;
  // Compute the ratio of the eigenvalues
  double ratio = lambda1 / lambda2;
  // Check if the keypoint is along an edge
  if (ratio < 10) {
    return false;
  }
  return true;
}


// Helper function: Filter keypoints from contrast
bool filter_low_contrast_keypoints(KeyPoint &kp, Mat &image) {
  int x = kp.pt.x;
  int y = kp.pt.y;
  // Compute the first and second derivatives at the keypoint location
  double dx = (image.at<uchar>(y, x + 1) - image.at<uchar>(y, x - 1)) / 2.0;
  double dy = (image.at<uchar>(y + 1, x) - image.at<uchar>(y - 1, x)) / 2.0;

  double dxx = image.at<uchar>(y, x + 1) + image.at<uchar>(y, x - 1) - 2 * image.at<uchar>(y, x);
  double dyy = image.at<uchar>(y + 1, x) + image.at<uchar>(y - 1, x) - 2 * image.at<uchar>(y, x);

  double D = sqrt(dx * dx + dy * dy);
  double Dxx = dxx + dyy;

  if (abs(Dxx) > numeric_limits<double>::epsilon()) { 
    double refined_x = x - D / Dxx;
    if (refined_x >= 0 && refined_x < image.cols && abs(dx) < 0.5) {
      // Compute the intensity at the refined keypoint location
      double refined_intensity = image.at<uchar>(y, int(refined_x));
      // Check if the intensity at the refined keypoint location satisfies the contrast threshold
      if (abs(refined_intensity - image.at<uchar>(y, x)) < 0.03) {
        return false;
      }
    }
  }
  return true;
}


// Step 1.1: Constructing the Scale space
void construct_scale_space(vector<vector<Mat>> &scale_space_octaves, Mat &image, int num_octaves=4, int num_scales=5, double sigma=1.6, double k=sqrt(2), int kernel_size=5) {
  Mat base_image = image;
  for (int octave = 0; octave < num_octaves; octave++) {
    vector<Mat> scale_space;
    for(int scale = 0; scale < num_scales; scale++) {
      // Calculate the sigma value for this scale
      float sigma_scale = pow(k, scale) * sigma;
      // Gaussian image
      Mat gau_image;
      GaussianBlur(base_image, gau_image, Size(5, 5), sigma_scale);
      scale_space.push_back(gau_image);
    }
    scale_space_octaves.push_back(scale_space);
    // Resize the image for the next octave
    resize(base_image, base_image, Size(base_image.cols / 2, base_image.rows / 2));
  }
}


// Step 1.2: Laplacian of Gaussian approximation kernel
void compute_dog_space(vector<vector<Mat>> &dog_space_octaves, vector<vector<Mat>> &scale_space_octaves) {
  for (int octave = 0; octave < scale_space_octaves.size(); octave++) {
    vector<Mat> dog_space;
    for (int scale = 1; scale < scale_space_octaves[0].size(); scale++) {
      Mat diff = scale_space_octaves[octave][scale] - scale_space_octaves[octave][scale - 1];
      dog_space.push_back(diff);
    }
    dog_space_octaves.push_back(dog_space);
  }
}


// Step 1.3: Keypoint localization
void localize_keypoints(vector<KeyPoint> &keypoints, vector<vector<Mat>> &dog_spaces_octaves) {
  for (int octave = 0; octave < dog_spaces_octaves.size(); octave++) {
    // Iterate over DoG images
    for (int scale = 1; scale < dog_spaces_octaves[0].size() - 1; scale++) {
      Mat current = dog_spaces_octaves[octave][scale];
      Mat prev = dog_spaces_octaves[octave][scale - 1];
      Mat next = dog_spaces_octaves[octave][scale + 1];
      // Information of input image
      int width = current.cols;
      int height = current.rows;
      int width_step = current.step[0];
      int n_channels= current.step[1];
      // Iterate over the pixels in the DoG image
      uchar* p_data_cur = (uchar*)current.data + width_step + n_channels;
      uchar* p_data_prev = (uchar*)prev.data + width_step + n_channels;
      uchar* p_data_next = (uchar*)next.data + width_step + n_channels;
      for (int y = 1; y < height - 1; y++, p_data_cur += width_step, p_data_prev += width_step, p_data_next += width_step) {
        uchar* p_row_cur = p_data_cur;
        uchar* p_row_prev = p_data_prev;
        uchar* p_row_next = p_data_next;
        for (int x = 1; x < width - 1; x++, p_row_cur += n_channels, p_row_prev += n_channels, p_row_next += n_channels) {
          // Remove low contrast points
          float value = p_row_cur[0];
          if (abs(p_row_cur[0]) < 0.8*0.015) continue;
          // Check if the pixel is a local extremum
          bool is_min = true, is_max = true;
          float neighbor = 0.0;
          for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
              // Check previous scale
              neighbor = p_row_prev[dx * width_step + dy * n_channels];
              if (neighbor > value) is_max = false;
              if (neighbor < value) is_min = false;
              // Check current scale
              neighbor = p_row_cur[dx * width_step + dy * n_channels];
              if (neighbor > value) is_max = false;
              if (neighbor < value) is_min = false;
              // Check next scale
              neighbor = p_row_next[dx * width_step + dy * n_channels];
              if (neighbor > value) is_max = false;
              if (neighbor < value) is_min = false;

              if (!is_min && !is_max) break;
            }
            if (!is_min && !is_max) break;
          }
          KeyPoint kp(x, y, 1, -1, 0, octave, scale);
          if (filter_low_contrast_keypoints(kp, current) && filter_keypoints_along_edges(kp, current)) {
            keypoints.push_back(kp);
          }
        }
      }
    }
  }
}


// Step 1.4: Orientation assignment
void assign_orientation(vector<KeyPoint> &keypoints, vector<vector<Mat>> &scale_space_octaves) {
  for(int i = 0; i < keypoints.size(); i++) {
    const int num_bins = 36;
    vector<float> hist(num_bins, 0.0);
    KeyPoint &kp = keypoints[i];
    Mat image = scale_space_octaves[kp.octave][kp.class_id];
    for (int y = -4; y < 4; y++) {
      for (int x = -4; x < 4; x++) {
        int dx = kp.pt.x + x;
        int dy = kp.pt.y + y;
        if (dx - 1 >= 0 && dx + 1 < image.cols && dy - 1 >= 0 && dy + 1 < image.rows) {
          float gx = image.at<uchar>(dy, dx + 1) - image.at<uchar>(dy, dx - 1);
          float gy = image.at<uchar>(dy + 1, dx) - image.at<uchar>(dy - 1, dx);
          float magnitude = gx * gx + gy * gy;
          float angle = (float)(atan((float)gy / gx) * 180) / M_PI;
          if (angle < 0) angle += 360.0;
          int bin = angle / (360.0 / 36);
          hist[bin] += magnitude;
        }
      }
    }
    // Find the dominant orientation(s)
    int index = *minmax_element(hist.begin(), hist.end()).second;
    kp.angle = index * 10;
  }
}


// Step 1.5: Compute an descriptor of keypoint
Mat compute_descriptor(vector<KeyPoint> &keypoints, vector<vector<Mat>> &scale_space_octaves) {
  Mat descriptor = Mat(keypoints.size(), 128, CV_32F);
  const int num_bins = 8;
  for(int i = 0; i < keypoints.size(); i++) {
    KeyPoint &kp = keypoints[i];
    Mat image = scale_space_octaves[kp.octave][kp.class_id];
    int count = 0;
    for (int y = -2; y < 2; y++) {
      for (int x = -2; x < 2; x++) {
        vector<float> hist(num_bins, 0.0);
        for (int y_sub = 0; y_sub < 4; y_sub++) {
          for (int x_sub = 0; x_sub < 4; x_sub++) {
            int dx = kp.pt.x + x * 4 + x_sub;
            int dy = kp.pt.y + y * 4 + y_sub;
            if (dx - 1 >= 0 && dx + 1 < image.cols && dy - 1 >= 0 && dy + 1 < image.rows) {
              float gx = image.at<uchar>(dy, dx + 1) - image.at<uchar>(dy, dx - 1);
              float gy = image.at<uchar>(dy + 1, dx) - image.at<uchar>(dy - 1, dx);
              float magnitude = gx * gx + gy * gy;
              float angle = (float)(atan((float)gy / gx) * 180) / M_PI;
              if (angle < 0) angle += 360.0;
              int bin = angle / (360.0 / 8);
              hist[bin] += magnitude;
            }
          }
        }
        for (int k = 0; k < num_bins; k++) {
          descriptor.at<float>(i, count++) = hist[k];
        }
      }
    }
  }
  return descriptor;
}


// Step 1: SIFT create
void detect_keypoints_from_scratch(Mat &img_scene, Mat &img_template, vector<KeyPoint> &keypoint_scene, vector<KeyPoint> &keypoint_template, Mat &descriptor_scene, Mat &descriptor_template) {
  // -- Step 1.1: Make grayscale
  Mat gray_scene, gray_template;
  cvtColor(img_scene, gray_scene, COLOR_BGR2GRAY);
  cvtColor(img_template, gray_template, COLOR_BGR2GRAY);

  // -- Step 1.2: Constructing the Scale space
  vector<vector<Mat>> scale_space_scene, scale_space_template;
  construct_scale_space(scale_space_scene, gray_scene);
  construct_scale_space(scale_space_template, gray_template);

  // -- Step 1.3: Laplacian of Gaussian approximation kernel
  vector<vector<Mat>> dog_space_scene, dog_space_template;
  compute_dog_space(dog_space_scene, scale_space_scene);
  compute_dog_space(dog_space_template, scale_space_template);

  // -- Step 1.4: Keypoint localization
  localize_keypoints(keypoint_scene, dog_space_scene);
  localize_keypoints(keypoint_template, dog_space_template);

  // -- Step 1.5: Orientation assignment
  assign_orientation(keypoint_scene, scale_space_scene);
  assign_orientation(keypoint_template, scale_space_template);

  // -- Step 1.6: Build descriptor
  descriptor_scene = compute_descriptor(keypoint_scene, scale_space_scene);
  descriptor_template = compute_descriptor(keypoint_template, scale_space_template);
}


// Step 1 (additional): Detect the keypoints using SURF Detector, compute the descriptors
void detect_keypoints_by_surf (Mat &img_scene, Mat &img_template, vector<KeyPoint> &keypoint_scene, vector<KeyPoint> &keypoint_template, Mat &descriptor_scene, Mat &descriptor_template) {
  int minHessian = 400;
  Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);
  detector->detectAndCompute( img_scene, noArray(), keypoint_scene, descriptor_scene );
  detector->detectAndCompute( img_template, noArray(), keypoint_template, descriptor_template );
}


// Step 2: Find the closest matches between descriptors from the first image to the second
vector<vector<DMatch>> match_descriptors (Mat &descriptor_scene, Mat &descriptor_template) {
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  vector<vector<DMatch>> knn_matches;
  matcher->knnMatch(descriptor_template, descriptor_scene, knn_matches, 2);
  return knn_matches;
}


// Step 3: Filter good matches
vector<DMatch> filter_matches (vector<vector<DMatch>> &knn_matches) {
  const float ratio_thresh = 0.9f;
  vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  return good_matches;
}

// Step 4: Homography to find a known object
void show_results(Mat &image_scene, Mat &image_template, vector<KeyPoint> &keypoint_scene, vector<KeyPoint> &keypoint_template, vector<DMatch> &good_matches, string output_path) {
  if (good_matches.size() > 30) {
    // vector for training and query keypoints for good matches
    vector<Point2f> tp;
    vector<Point2f> qp;
    for (size_t k = 0; k < good_matches.size(); k++) {
			qp.push_back(keypoint_scene[good_matches[k].trainIdx].pt);
			tp.push_back(keypoint_template[good_matches[k].queryIdx].pt);
		}
    // homograph transform
    Mat H = findHomography(tp, qp, RANSAC);
    // training image size
    Size s = image_template.size();
    int rows = s.height;
		int cols = s.width;
    // training object vertices starting from top left and then moving clockwise
    vector<Point2f> trainingBorder(4);
    trainingBorder[0] = Point(0, 0);
		trainingBorder[1] = Point(cols - 1, 0);
		trainingBorder[2] = Point(cols - 1, rows - 1);
		trainingBorder[3] = Point(0, rows - 1);
    // query object vertices
		vector<Point2f> QueryBorder(4);
		perspectiveTransform(trainingBorder, QueryBorder, H);
    // draw a green colored border around the object
		line(image_scene, QueryBorder[0], QueryBorder[1], Scalar(0, 255, 0), 4);
		line(image_scene, QueryBorder[1], QueryBorder[2], Scalar(0, 255, 0), 4);
		line(image_scene, QueryBorder[2], QueryBorder[3], Scalar(0, 255, 0), 4);
		line(image_scene, QueryBorder[3], QueryBorder[0], Scalar(0, 255, 0), 4);
  }

  // Draw matches
  Mat img_matches;
  drawMatches(image_template, keypoint_template, image_scene, keypoint_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  imshow("Good Matches", img_matches);
  imwrite(output_path, img_matches);
  waitKey(0);
}




int main(int argc, char* argv[]) {
  // Information of command line arguments
  std::string command = argv[1];
  std::string template_path = argv[2];
  std::string scene_path = argv[3];
  std::string output_path = argv[4];

  if (command == "-sift") {
    // Read image
    Mat img_scene = imread(scene_path);
    Mat img_template = imread(template_path);

    // Check file existed
    if (img_scene.empty() || img_template.empty()) {
      cout << "Could not open the image!" << endl;
      return -1;
    }

    // Step 1: From scratch
    vector<KeyPoint> keypoint_scene, keypoint_template;
    Mat descriptor_scene, descriptor_template;
    detect_keypoints_from_scratch(img_scene, img_template, keypoint_scene, keypoint_template, descriptor_scene, descriptor_template);
    
    // Step 1: From opencv
    detect_keypoints_by_surf(img_scene, img_template, keypoint_scene, keypoint_template, descriptor_scene, descriptor_template);

    // Step 2: 
    vector<vector<DMatch>> knn_matches = match_descriptors(descriptor_scene, descriptor_template);

    // Step 3:
    vector<DMatch> good_matches = filter_matches(knn_matches);

    // Step 4:
    show_results(img_scene, img_template, keypoint_scene, keypoint_template, good_matches, output_path);

  } else {
    cout << "We do not have this command!" << endl;
    return -1;
  }

  return 0;
}
