#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>

cv::Mat F_Matrix_Eight_Point(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2) {
    // Check if the number of points is at least 8
    if (points1.size() < 8 || points2.size() < 8) {
        std::cerr << "Insufficient number of points for computing Fundamental Matrix." << std::endl;
        return cv::Mat();
    }

    // Construct the A matrix
    cv::Mat A(points1.size(), 9, CV_64F);
    for (size_t i = 0; i < points1.size(); ++i) {
        const double u1 = points1[i].x;
        const double v1 = points1[i].y;
        const double u2 = points2[i].x;
        const double v2 = points2[i].y;
        A.at<double>(i, 0) = u1 * u2;
        A.at<double>(i, 1) = v1 * u2;
        A.at<double>(i, 2) = u2;
        A.at<double>(i, 3) = u1 * v2;
        A.at<double>(i, 4) = v1 * v2;
        A.at<double>(i, 5) = v2;
        A.at<double>(i, 6) = u1;
        A.at<double>(i, 7) = v1;
        A.at<double>(i, 8) = 1.0;
    }

    // Compute SVD of A
    cv::SVD svd(A, cv::SVD::FULL_UV);

    // Extract the last column of V (which corresponds to the smallest singular value)
    cv::Mat F = svd.vt.row(svd.vt.rows - 1).reshape(0, 3);

    // Enforce rank 2 constraint on F
    cv::SVD rank2_svd(F, cv::SVD::FULL_UV);
    cv::Mat singular_values = rank2_svd.w;
    singular_values.at<double>(2) = 0.0;
    F = rank2_svd.u * cv::Mat::diag(singular_values) * rank2_svd.vt;

    // Normalize the matrix
    F /= F.at<double>(2, 2);

    return F;
}

int main()
{
    cv::Mat img1 = cv::imread("../img1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../img2.png", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find the images!\n";
        return -1;   
    }

    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    
    // Convert descriptors to type CV_32F for FLANN-based matcher
    if (descriptors1.type() != CV_32F) descriptors1.convertTo(descriptors1, CV_32F);
    if (descriptors2.type() != CV_32F) descriptors2.convertTo(descriptors2, CV_32F);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2); // Find 2 nearest matches

    // Lowe's ratio test to filter matches
    const float ratio_thresh = 0.9f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // Extract location of good matches
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    // Visualize the inlier matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches", img_matches);
    cv::waitKey();

    // Compute the Fundamental Matrix using the custom function
    cv::Mat fundamentalMatrix = F_Matrix_Eight_Point(points1, points2);
    std::cout << fundamentalMatrix << std::endl;
    
    return 0;
}
