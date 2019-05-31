#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    default_random_engine rng;
    rng.seed(chrono::system_clock::now().time_since_epoch().count());

    // read the background image, forground is blotted out in pure red
    Mat image;
    image = imread("coos-bg.png", CV_LOAD_IMAGE_COLOR);
	if (!image.data) {
		cerr << "could not load image" << endl;
		return 1;
	}	
	cout << "Read image " << image.cols << "x" << image.rows << " pixels" << endl;

    // extract all background pixels
    Vec3b red(0, 0, 255);
    vector<Vec3b> bgpixels;
    for (int x = 0; x < image.cols; ++x) {
        for (int y = 0; y < image.rows; ++y) {
            Vec3b color = image.at<Vec3b>(y, x);
            if (color != red) {
                bgpixels.push_back(color);
            }
        }
    }
    cout << "Found " << bgpixels.size() << " background pixels, out of " << (image.cols*image.rows) << endl;

    cout << "cutting down to save time...." << endl;
    shuffle(bgpixels.begin(), bgpixels.end(), rng);
    bgpixels.resize(2000);

    // k-means cluster them
    for (int n = 2; n < 3; ++n) {
        float prev_tsi = -1;
        vector<Vec<float,3>> prev_centers(n, {0, 0, 0});
        for (int reps = 0; reps < 1; reps++) {
            // draw n random ones as cluster centers
            vector<Vec<float,3>> centers(n, {0, 0, 0});
            uniform_int_distribution<int> uni_dist(0, bgpixels.size()-1);
            centers[0] = bgpixels[uni_dist(rng)];
            cout << "initial center 0: " << centers[0] << endl;
            for (int i = 1; i < n; i++) {
                // try a handful of different ones to avoid pathological cases
                float max_score = -1;
                for (int j = 0; (j < 25) || (max_score < 0.1); ++j) {    
                    Vec<float,3> cc = bgpixels[uni_dist(rng)];
                    float min_dist = 255+255+255;
                    for (int k = 0; k < i; k++) {
                        float dist = norm(cc-centers[k]);
                        if (dist < min_dist) {
                            min_dist = dist;
                        }
                    }
                    if (min_dist > max_score) {
                        centers[i] = cc;
                        max_score = min_dist;
                    }
                }
                cout << "initial center " << i << ": " << centers[i] << endl;
            }

            bool converged = false;
            do {
                // for each pixel, find which center is the nearest, and sum the values
                // for each new center
                vector<Vec<float, 3>> new_centers(n, {0, 0, 0});
                vector<int> new_center_counts(n, 0);
                for (int j = 0; j < bgpixels.size(); ++j) {
                    Vec<float,3> pixel = bgpixels[j];
                    int selected = -1;
                    float prev_distance = 255+255+255; // way larger than the diagonal of the color space
                    for (int i = 0; i < n; ++i) {
                        Vec<float,3> cc = centers[i];
                        float distance = norm(pixel-cc);
                        if (distance < prev_distance) {
                            selected = i;
                            prev_distance = distance;
                        }
                    }
                    new_centers[selected] += pixel;
                    new_center_counts[selected]++;
                }
                // compute the new centers
                float difference = 0;
                int active_count = 0;
                for (int i = 0; i < n; ++i) {
                    Vec<float, 3> nc(new_centers[i][0] / new_center_counts[i], 
                            new_centers[i][1] / new_center_counts[i], 
                            new_centers[i][2] / new_center_counts[i]);
                    if (new_center_counts[i]) {
                        difference += norm(centers[i]-nc);
                        active_count++;
                    }
                    centers[i] = nc;
                    cout << "  now " << nc << " " << new_center_counts[i] << endl;
                }
                cout << "difference to previous step: " << (difference / active_count) << endl;
                if (difference < 0.01) {
                    converged = true;
                }
            } while (!converged);

            // compute the silhouette 
            vector<vector<Vec3b>> clusters(n);
            for (int j = 0; j < bgpixels.size(); ++j) {
                Vec<float,3> pixel = bgpixels[j];
                int selected = -1;
                float prev_distance = 255+255+255; // way larger than the diagonal of the color space
                for (int i = 0; i < n; ++i) {
                    Vec<float,3> cc = centers[i];
                    float distance = norm(pixel-cc);
                    if (distance < prev_distance) {
                        selected = i;
                        prev_distance = distance;
                    }
                }
                clusters[selected].push_back(bgpixels[j]);
            }
            // for each pixel
            float tsi = 0;
            int count = 0;
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < clusters[i].size(); ++a) {
                    Vec<float,3> pa = clusters[i][a];
                    // average distance to other pixels in the same cluster
                    float ai = 0;
                    for (int b = 0; b < clusters[i].size(); ++b) {
                        Vec<float,3> pb = clusters[i][b];
                        ai += norm(pb-pa);
                    }
                    ai /= clusters[i].size() - 1;
                    // minimum distance to other clusters
                    float bi = 255+255+255;
                    for (int j = 0; j < n; ++j) {
                        if (j != i) {
                            float cbi = 0;
                            for (int b = 0; b < clusters[j].size(); ++b) {
                                Vec<float,3> pb = clusters[j][b];
                                cbi += norm(pb-pa);
                            }
                            cbi /= clusters[j].size();
                            if (cbi < bi) {
                                bi = cbi;
                            }
                        }
                    }
                    float si = (bi - ai) / max(ai, bi);
                    if (clusters[i].size() == 1) {
                        si = 0;
                    }
                    tsi += si;
                    count++;
                }
            }
            tsi /= count;
            if (tsi > prev_tsi) {
                prev_tsi = tsi;
                prev_centers = centers;
            }
            cout << endl;
            // compute covariance matrix for each cluster
            for (int i = 0; i < n; ++i) {
                Mat covariance(3, 3, CV_32F, 0.0);
                for (int a = 0; a < clusters[i].size(); ++a) {
                    Vec<float,3> pa = clusters[i][a];
                    // this is a bit stupid as the covariance matrix is
                    // symmetric, but hey, one thing at a time
                    for (int x = 0; x < 3; x++) {
                        for (int y = 0; y < 3; y++) {
                            covariance.at<float>(y, x) += (pa[y] - centers[i][y])*(pa[x] - centers[i][x]);
                        }
                    }
                }
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        covariance.at<float>(y, x) /= (clusters[i].size()-1);
                    }
                }
                // dump
                cout << "covariance for cluster " << i << " [";
                for (int x = 0; x < 3; x++) {
                    if (x) {
                        cout << ",";
                    }
                    cout << "[";
                    for (int y = 0; y < 3; y++) {
                        if (y) {
                            cout << ",";
                        }
                        cout << covariance.at<float>(y, x);
                    }
                    cout << "]";
                }
                cout << "]" << endl;
                cout << "mean for cluster " << i << " " << centers[i] << endl;
            }
            // raw data per cluster
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < clusters[i].size(); ++a) {
                    Vec<float,3> pa = clusters[i][a];
                    cout << i << " " << pa[0] << " " << pa[1] << endl;
                }
            }

        }
        cout << "average silhouette for n = " << n << " is " << prev_tsi << endl;

    /*
        // take the image and write it back with alll pixels moved to the
        // nearest center
        Mat out(image.rows, image.cols, CV_8UC3);
        for (int x = 0; x < image.cols; ++x) {
            for (int y = 0; y < image.rows; ++y) {
                Vec<float,3> pixel = image.at<Vec3b>(y, x);
                float lowest = 255+255+255;
                for (int i = 0; i < n; ++i) {
                    float distance = norm(prev_centers[i]-pixel);
                    if (distance < lowest) {
                        lowest = distance;
                        out.at<Vec3b>(Point(x,y)) = prev_centers[i]; 
                    }
                }
            }
        }
        imshow("image", out);
        waitKey(0);
      */      
    }

    // XXX select n based on best silhouette, also select the covariances

    // so now we have k-means clusters, we can use them to seed EM gaussians
    // XXX which is slightly weird, as EM isn't much more compicated than
    // k-means, why do we not do EM right away? k-means can potentialy be much
    // quicker, perhaps that's worth it? we could also run the EM with the full
    // dataset, and k-means just with a fraction...

    

	return 0;
}
