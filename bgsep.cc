#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat matFromVec(Vec<float,3> v) {
    Mat ret(3, 1, CV_32F);
    ret.at<float>(0, 0) = v[0];
    ret.at<float>(1, 0) = v[1];
    ret.at<float>(2, 0) = v[2];
    return ret;
}

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
    vector<Vec<float,3>> centers;
    vector<Mat> covariances;
    for (int n = 3; n < 4; ++n) {
        cout << "k-means clustering for " << n << " clusters..." << endl;
        float prev_tsi = -1;
        vector<Vec<float,3>> prev_centers(n, {0, 0, 0});
        for (int reps = 0; reps < 1; reps++) {
            // draw n random ones as cluster centers
            centers = vector<Vec<float,3>>(n, {0, 0, 0});
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
            cout << "determining the silhouette..." << endl;
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
            covariances = vector<Mat>(n);
            for (int i = 0; i < n; ++i) {
                Mat covariance(3, 3, CV_32F, 0.0);
                for (int a = 0; a < clusters[i].size(); ++a) {
                    Vec<float,3> pa = clusters[i][a];
                    // XXX this is a bit stupid as the covariance matrix is
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
                covariances[i] = covariance;
            }
            /*
            // raw data per cluster
            for (int i = 0; i < n; ++i) {
                for (int a = 0; a < clusters[i].size(); ++a) {
                    Vec<float,3> pa = clusters[i][a];
                    cout << i << " " << pa[0] << " " << pa[1] << endl;
                }
            }
            */

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
    cout << endl;

    // XXX select n based on best silhouette, also select the covariances

    // so now we have k-means clusters, we can use them to seed EM gaussians
    // k-means is a special case of EM, so this looks odd. but with EM it is
    // hard to get a full covariance matrix with few samples (too many factors)
    // so seeding is great
    int n = covariances.size();
    vector<double> cluster_priors(3, 1.0);
    cout << "EM algorithm with " << n << " clusters" << endl;
    bool converged = false;
    do {
        cout << "-------" << endl;
        for (int i = 0; i < n; ++i) {
            cout << "mean for cluster " << i << " " << centers[i] << endl;
            cout << "covariance for cluster " << i << " [";
            for (int x = 0; x < 3; x++) {
                if (x) {
                    cout << ", ";
                }
                cout << "[";
                for (int y = 0; y < 3; y++) {
                    if (y) {
                        cout << ", ";
                    }
                    cout << covariances[i].at<float>(y, x);
                }
                cout << "]";
            }
            cout << "]" << endl;
            cout << "prior prob for cluster " << i << " is " << cluster_priors[i] << endl;
        }

        // determine the relative probabilities for each pixel and cluster
        double weights[bgpixels.size()][n];
        for (int p = 0; p < bgpixels.size(); ++p) {
            Vec<float,3> pixel = bgpixels[p];
            
            double weightsum = 0.0;
            for (int c = 0; c < n; ++c) {
                Vec<float,3> center = centers[c];
                Mat covariance = covariances[c];
                Mat a =   (matFromVec(pixel) - matFromVec(center)).t() 
                                        * covariance.inv() 
                                        * (matFromVec(pixel) - matFromVec(center));
                double aa = a.at<float>(0,0);
                double af = exp(-.5 * aa);
                double bf =  sqrt( pow(2 * M_PI, 3.0) * determinant(covariance) );
                weights[p][c] = af / bf * cluster_priors[c];
                weightsum += weights[p][c];
            }
            for (int c = 0; c < n; ++c) {
                weights[p][c] /= weightsum;
            }
        }

        // compute the new means
        vector<Vec<float,3>> new_centers(n);
        for (int c = 0; c < n; ++c) {
            Vec<float,3> new_center(0.0, 0.0, 0.0);
            double weightsum = 0.0;
            for (int p = 0; p < bgpixels.size(); ++p) {
                Vec<float,3> pixel = bgpixels[p];
                pixel *= weights[p][c];
                new_center += pixel;
                weightsum += weights[p][c];
            }
            new_center /= weightsum;
            new_centers[c] = new_center;
            cluster_priors[c] = weightsum / bgpixels.size();
        }

        
        // and the new covariance
        vector<Mat> new_covariances = vector<Mat>(n);
        for (int c = 0; c < n; ++c) {
            Mat covariance(3, 3, CV_32F, 0.0);
            double weightsum = 0.0;
            for (int p = 0; p < bgpixels.size(); ++p) {
                Vec<float,3> pixel = bgpixels[p];
                // XXX this is a bit stupid as the covariance matrix is
                // symmetric, but hey, one thing at a time
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        covariance.at<float>(y, x) += weights[p][c] * (pixel[y] - new_centers[c][y])*(pixel[x] - new_centers[c][x]);
                    }
                }
                weightsum += weights[p][c];
            }
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    covariance.at<float>(y, x) /= weightsum;
                }
            }
            new_covariances[c] = covariance;
        }
        // determine how much the centers have moved since the last iteration,
        // if this is really small, then we have convergence
        double drift = 0.0;
        for (int c = 0; c < n; ++c) {
            drift += norm(centers[c]-new_centers[c]);
        }
        drift /= n;
        cout << "drift " << drift << endl;
        if (drift < 0.1) { // totally arbitrary value, but conservative
            converged = true;
        }

        covariances = new_covariances;
        centers = new_centers;

    } while (!converged);
    

    // now read the actual image and for each pixel determine whether it is more
    // likely to be forground or background

    image = imread("coos.png", CV_LOAD_IMAGE_COLOR);
	if (!image.data) {
		cerr << "could not load image" << endl;
		return 1;
	}	
	cout << "Read image " << image.cols << "x" << image.rows << " pixels" << endl;
    Vec3b white{255,255,255};
    double max_bg_prop = 4e-5;
    for (int x = 0; x < image.cols; ++x) {
        for (int y = 0; y < image.rows; ++y) {
            Vec3b color = image.at<Vec3b>(y, x);
            double bg_prop = 0.0;
            for (int c = 0; c < n; ++c) {
                Vec<float,3> center = centers[c];
                Mat covariance = covariances[c];
                Mat a =   (matFromVec(color) - matFromVec(center)).t() 
                                        * covariance.inv() 
                                        * (matFromVec(color) - matFromVec(center));
                double aa = a.at<float>(0,0);
                double af = exp(-.5 * aa);
                double bf =  sqrt( pow(2 * M_PI, 3.0) * determinant(covariance) );
                bg_prop += af / bf * cluster_priors[c];
            }
            // XXX this should b 1 if summed over all colors? is it? 
            double fg_prop = 1.0 / 255.0 / 255.0 / 255.0;
            if (fg_prop < bg_prop) {
                image.at<Vec3b>(y, x) = white;
            }
/*
            int rel_prob = (int)(512.0 * bg_prop / fg_prop);
            if (rel_prob > 255.0) {
                rel_prob = 255.0;
            }
            cout << "rel prob " << rel_prob << endl;
            Vec3b false_col{rel_prob, 255 - rel_prob, 255 - rel_prob};
            image.at<Vec3b>(y, x) = false_col;*/
        }
    }
 //  cout << "maximum background probability is " << max_bg_prop << endl;
    // save to chck it out
    imwrite("coos-gmm.png", image);


    /*
    // as an experiment, let's sum the probabilities over all possibel pixel
    // values and see if that sums up to 1, independently from the number of
    // clusters...
    double sum = 0.0;
    for (int x = 0; x < 256; ++x) {
        for (int y = 0; y < 256; ++y) {
            for (int z = 0; z < 256; ++z) {
                Vec3b color{x,y,z};
                double bg_prop = 0.0;
                for (int c = 0; c < n; ++c) {
                    Vec<float,3> center = centers[c];
                    Mat covariance = covariances[c];
                    Mat a =   (matFromVec(color) - matFromVec(center)).t() 
                                            * covariance.inv() 
                                            * (matFromVec(color) - matFromVec(center));
                    double aa = a.at<float>(0,0);
                    double af = exp(-.5 * aa);
                    double bf =  sqrt( pow(2 * M_PI, 3.0) * determinant(covariance) );
                    bg_prop += af / bf;
                }
                sum += bg_prop;
            }
        }
    }
    cout << "sum over all color values is " << sum << " with " << n << " gaussians" << endl;
    // XXX -> so it looks like this is not normalised, which does not make much
    // sense. we should noramlise it, but could that not lead to instabilites
    // during converging it? perhaps we should not take this factor into account
    // when converging, or perhaps not initially?
*/
	return 0;
}
