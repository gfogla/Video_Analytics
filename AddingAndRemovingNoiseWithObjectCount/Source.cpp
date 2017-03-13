#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2\imgproc\imgproc.hpp>
#include <windows.h>
#include <tchar.h>
#include <iomanip>
#include <math.h>

using namespace cv;
using namespace std;

int a[9];
Mat ga,ra;
RNG rng(12345);

void imgResize(Mat &image, int p) 
{
	Size size(image.cols + p - 1, image.rows + p - 1);
	Mat temp = image.clone();
	resize(image, image, size);

	for (int y = (p / 2), m = 0; m < temp.rows; y++, m++) {
		for (int x = (p / 2), n = 0; n < temp.cols; x++, n++) {
			image.at<Vec3b>(y, x) = temp.at<Vec3b>(m, n);
		}
	}

	for (int i = (p / 2); i < image.rows - (p / 2); i++)
	{
		for (int j = 0; j < (p / 2); j++) 
		{
			image.at<Vec3b>(i, j) = temp.at<Vec3b>(i - (p / 2), 0);
			image.at<Vec3b>(i, (image.cols - 1 - j)) = temp.at<Vec3b>(i - (p / 2), temp.cols - 1);
		}
	}

	for (int i = 0; i < image.cols; i++)
	{
		for (int j = 0; j < (p / 2); j++) 
		{
			image.at<Vec3b>(j, i) = image.at<Vec3b>((p / 2), i);
			image.at<Vec3b>((image.rows - 1) - j, i) = image.at<Vec3b>((image.rows - (p / 2)), i);
		}
	}
}

void putBlack(Mat img, int y, int x)
{
	img.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(0);
	img.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(0);
	img.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(0);
}

void putWhite(Mat img, int y, int x)
{
	img.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(255);
	img.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(255);
	img.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(255);
}

void AddSaltPepperNoise(Mat &imgSP, float p)

{
	RNG rng;
	int amount = imgSP.rows*imgSP.cols*p;

	for (int counter = 0; counter<amount; ++counter){
		putBlack(imgSP, rng.uniform(0, imgSP.rows), rng.uniform(0, imgSP.cols));
		putWhite(imgSP, rng.uniform(0, imgSP.rows), rng.uniform(0, imgSP.cols));
	}		
}

void AddGaussianNoise(const Mat image, Mat &imgGauss, double Sigma)

{
	Mat img_16SC;
	Sigma *= 255;
	Mat g_noise = Mat(image.size(), CV_16SC3);
	randn(g_noise, Scalar::all(0), Scalar::all(Sigma));
	
	image.convertTo(img_16SC, CV_16SC3);
	addWeighted(img_16SC, 1.0, g_noise, 1.0, 0.0, img_16SC);
	img_16SC.convertTo(imgGauss, image.type());

}

void insertionSort()
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = a[i];
		for (j = i - 1; j >= 0 && temp < a[j]; j--) {
			a[j + 1] = a[j];
		}
		a[j + 1] = temp;
	}
}

void RemoveSaltPepperNoise(Mat &imgSP,Mat &imgRmSP) 
{

	imgRmSP = imgSP.clone();
	
	imgResize(imgSP,3);
	
	for (int y = 1, m = 0; y < imgSP.rows - 2; y++, m++) {
		for (int x = 1, n = 0; x < imgSP.cols - 2; x++, n++) {
			for (int k = 0; k < 3; k++)
			{
				// Pick up window element

				a[0] = imgSP.at<Vec3b>(y - 1, x - 1)[k];
				a[1] = imgSP.at<Vec3b>(y, x - 1)[k];
				a[2] = imgSP.at<Vec3b>(y + 1, x - 1)[k];
				a[3] = imgSP.at<Vec3b>(y - 1, x)[k];
				a[4] = imgSP.at<Vec3b>(y, x)[k];
				a[5] = imgSP.at<Vec3b>(y + 1, x)[k];
				a[6] = imgSP.at<Vec3b>(y - 1, x + 1)[k];
				a[7] = imgSP.at<Vec3b>(y, x + 1)[k];
				a[8] = imgSP.at<Vec3b>(y + 1, x + 1)[k];

				// sort the window to find median
				insertionSort();

				// assign the median to centered element of the matrix
				
				imgRmSP.at<Vec3b>(m, n)[k] = a[4];
			}
		}
	}
}

void RemoveGaussianNoise(Mat &imgG, Mat &imgRmGauss)
{

	double sigma = 1;
	int W = 5;
	double kernel[5][5];
	double mean = W / 2;
	double sum = 0.0; // For accumulating the kernel values
	for (int x = 0; x < W; ++x)
		for (int y = 0; y < W; ++y) {
			kernel[x][y] = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
				/ (2 * M_PI * sigma * sigma);

			// Accumulate the kernel values
			sum += kernel[x][y];
		}

	// normalize the Kernel
	for (int i = 0; i < W; ++i)
	{	
		for (int j = 0; j < W; ++j)
			kernel[i][j] /= sum;
	}
			

	imgResize(imgG,W);

	for (int y = 2, m = 0; y < imgG.rows - 4; y++, m++) {
		for (int x = 2, n = 0; x < imgG.cols - 4; x++, n++) {
			for (int k = 0; k < 3; k++)
			{	
				int sum = 0;
				for (int i = -(W / 2); i <= (W / 2); ++i) 
				{
					for (int j = -(W / 2); j <= (W / 2); ++j) 
					{
						sum = sum + imgG.at<Vec3b>((y + i),(x + j))[k] * kernel[(W / 2) + i][(W / 2) + j];
					}
				}
				imgRmGauss.at<Vec3b>(m, n)[k] = sum;
			}
		}
	}
}

cv::Mat imageBrighten(Mat image)
{
	// Increasing the brightness by 50.

	Mat imgBright = Mat::zeros(image.size(), image.type());
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				imgBright.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((image.at<Vec3b>(y, x)[c]) + 50);
			}
		}
	}

	return imgBright;
}

int cal_apples(Mat image, int np1, int np2)
{
	cvtColor(image, ga, CV_BGR2GRAY);
	threshold(ga, ga, 130, 255, CV_THRESH_BINARY);
	Mat element;
	
	switch (np1)
	{
		case 1:
			element = getStructuringElement(MORPH_ELLIPSE, Size(47, 47));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
			dilate(ga, ga, element);
			
			ga.convertTo(ga, CV_8UC1, 255, 0);
			
			break;

		case 2:
			element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
			dilate(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(50, 50));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
			dilate(ga, ga, element);
			
			ga.convertTo(ga, CV_8UC1, 255, 0);
			
			break;

		case 3:
			element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
			dilate(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(46, 46));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(17, 17));
			dilate(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
			dilate(ga, ga, element);
			
			break;

		case 4:
			element = getStructuringElement(MORPH_ELLIPSE, Size(47, 47));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
			dilate(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
			dilate(ga, ga, element);
			
			ga.convertTo(ga, CV_8UC1, 255, 0);
			
			break;

		case 5:
			element = getStructuringElement(MORPH_ELLIPSE, Size(47, 47));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(40, 40));
			erode(ga, ga, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
			dilate(ga, ga, element);
			
		default :
			break;
	}
	
	
	Mat imgCount;
	int countGreen = connectedComponents(ga, imgCount, 8, CV_16U) - 1;
	cout << "\nNumber of green apples : " << countGreen;
	cvtColor(image, ra, CV_BGR2HSV);
	
	inRange(ra, Scalar(0,0,0), Scalar(12,255,255), ra);
	
	switch (np2)
	{
		case 1:
			element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
			dilate(ra, ra, element);
			
			break;

		case 2:
			element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
			dilate(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(35, 35));
			dilate(ra, ra, element);
			
			break;

		case 3:
			element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
			dilate(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
			dilate(ra, ra, element);
			
			break;

		case 4:
			element = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
			erode(ra, ra, element);
			
			element = getStructuringElement(MORPH_ELLIPSE, Size(35, 35));
			dilate(ra, ra, element);
			
		default :	
			break;
	}

	
	int countRed = connectedComponents(ra, imgCount, 8, CV_16U) - 1;
	cout << "\nNumber of Red apples : " << countRed;
	return countGreen + countRed;
}

int main(int argc, char** argv)
{

	Mat image = imread(argv[1], IMREAD_COLOR);						// Read the file

	if (!image.data)                                                // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	namedWindow("Original Image", WINDOW_AUTOSIZE);					// Create a window for display.
	imshow("Original Image", image);

	//Adding salt n pepper noise
	Mat imgSP = image.clone();
	Mat dst;
	AddSaltPepperNoise(imgSP, 0.02);
	imshow("Image with Salt n Pepper Noise.", imgSP);

	//Adding gaussian noise
	Mat imgGauss(image.size(),image.type());
	AddGaussianNoise(image, imgGauss,0.15);
	imshow("Image with Gaussian Noise.", imgGauss);

	//Removing Salt n Pepper Noise
	Mat imgRmSP;
	RemoveSaltPepperNoise(imgSP, imgRmSP);
	imshow("Removal of SP Noise.", imgRmSP);

	//Removing Gaussian Noise
	Mat imgRmGauss(image.size(), image.type());
	RemoveGaussianNoise(imgGauss, imgRmGauss);
	imshow("Removal of Gaussian Noise.", imgRmGauss);

	Mat imgBright = imageBrighten(image);
	imshow("Brightened Original Image", imgBright);

	int Count;
	Count = cal_apples(image, 1, 1);
	cout << "\nTotal number of apples(image) are : " << Count;
	
	Count = cal_apples(imgSP, 2, 1);
	cout << "\nTotal number of apples(With Salt-pepper) are : " << Count;
	
	Count = cal_apples(imgRmSP, 1, 1);
	cout << "\nTotal number of apples(Without Salt-pepper) are : " << Count;

	Count = cal_apples(imgGauss, 3, 2);
	cout << "\nTotal number of apples(With Gaussian) are : " << Count;

	Count = cal_apples(imgRmGauss, 4, 3);
	cout << "\nTotal number of apples(Without Gaussian) are : " << Count;

	Count = cal_apples(imgBright, 5, 4);
	cout << "\nTotal number of apples(Brightened Image) are : " << Count;

	CreateDirectory(_T("OutputImages"), 0);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);
	cv::imwrite("OutputImages/OriginalInputImage.jpg", image, compression_params);
	cv::imwrite("OutputImages/BrightenedImage.jpg", imgBright, compression_params);
	cv::imwrite("OutputImages/ImageWithSaltPepper.jpg", imgSP, compression_params);
	cv::imwrite("OutputImages/ImageAfterSPRemoval.jpg", imgRmSP, compression_params);
	cv::imwrite("OutputImages/ImageWithGaussian.jpg", imgGauss, compression_params);
	cv::imwrite("OutputImages/ImageAfterGausianRemoval.jpg", imgRmGauss, compression_params);

	waitKey(0);														// Wait for a keystroke in the window
	return 0;
}