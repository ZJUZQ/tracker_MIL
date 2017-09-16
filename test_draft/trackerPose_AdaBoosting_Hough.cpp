#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "BOOSTING/trackerAdaBoosting.hpp"
#include "BOOSTING/roiSelector.hpp"

using namespace std;
const double PI = 3.1415926;

static const char* keys =
{   "{@video_name      		| | 	video name        }"
	"{@imgTemplate_name     | | 	template img name        }"
    "{@start_frame     		|0| 	Start frame       }"
    "{@bounding_frame  		|0,0,0,0| 	Initial bounding frame}"};

static void help()
{
  cout << "usage: ./tracker <video_name> <start_frame> [<bounding_frame>]\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
}

double angleTemplateMatch( const cv::Mat& imgT, const cv::Point2d& centerT, const cv::Mat& image, const cv::Point2d& center );

int main( int argc, char** argv ){
	cv::CommandLineParser parser( argc, argv, keys );

	string video_name = parser.get<string>( 0 );
	string imgT_name = parser.get<string>( 1 );
	int start_frame = parser.get<int>( 2 );

	if( video_name.empty() || imgT_name.empty() ){
		help();
		return -1;
	}

  	int coords[4]={0,0,0,0};
  	bool initBoxWasGivenInCommandLine=false;

	{
		string initBoundingBox = parser.get<string>(3);

		for( size_t npos = 0, pos = 0, ctr = 0; ctr < 4; ctr++ ){
			npos = initBoundingBox.find_first_of( ',', pos ); // size_t find_first_of (const string& str, size_t pos = 0) const;

			if( npos == string::npos && ctr < 3 ){
				printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer cordinates of opposed corners of bdd box\n");
				printf("got: %s\n",initBoundingBox.substr(pos,string::npos).c_str());
				printf("manual selection of bounding box will be employed\n");
				break;
			}

			int num = atoi( initBoundingBox.substr( pos, ( ctr == 3 ) ? ( string::npos ) : ( npos - pos ) ).c_str() );
			if(num<=0){
				printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer cordinates of opposed corners of bdd box\n");
				printf("got: %s\n",initBoundingBox.substr(pos,npos-pos).c_str());
				printf("manual selection of bounding box will be employed\n");
				break;
			}
			coords[ctr]=num;
			pos=npos+1;
		}
		if( coords[0] > 0 && coords[1] > 0 && coords[2] > 0 && coords[3] > 0 ){
		  	initBoxWasGivenInCommandLine=true;
		}
	}

	//open the capture
	VideoCapture cap;
	cap.open( video_name );
	cap.set( CAP_PROP_POS_FRAMES, start_frame );

	if( !cap.isOpened() ){
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}

	Mat frame;
	cv::namedWindow( "Tracking_AdaBoosting", 1 );

	cv::Mat image;
	cv::Rect2d boundingBox;
	bool paused = false;

  	//instantiates the specific Tracker
	cv::Ptr<BOOSTING::Tracker> tracker = BOOSTING::TrackerBoosting::create();
  	if( tracker == NULL ){
    	cout << "***Error in the instantiation of the tracker...***\n";
    	return -1;
  	}

  	//get the first frame
  	cap >> frame;
  	frame.copyTo( image );
  	if(initBoxWasGivenInCommandLine){
		boundingBox.x = coords[0];
		boundingBox.y = coords[1];
		boundingBox.width = std::abs( coords[2] - coords[0] );
		boundingBox.height = std::abs( coords[3]-coords[1]);
		printf("bounding box with vertices (%d,%d) and (%d,%d) was given in command line\n",coords[0],coords[1],coords[2],coords[3]);
		cv::rectangle( image, boundingBox, cv::Scalar( 255, 0, 0 ), 2, 1 );
  	}
  	else{
  		boundingBox = BOOSTING::selectROI("Tracking_AdaBoosting", image);
  	}
    
  	cv::imshow( "Tracking_AdaBoosting", image );

  	bool initialized = false;
  	int frameCounter = 0;

  	cv::Mat imgT;
  	imgT = cv::imread(imgT_name, IMREAD_COLOR );
  	/*
  		train a classifier which can judeg the up and down
  	*/
  	// step1: get training samples

  	// step2: extract features

  	

  	for ( ;; ){

		if( !paused ){
	  		if(initialized){
	      		cap >> frame;
	      		if(frame.empty())
	        		break;
	      		
	      		frame.copyTo( image );
	  		}

	  		if( !initialized ){
	    		//initializes the tracker
	    		if( !tracker->init( frame, boundingBox ) ){
	      			cout << "***Could not initialize tracker...***\n";
	      			return -1;
	    		}
	    		initialized = true;
	  		}
	  		else if( initialized ){
	    		//updates the tracker
	    		if( tracker->update( frame, boundingBox ) ){
	      			cv::rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );

	      			cv::Mat imgFlip;
	      			cv::flip( frame( boundingBox ), imgFlip, -1); // flipping around both axes
	      			imgFlip.copyTo( frame( boundingBox ) );

	      			cv::Rect2d bb;
	      			tracker->estimateOnly( frame, bb );
	      			cv::rectangle( image, bb, Scalar( 255, 0, 0 ), 2, 1 );

	      			cv::Point2d center; // the center of the circularly symmetrical part of the track object
	      			center.x = boundingBox.x + ( boundingBox.x + boundingBox.width - bb.x ) / 2;
	      			center.y = boundingBox.y + ( boundingBox.y + boundingBox.height - bb.y ) / 2;
	      			cv::circle( image, center, 3, cv::Scalar(0, 255, 0), 4, 1 );

	      			/*
	      			double theta = angleTemplateMatch( imgT, cv::Point2d( 39.5, 45 ), frame, center );
	      			Eigen::Vector2d endP = Eigen::Rotation2Dd(theta) * Eigen::Vector2d( 0, -30 );
	      			cv::line( image, center, cv::Point2d( center.x + endP[0], center.y + endP[1] ), cv::Scalar(255, 0, 0), 2, 1 );
	    			*/
	    		}
	  		}

	  		cv::imshow( "Tracking_AdaBoosting", image );
	  		frameCounter++;
		}

		char c = (char) cv::waitKey( 2 );
		if( c == 'q' )
	  		break;
		if( c == 'p' )
	  		paused = !paused;
	}

  	return 0;
}

double angleTemplateMatch( const cv::Mat& imgT, const cv::Point2d& centerT, const cv::Mat& image, const cv::Point2d& center ){

	int angleStep = 3; // degree
	double d2r = PI / 180;

	double score = 0;
	double sqSumT = 0, sqSumI = 0;

	std::vector<double> scores;

	double scoreBest = -1;
	double radianBest = 0;

	cv::Mat grayT;
	cv::cvtColor(imgT, grayT, COLOR_BGR2GRAY);

	for( int i = 0; i < 360 / angleStep; i++ ){
		score = sqSumT = sqSumI = 0;

		for( int c = 1; c < grayT.cols - 1; c++ )
			for( int r = 1; r < grayT.rows - 1; r++ ){
				if( std::abs( grayT.at<uchar>( c + 1, r ) - grayT.at<uchar>( c - 1, r ) ) + std::abs( grayT.at<uchar>( c, r + 1 ) - grayT.at<uchar>( c, r - 1 ) ) < 10 )
					continue;

				Eigen::Vector2d xy = Eigen::Rotation2Dd(angleStep * i * d2r) * Eigen::Vector2d( c - centerT.x, r - centerT.y ) + 
										Eigen::Vector2d( center.x, center.y );
				cv::Point2i xyI;
				xyI.x = cvRound( xy[0] );
				xyI.y = cvRound( xy[1] );
				Vec3b intensity = image.at<Vec3b>( xyI.y, xyI.x );
				int intensityI = ( intensity.val[0] + intensity.val[1] + intensity.val[2] ) / 3;

				score = intensityI * grayT.at<uchar>( c, r );
				sqSumT += std::pow( grayT.at<uchar>( c, r ), 2 );
				sqSumI += std::pow( intensityI, 2 );
			}

		score = score / std::pow( sqSumT, 0.5 ) / std::pow( sqSumI, 0.5 );

		if( score > scoreBest ){
			scoreBest = score;
			radianBest = angleStep * i * d2r;
		}
	}

	return radianBest;
}

/*
double direction_judeg(cv::Mat& img_template_neg, cv::Mat& img_template_pos, cv::Mat& img_, cv::Rect2d& bb){
	// 使用TM_CCORR_NORMED和二分法判断机器人方向角
	double L = -PI;
	double H = PI;
	double mid;

	double score_neg, score_pos, sq_sum_img, sq_sum_neg, sq_sum_pos;
	int channels = img_template_neg.channels();

	for(int num = 0; num < 8; num++){
		mid = (L + H) / 2;
		score_pos = 0;
		score_neg = 0;
		sq_sum_neg = sq_sum_pos = sq_sum_img = 0;

		for(int c = -img_template_neg.cols/2; c < img_template_neg.cols/2 - 1; c++)
			for(int r = -img_template_neg.rows/2; r < img_template_neg.rows/2 -1; r++){
				// img_template, robot point up
				Eigen::Vector2d xy = Eigen::Rotation2Dd(mid) * Eigen::Vector2d(c, r)
									 + Eigen::Vector2d(bb.x + bb.width / 2, bb.y + bb.height / 2);

				// TO DO: bilinear interpolation

				for(int i = 0; i < channels; i++){
					score_neg += img_template_neg.ptr<uchar>( int(r + img_template_neg.rows/2) )[ int(c + img_template_neg.cols/2) * channels + i ]
								 * img_.ptr<uchar>( int(xy[1]) )[ int(xy[0]) * channels + i ];
					score_pos += img_template_pos.ptr<uchar>( int(r + img_template_pos.rows/2) )[ int(c + img_template_pos.cols/2) * channels + i ]
					             * img_.ptr<uchar>( int(xy[1]) )[ int(xy[0]) * channels + i ];

					sq_sum_neg += pow(img_template_neg.ptr<uchar>( int(r + img_template_neg.rows/2) )[ int(c + img_template_neg.cols/2) * channels + i ], 2);
					sq_sum_pos += pow(img_template_pos.ptr<uchar>( int(r + img_template_pos.rows/2) )[ int(c + img_template_pos.cols/2) * channels + i ], 2);
					sq_sum_img += pow(img_.ptr<uchar>( int(xy[1]) )[ int(xy[0]) * channels + i ], 2);
				}

			}
		score_neg = score_neg / pow(sq_sum_neg, 0.5) / pow(sq_sum_img, 0.5);
		score_pos = score_pos / pow(sq_sum_pos, 0.5) / pow(sq_sum_img, 0.5);

		if(score_neg > score_pos)
			H = mid;
		else
			L = mid;
	}
	return (L + H) / 2;
}
*/
