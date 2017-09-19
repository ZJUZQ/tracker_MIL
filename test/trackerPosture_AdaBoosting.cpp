#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdio.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "BOOSTING/trackerAdaBoosting.hpp"
#include "BOOSTING/roiSelector.hpp"

#include "BOOSTING/trackerAdaBoostingClassifier.hpp"
#include "BOOSTING/trackerFeature.hpp"


using namespace std;

const double PI = 3.1415926;

static const char* keys =
{   "{@video_name      		| | 	video name        }"
	"{@imgTemplate_name     | | 	template img name        }"
    "{@start_frame     		|0| 	Start frame       }"
    "{@bounding_frame  		|0,0,0,0| 	Initial bounding frame}"};

static void help()
{
  cout << "usage: ./tracker <video_name> <imgTemplate_name> <start_frame> [<bounding_frame>]\n"
       << endl;

  cout << "\nHot keys: \n"
       	  "\tq - quit the program\n"
          "\tp - pause video\n";
}

/* method 1: using Template matching to compute robot's direction angle */
double angleTemplateMatch( const cv::Mat& imgT, const cv::Point2d& centerT, const cv::Mat& image, const cv::Point2d& center );

int main( int argc, char** argv )
{
	cv::CommandLineParser parser( argc, argv, keys );

	string video_name = parser.get<string>( 0 );
	string imgT_name = parser.get<string>( 1 );
	int start_frame = parser.get<int>( 2 );

	if( video_name.empty() || imgT_name.empty() ){
		help();
		return -1;
	}

  	int coords[4]={0,0,0,0};
  	/* extract boundingbox given in command line */
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

	cv::Mat frame;
	cv::namedWindow( "Tracking_AdaBoosting", 1 );

	cv::Mat image;
	cv::Rect2d boundingBox;
	bool paused = false;

  	//instantiates the AdaBoosting Tracker
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

  	cv::Mat imgT; // the input template image
  	imgT = cv::imread(imgT_name, cv::IMREAD_COLOR );

  	
/*********************************** train up-down classifier **********************************/
/*

*/
    // parameters
    int d_numbaseclfs = 100;
    int d_numweakclfs = d_numbaseclfs * 10;
    bool d_useFeatureExchange = true; 
    int d_iterationInit = 5;
    cv::Size d_patchSize = cv::Size( 46, 46 ); // TODO: actual size of training sample
 		
 	cout << "debug_1\n";
    // strong classifier
    cv::Ptr<BOOSTING::StrongClassifierDirectSelection> d_strongClassifier = cv::Ptr<BOOSTING::StrongClassifierDirectSelection>( 
            new BOOSTING::StrongClassifierDirectSelection( d_numbaseclfs, d_numweakclfs, d_patchSize, cv::Rect( 0, 0, imgT.cols, imgT.rows ), d_useFeatureExchange, d_iterationInit ) );
        // init base classifiers
    d_strongClassifier->initBaseClassifiers();
 
 	cout << "debug_2\n";
    // HAAR feature
    BOOSTING::TrackerFeatureHAAR::Params d_HAARparameters;
    d_HAARparameters.numFeatures = d_numbaseclfs * 10  + d_iterationInit;
    d_HAARparameters.isIntegral = true;
    d_HAARparameters.rectSize = cv::Size( static_cast<int>( d_patchSize.width ), static_cast<int>( d_patchSize.height ) );
    cv::Ptr<BOOSTING::TrackerFeature> d_trackerFeature = cv::Ptr<BOOSTING::TrackerFeatureHAAR>( new BOOSTING::TrackerFeatureHAAR( d_HAARparameters ) );
     
    // training the strong classifier
    int numTotalSamples = 900;
    cv::Point2f centerT( 39.5, 37.5 ); // TODO: change with imgT
    cv::Rect2d boundingBoxT = cv::Rect2d( centerT.x - d_patchSize.width / 2.0, centerT.y - d_patchSize.height / 2.0,
    									  d_patchSize.width, d_patchSize.height ); // TODO: change with imgT
 
    for( int i = 0; i < numTotalSamples; i++ ){
 
        // 将随机生成新的TrackerFeatureHAAR，用来替换每次用一个样本训练时挑出的最差feature
        BOOSTING::TrackerFeatureHAAR::Params HAARparameters2;
        HAARparameters2.numFeatures = 1;
        HAARparameters2.isIntegral = true;
        HAARparameters2.rectSize = cv::Size( (int)d_patchSize.width, (int)d_patchSize.height );
        cv::Ptr<BOOSTING::TrackerFeatureHAAR> trackerFeature2 = cv::Ptr<BOOSTING::TrackerFeatureHAAR>( new BOOSTING::TrackerFeatureHAAR( HAARparameters2 ) );
 
        std::vector<cv::Mat> d_samples;
 
        int d_angle = std::rand() % 360; // rotation angle in degrees
        if( abs( d_angle - 90 ) <= 2 || abs( d_angle - 270 ) <= 2 )
            continue;
 
        cv::Mat d_image;
        cv::Mat rotateMat = cv::getRotationMatrix2D( centerT, d_angle, 1 ); // rotation cunter_clockwise
        cv::warpAffine( imgT, d_image, rotateMat, imgT.size() );
 
        cv::Mat_<int> intImage;
        cv::Mat_<double> intSqImage;
        cv::Mat d_image_;
        cv::cvtColor( d_image, d_image_, CV_RGB2GRAY );
        cv::integral( d_image_, intImage, intSqImage, CV_32S );
 
 		//d_samples.push_back( intImage( boundingBoxT ) );
 		d_samples.push_back( intImage( cv::Rect( boundingBoxT.x + std::rand() % 7 - 3,
 												 boundingBoxT.y + std::rand() % 7 - 3, boundingBoxT.width, boundingBoxT.height ) ) );


        int currentUp = -1; // robot turns up, angle [-180, 0]

        if( d_angle > 90 && d_angle < 270 ){
            currentUp = 1; 	// robot turns down, angle [0, 180]
        }
       
 
        cv::Mat response;
        d_trackerFeature->compute( d_samples, response );
 
        d_strongClassifier->update( response.col(0), currentUp ); // for each training sample, update all weakclassifiers and strongClassifier, Algorithm 2.1
 
        int replacedWeakClassifier, swappedWeakClassifier;
        if( d_useFeatureExchange ){
            replacedWeakClassifier = d_strongClassifier->getReplacedClassifier(); // each traing sample will produce one bad weakclassifier to be replaced
            swappedWeakClassifier = d_strongClassifier->getSwappedClassifier();
            if( replacedWeakClassifier >= 0 && swappedWeakClassifier >= 0 )
                d_strongClassifier->replaceWeakClassifier( replacedWeakClassifier );
            }
        else{
            replacedWeakClassifier = -1;
            swappedWeakClassifier = -1;
        }
 
        /*  因为weakclassifier是基于TrackerFeatureHAAR的，因此，在交换了weakClassifierHaarFeature(没有实际实现Haar feature)后，
            还需要实际交换实际的TrackerFeatureHAAR，两者之间索引是一致的。 
        */
        if( replacedWeakClassifier != -1 && swappedWeakClassifier != -1 ){
            d_trackerFeature.staticCast<BOOSTING::TrackerFeatureHAAR>()->swapFeature( replacedWeakClassifier, swappedWeakClassifier );
            d_trackerFeature.staticCast<BOOSTING::TrackerFeatureHAAR>()->swapFeature( swappedWeakClassifier, trackerFeature2->getFeatureAt( 0 ) );
        }   
    }

    /******************************* test up-down classifier ****************************/
    /*
    int numTestSamples = 600;

    for( int i = 0; i < numTestSamples; i++ )
    {
    	int d_angle = std::rand() % 360; // rotation angle in degrees

    	int currentUp = -1; // robot turns up, angle [-180, 0]
        if( d_angle > 90 && d_angle < 270 ){
            currentUp = 1; 	// robot turns down, angle [0, 180]
        }
     
     	if( abs( d_angle - 90 ) <= 2 || abs( d_angle - 270 ) <= 2 )
            continue; 

        cv::Mat d_image;
        cv::Mat rotateMat = cv::getRotationMatrix2D( centerT, d_angle, 1 );
        cv::warpAffine( imgT, d_image, rotateMat, imgT.size() );
 
        cv::Mat_<int> intImage;
        cv::Mat_<double> intSqImage;
        cv::Mat d_image_;
        cv::cvtColor( d_image, d_image_, CV_RGB2GRAY );
        cv::integral( d_image_, intImage, intSqImage, CV_32S );
 
        std::vector<cv::Mat> d_samples;
 		//d_samples.push_back( intImage( boundingBoxT ) );
 		d_samples.push_back( intImage( cv::Rect( boundingBoxT.x + std::rand() % 7 - 3,
 												 boundingBoxT.y + std::rand() % 7 - 3, boundingBoxT.width, boundingBoxT.height ) ) );

 		cv::Mat response;
        d_trackerFeature->compute( d_samples, response );
        float conf = d_strongClassifier->eval( response.col( 0 ) );

        if( conf * currentUp > 0 ) // robot turns down
            cout << "predict true: conf = " << conf << ", d_angle = " << d_angle << endl;
        else
            cout << "predict wrong, d_angle = " << d_angle << endl;

        char charTestImage[50];
        sprintf( charTestImage, "./test_imgs/img%d_angle%d_conf%d.png", i, d_angle, (int)(conf * 10) );
        if( i % 30 == 0 )
        	cv::imwrite( cv::String( charTestImage ), d_image );
    }
    */


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
	    			cv::Mat frame_temp;
	    			frame.copyTo( frame_temp );

	      			cv::rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );

	      			cv::Mat imgFlip;
	      			cv::flip( frame_temp( boundingBox ), imgFlip, -1); // flipping around both axes
	      			imgFlip.copyTo( frame_temp( boundingBox ) );

	      			cv::Rect2d bb;
	      			tracker->estimateOnly( frame_temp, bb );
	      			cv::rectangle( image, bb, Scalar( 255, 0, 0 ), 2, 1 );

	      			cv::Point2d center; // the center of the circularly symmetrical part of the track object
	      			center.x = boundingBox.x + ( boundingBox.x + boundingBox.width - bb.x ) / 2;
	      			center.y = boundingBox.y + ( boundingBox.y + boundingBox.height - bb.y ) / 2;
	      			cv::circle( image, center, 3, cv::Scalar(0, 255, 0), 4, 1 );

	      			/* method 1: TM
                    double theta = angleTemplateMatch( imgT, cv::Point2d( 39.5, 45 ), frame, center );
                    Eigen::Vector2d endP = Eigen::Rotation2Dd(theta) * Eigen::Vector2d( 0, -30 );
                    cv::line( image, center, cv::Point2d( center.x + endP[0], center.y + endP[1] ), cv::Scalar(255, 0, 0), 2, 1 );
                    */
 
                    /* method 2: boost */

                    float uplimitAngle = 180;
                    float downlimitAngle = -180;
                    cv::Mat img_part = frame( cv::Rect( center.x - d_patchSize.width, center.y - d_patchSize.height, d_patchSize.width * 2, d_patchSize.height * 2 ) );
                    
                    char charOutImage[40];
                    static int imgCount = 0;

                    for( int i = 0; i < 6; i++ )
                    {
                    	cv::Point center_part( d_patchSize.width, d_patchSize.height );
                        float midAngle = ( uplimitAngle + downlimitAngle ) / 2;
                        cv::Mat rotateMat = cv::getRotationMatrix2D( center_part, midAngle, 1 );
                        cv::Mat imgA;
                        cv::warpAffine( img_part, imgA, rotateMat, cv::Size( img_part.cols, img_part.rows ) );
                        cv::Mat imgA_gray;            
                        cv::Mat_<int> intImageA;
				        cv::Mat_<double> intSqImageA;
				        cv::cvtColor( imgA, imgA_gray, CV_RGB2GRAY );		        
				        cv::integral( imgA_gray, intImageA, intSqImageA, CV_32S );

				        //float conf_best = 0;
				        float conf_pos = 0;
				        float conf_neg = 0;
				        float conf;

				        char outNameChar[50];

				        static int outImageNum = 0;

				        cv::Mat sampleA = intImageA( cv::Rect2d( center_part.x - d_patchSize.width / 2, center_part.y - d_patchSize.height / 2, d_patchSize.width, d_patchSize.height ) );
                        std::vector<cv::Mat> samplesA;
                        samplesA.push_back( sampleA );

                        cv::Mat response;
                        d_trackerFeature->compute( samplesA, response );
                        conf = d_strongClassifier->eval( response.col( 0 ) );

                        if( conf > 0 ){
                        	downlimitAngle = midAngle;
                        	sprintf( outNameChar, "./run_imgs/image%d_conf_pos.png", outImageNum );
                        }
                        else{
                        	uplimitAngle = midAngle;
                            sprintf( outNameChar, "./run_imgs/image%d_conf_neg.png", outImageNum );
                        }

				        /*
				        for( int r = -2; r < 3 ; r++ )
				        	for( int c = -2; c < 3; c++ )
				        	{
				        		cv::Mat sampleA = intImageA( cv::Rect2d( center_part.x - d_patchSize.width / 2 + c, center_part.y - d_patchSize.height / 2 + r, d_patchSize.width, d_patchSize.height ) );
		                        std::vector<cv::Mat> samplesA;
		                        samplesA.push_back( sampleA );

		                        cv::Mat response;
		                        d_trackerFeature->compute( samplesA, response );
		                        conf = d_strongClassifier->eval( response.col( 0 ) );

		                        if( conf > 0 )
		                        	conf_pos += conf;
		                        else
		                        	conf_neg += conf; 
				        	}
				        
                        if( std::abs( conf_pos ) > std::abs( conf_neg ) ) // robot turns down
                        {
                        	downlimitAngle = midAngle;
                        	sprintf( outNameChar, "./images/image%d_conf_pos.png", outImageNum );
                        }   
                        else{
                            uplimitAngle = midAngle;
                            sprintf( outNameChar, "./images/image%d_conf_neg.png", outImageNum );
                        }
                        */

                        if( outImageNum % 60 == 0 && outImageNum > 100 ){
                        	/*
                        	cv::imwrite( cv::String( outNameChar ), 
                        				 imgA( cv::Rect2d( center_part.x - d_patchSize.width / 2, center_part.y - d_patchSize.height / 2, d_patchSize.width, d_patchSize.height ) ) );
							*/
                        }
                      	outImageNum++;

                        //cout << "conf_best = " << conf_best << endl;
	    			}

	    			if( imgCount % 40 == 0 ){
	    				//sprintf( charOutImage, "./OUTimgs/img%d_angle%d.png", imgCount, int( ( uplimitAngle + downlimitAngle ) / 2 ) );
	    				//cv::imwrite( cv::String( charOutImage ), img_part );
	    			}
	    			imgCount++;
	    		

	    			Eigen::Vector2d endP = Eigen::Rotation2Dd( ( uplimitAngle + downlimitAngle ) / 2.0 * PI / 180 ) * Eigen::Vector2d( 50, 0 );
                    cv::line( image, center, cv::Point2d( center.x + endP[0], center.y + endP[1] ), cv::Scalar(0, 0, 255), 2, 1 );
	    		}
	  		}

	  		cv::imshow( "Tracking_AdaBoosting", image );
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
