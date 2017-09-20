#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "BOOSTING/trackerAdaBoosting.hpp"
#include "BOOSTING/roiSelector.hpp"

#include "BOOSTING_DIRECTION/directionAdaBoosting.hpp"


using namespace std;

const double PI = 3.1415926;

static const char* keys =
{   "{@video_name      		| | 	video name        }"
	"{@imgTemplate_name     | | 	template img name        }"
    "{@start_frame     		|0| 	Start frame       }"
    "{@bounding_frame  		|0,0,0,0| 	Initial bounding frame}"
};

static void help()
{
  cout << "usage: ./tracker <video_name> <imgTemplate_name> <start_frame> [<bounding_frame>]\n"
       << endl;

  cout << "\nHot keys: \n"
       	  "\tq - quit the program\n"
          "\tp - pause video\n";
}

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

    if( !tracker->init( frame, boundingBox ) ){
        cout << "***Could not initialize tracker...***\n";
        return -1;
    }

  	cv::Mat imgT; // the input template image
  	imgT = cv::imread(imgT_name, cv::IMREAD_COLOR );

  	
/*********************************** initialize direction classifier **********************************/

    BOOSTING_DIRECTION::directionAdaBoosting::Params parameters;
    parameters.numBaseClfs = 100;
    parameters.numWeakClfs = parameters.numBaseClfs * 10;
    parameters.numAllWeakClfs = parameters.numWeakClfs + 5;
    parameters.patchSize = cv::Size( 46, 46 ); // object size which is circular symmetry
    parameters.useFeatureExchange = true;

    cv::Ptr<BOOSTING_DIRECTION::directionAdaBoosting> directionClf = new BOOSTING_DIRECTION::directionAdaBoosting( parameters );

    cv::Rect objectBoundingBox( imgT.cols / 2 - parameters.patchSize.width / 2, 
                                imgT.rows / 2 - parameters.patchSize.height / 2,
                                parameters.patchSize.width,
                                parameters.patchSize.height );

    if( !directionClf->init( imgT, objectBoundingBox ) ){     // training the direction classifier
        cout << "could not initialize the direction classifier\n";
        return -1;
    }

    /******************************* test direction classifier ****************************/
    /*
    int numTestSamples = 600;
    cv::Point2f centerT( imgT.cols / 2.0, imgT.rows / 2.0 );

    for( int i = 0; i < numTestSamples; i++ )
    {
    	int d_angle = std::rand() % 360; // rotation angle in degrees

    	int currentLabel = -1; // robot turns up, angle [-180, 0]
        if( d_angle > 90 && d_angle < 270 ){
            currentLabel = 1; 	// robot turns down, angle [0, 180]
        }
     
     	if( abs( d_angle - 90 ) <= 2 || abs( d_angle - 270 ) <= 2 )
            continue; 

        cv::Mat d_image;
        cv::Mat rotateMat = cv::getRotationMatrix2D( centerT, d_angle, 1 );
        cv::warpAffine( imgT, d_image, rotateMat, imgT.size() );

        int label = directionClf->classifierSample( d_image( cv::Rect( objectBoundingBox.x + std::rand() % 7 - 3,
                                                                       objectBoundingBox.y + std::rand() % 7 - 3, 
                                                                       objectBoundingBox.width, 
                                                                       objectBoundingBox.height ) ) );

        if( label * currentLabel > 0 ) // robot turns down
            cout << "predict true: label = " << label << ", d_angle = " << d_angle << endl;
        else
            cout << "predict wrong: label = " << label << ", d_angle = " << d_angle << endl;

        char charTestImage[50];
        sprintf( charTestImage, "./test_imgs/img%d_angle%d_label%d.png", i, d_angle, label );
        if( i % 30 == 0 )
        	cv::imwrite( cv::String( charTestImage ), d_image );
    }
    */

  	for ( ; ; ){
		if( !paused ){ 	
      		cap >> frame;
      		if(frame.empty())
        		break;

      		frame.copyTo( image );

            clock_t t1 = clock();
	  		
    		//updates the tracker successfully
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
      			cv::circle( image, center, 2, cv::Scalar(0, 255, 0), 3, 1 );

                clock_t t2 = clock();
                cout << "double tracker_update consumes time: " << 1.0 * ( t2 - t1 ) / CLOCKS_PER_SEC << " s\n";

                /********** compute current direction angle through boosting method *************/

                float uplimitAngle = 180;
                float downlimitAngle = -180;

                cv::Size patchsize = directionClf->params.patchSize;
                cv::Mat imgPart = frame( cv::Rect( center.x - patchsize.width, center.y - patchsize.height, patchsize.width * 2, patchsize.height * 2 ) );
                cv::Point centerPart( patchsize.width, patchsize.height ); // center of the imgPart

                for( int i = 0; i < 7; i++ ){	
                    float midAngle = ( uplimitAngle + downlimitAngle ) / 2;
                    cv::Mat rotateMat = cv::getRotationMatrix2D( centerPart, midAngle, 1 ); // rotation counter-clockwise
                    cv::Mat img;
                    cv::warpAffine( imgPart, img, rotateMat, cv::Size( imgPart.cols, imgPart.rows ) ); 
		       
			        int label = directionClf->classifierSample( img( cv::Rect2d( centerPart.x - patchsize.width / 2, 
                                                                                 centerPart.y - patchsize.height / 2, 
                                                                                 patchsize.width, 
                                                                                 patchsize.height ) ) );

			        char outNameChar[50];

			        static int outImageNum = 0;

                    if( label == 1 ){   // direction: down
                    	downlimitAngle = midAngle;
                    	//sprintf( outNameChar, "./run_imgs/image%d_conf_pos.png", outImageNum );
                    }
                    else{               // direction: up
                    	uplimitAngle = midAngle;
                        //sprintf( outNameChar, "./run_imgs/image%d_conf_neg.png", outImageNum );
                    }
    			}

                cout << "directionClf consumes time: " << 1.0 * ( clock() - t2 ) / CLOCKS_PER_SEC << " s\n";

                Eigen::Vector2d endP = Eigen::Rotation2Dd( ( uplimitAngle + downlimitAngle ) / 2.0 * PI / 180 ) * Eigen::Vector2d( 50, 0 );
                cv::line( image, center, cv::Point2d( center.x + endP[0], center.y + endP[1] ), cv::Scalar(0, 0, 255), 2, 1 );
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
