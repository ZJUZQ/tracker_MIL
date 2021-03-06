#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#include "MIL/trackerMIL.hpp"
#include "MIL/roiSelector.hpp"

using namespace std;

static const char* keys =
{   "{@video_name      | | video name        }"
    "{@start_frame     |0| Start frame       }"
    "{@bounding_frame  |0,0,0,0| Initial bounding frame}"};

static void help()
{
  cout << "usage: ./tracker_MILBoosting <video_name> <start_frame> [<bounding_frame>]\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
}

int main( int argc, char** argv ){
	cv::CommandLineParser parser( argc, argv, keys );

	string video_name = parser.get<string>( 0 );
	int start_frame = parser.get<int>( 1 );

	if( video_name.empty() ){
		help();
		return -1;
	}

  	int coords[4]={0,0,0,0};
  	bool initBoxWasGivenInCommandLine=false;

	{
		string initBoundingBox = parser.get<string>(2);

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
	cv::namedWindow( "Tracking_MILBoosting", 1 );

	cv::Mat image;
	cv::Rect2d boundingBox;
	bool paused = false;

  	//instantiates the specific Tracker
  	cv::Ptr<MIL::Tracker> tracker = MIL::TrackerMIL::create();
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
  		boundingBox = MIL::selectROI("Tracking_MILBoosting", image);
  	}
    

  	cv::imshow( "Tracking_MILBoosting", image );

  	bool initialized = false;
  	int frameCounter = 0;

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
	    		}
	  		}

	  		cv::imshow( "Tracking_MILBoosting", image );
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
