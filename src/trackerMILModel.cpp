#include "MIL/trackerMILModel.hpp"

namespace MIL
{

TrackerMILModel::TrackerMILModel( const cv::Rect& boundingBox ){
	currentSample.clear();
	mode = MODE_POSITIVE;
	width = boundingBox.width;
	height = boundingBox.height;

	cv::Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> initState = cv::Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState>(
		new TrackerStateEstimatorMILBoosting::TrackerMILTargetState( cv::Point2f( (float)boundingBox.x, (float)boundingBox.y ), boundingBox.width, boundingBox.height, true, cv::Mat() ));

	trajectory.push_back(  initState );
}

void TrackerMILModel::responseToConfidenceMap( const std::vector<cv::Mat>& responses, ConfidenceMap& confidenceMap ){
	if( currentSample.empty() ){
		CV_Error( -1, "The samples in Model estimation are empyt" );
		return;
	}

	for( size_t i = 0; i < responses.size(); i++ ){
		//for each column (one sample) there are #num_feature
    	//get informations from currentSample

    	for( int j = 0; j < responses.at( i ).cols; j++ ){
    		// cv::Mat response -->  行表示特征编号，列表示样本编号

    		cv::Size currentSize;
    		cv::Point currentOfs;
    		currentSample.at( j ).locateROI( currentSize, currentOfs ); // std::vector<Mat> currentSample, 当前样本集合
    		/*
			  void cv::Mat::locateROI ( Size &    wholeSize,
			                            Point &   ofs 
			                          )     const
			  返回ROI所在的原图像的尺寸以及ROI在原图像的位置
			*/
    		bool foreground = false;
    		if( mode == MODE_POSITIVE || mode == MODE_ESTIMATION )
    			foreground = true;
    		else if( mode == MODE_NEGATIVE )
    			foreground = false;

    		// get the column of the HAAR responses
    		cv::Mat singleResponse = responses.at( i ).col( j ); // 样本j的所有HAAR特征的计算值

    		// create the state
    		// TrackerStateEstimatorMILBoosting::TrackerMILTargetState::TrackerMILTargetState( const cv::Point2f position, int width, int height, bool foreground, const cv::Mat& features );
    		cv::Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> currentState = cv::Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState>( 
    			new TrackerStateEstimatorMILBoosting::TrackerMILTargetState( currentOfs, width, height, foreground, singleResponse ) );

    		confidenceMap.push_back( std::make_pair( currentState, 0.0f ) );
    	}
	}
}

void TrackerMILModel::modelEstimationImpl( const std::vector<cv::Mat>& responses ){
	responseToConfidenceMap( responses, currentConfidenceMap );
}

void TrackerMILModel::modelUpdateImpl(){

}

void TrackerMILModel::setMode( int trainingMode, const std::vector<cv::Mat>& samples ){
	currentSample.clear();
	currentSample = samples;

	mode = trainingMode;
}


} /* namespace MIL */