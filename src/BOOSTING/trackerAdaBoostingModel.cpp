#include "BOOSTING/trackerAdaBoostingModel.hpp"

namespace BOOSTING
{

/********************************** TrackerBoostingModel *************************************/

TrackerBoostingModel::TrackerBoostingModel( const cv::Rect& boundingBox ){

 	mode = MODE_POSITIVE;

  	cv::Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState> initState =
      	cv::Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState>(
          new TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState( cv::Point2f( (float)boundingBox.x, (float)boundingBox.y ), boundingBox.width,
                                                                               boundingBox.height, true, cv::Mat() ) );
  	trajectory.push_back( initState );
  	maxCMLength = 10;
}

void TrackerBoostingModel::modelEstimationImpl( const std::vector<cv::Mat>& responses ){
	responseToConfidenceMap( responses, currentConfidenceMap );
}

void TrackerBoostingModel::modelUpdateImpl(){

}

void TrackerBoostingModel::setMode( int trainingMode, const std::vector<cv::Mat>& samples ){
  	currentSamples.clear();
  	currentSamples = samples;

  	mode = trainingMode;
}

std::vector<int> TrackerBoostingModel::getSelectedWeakClassifier(){
  	return stateEstimator.staticCast<TrackerStateEstimatorAdaBoosting>()->computeSelectedWeakClassifier();
}

void TrackerBoostingModel::responseToConfidenceMap( const std::vector<cv::Mat>& responses, ConfidenceMap& confidenceMap ){
	if( currentSamples.empty() ){
		CV_Error( -1, "The samples in Model estimation are empty" );
		return;
	}

	for ( size_t i = 0; i < currentSamples.size(); i++ ){

		cv::Size currentSize;
		cv::Point currentOfs;

		/*
		void cv::Mat::locateROI (  Size&  wholeSize,  Point& ofs )  const

		Parameters
		  wholeSize : Output parameter that contains the size of the whole matrix containing this as a part.
		  ofs       : Output parameter that contains an offset of this inside the whole matrix. 
		*/
		currentSamples.at( i ).locateROI( currentSize, currentOfs );

		bool foreground = false;
		if( mode == MODE_POSITIVE || mode == MODE_CLASSIFY )
			foreground = true;
		else if( mode == MODE_NEGATIVE )
			foreground = false;

		const cv::Mat resp = responses[0].col( (int)i ); // responses[0]: 类型Mat, 列表示samples, 行表示features

		//create the state
		cv::Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState> currentState = cv::Ptr<TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState>( 
				new TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState( currentOfs, currentSamples.at( i ).cols, currentSamples.at( i ).rows, foreground, resp ) );

		confidenceMap.push_back( std::make_pair( currentState, 0.0f ) );

	}
}


} /* namespace BOOSTING */