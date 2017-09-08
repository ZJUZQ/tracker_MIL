#include "MIL/trackerStateEstimator.hpp"

namespace MIL
{

TrackerStateEstimator::~TrackerStateEstimator(){

}

cv::Ptr<TrackerTargetState> TrackerStateEstimator::estimate( const std::vector<ConfidenceMap>& confidenceMaps ){
	// Estimate the most likey target state, return the estimated state
	if( confidenceMaps.empty() )
		retrun cv::Ptr<TrackerTargetState>();

	return estimateImpl( confidenceMaps );
}

void TrackerStateEstimator::update( std::vector<ConfidenceMap>& confidenceMaps ){
	// Update the ConfidenceMap with the scores
	if( confidenceMaps.empty() )
		return;

	return updateImpl( confidenceMaps );
}

cv::Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const std::string& trackerStateEstimatorType ){
	// the modes available now: BOOSTING, SVM
	if( trackerStateEstimatorType.find( "SVM" ) == 0 ){
		return cv::Ptr<TrackerStateEstimatorSVM>(new TrackerStateEstimatorSVM() );
	}

	if( trackerStateEstimatorType.find( "BOOSTING" ) == 0 ){
		return cv::Ptr<TrackerStateEstimatorMILBOOSTING>(new TrackerStateEstimatorMILBOOSTING() );
	}

	cv::CV_Error( -1, "Tracker state estimator type not supported" );
	return cv::Ptr<TrackerStateEstimator>();
}

std::string TrackerStateEstimator::getClassName() const{
	return className;
}


} /* namespace MIL */