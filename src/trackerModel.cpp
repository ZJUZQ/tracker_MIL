#include "MIL/trackerModel.hpp"

namespace MIL
{

TrackerModel::TrackerModel(){
	stateEstimator = cv::Ptr<TrackerStateEstimator>();
	maxCMLength = 10;
}

TrackerModel::~TrackerModel(){

}

bool TrackerModel::setTrackerStateEstimator( cv::Ptr<TrackerStateEstimator> trackerStateEstimator ){
	// Set TrackerEstimator, return true if the tracker state estimator is added, false otherwise. 
	if( stateEstimator != 0 )
		return false;

	stateEstimator = trackerStateEstimator;
	return true;
}

cv::Ptr<TrackerStateEstimator> TrackerModel::getTrackerStateEstimator() const{
	return stateEstimator;
}

void TrackerModel::modelEstimation( const std::vector<cv::Mat>& responses ){
	// Estimate the most likely target location. 
	modelEstimationImpl( responses );
}

void TrackerModel::clearCurrentConfidenceMap(){
	currentConfidenceMap.clear();
}

void TrackerModel::modelUpdate(){
	// Update the model
	modelUpdateImpl();

	if( maxCMLength != -1 && (int)confidenceMaps.size() >= maxCMLength - 1 ){
		int l = maxCMLength / 2;
		confidenceMaps.erase( confidenceMaps.begin(), confidenceMaps.begin() + l );
		// emoves from the vector either a single element (position) or a range of elements ([first,last)).
    	// return value: An iterator pointing to the new location of the element that followed the last element erased by the function call. 
	}

	if( maxCMLength != -1 && (int)trajectory.size() >= maxCMLength - 1){
		int l = maxCMLength / 2;
		trajectory.erase(trajectory.begin(), trajectory.begin() + l );
	}

	confidenceMaps.push_back( currentConfidenceMap );
	stateEstimator->update( confidenceMaps );

	clearCurrentConfidenceMap();
}

bool TrackerModel::runStateEstimator(){
	// Run the TrackerStateEstimator, return true if is possible to estimate a new state, false otherwise. 
	if( stateEstimator == 0 ){
		cv::CV_Error(-1, "Tracker state estimator is not setted" );
		return false;
	}
	cv::Ptr<TrackerTargetState> targetState  = stateEstimator->estimate( confidenceMaps );
	if( targetState == 0 )
		return false;

	setLastTargetState( targetState );
	return true;
}

void TrackerModel::setLastTargetState( const cv::Ptr<TrackerTargetState>& lastTargetState ){
	trajectory.push_back( lastTargetState );
}

cv::Ptr<TrackerTargetState> TrackerModel::getLastTargetState() const{
	return trajectory.back();
}

const std::vector<ConfidenceMap>& TrackerModel::getConfidenceMaps() const{
  return confidenceMaps;
}

const ConfidenceMap& TrackerModel::getLastConfidenceMap() const{
  return confidenceMaps.back();
}


} /* namespace MIL */