#include "BOOSTING/trackerStateEstimator.hpp"

namespace BOOSTING
{

/********************************** TrackerStateEstimator ***********************************/

TrackerStateEstimator::~TrackerStateEstimator(){

}

cv::Ptr<TrackerTargetState> TrackerStateEstimator::estimate( const std::vector<ConfidenceMap>& confidenceMaps ){
	// Estimate the most likey target state, return the estimated state
	if( confidenceMaps.empty() )
		return cv::Ptr<TrackerTargetState>();

	return estimateImpl( confidenceMaps ); // typedef std::vector< std::pair<cv::Ptr<TrackerTargetState>, float> > ConfidenceMap;
}

void TrackerStateEstimator::update( std::vector<ConfidenceMap>& confidenceMaps ){
	// Update the ConfidenceMap with the scores
	if( confidenceMaps.empty() )
		return;

	return updateImpl( confidenceMaps );
}

cv::Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const std::string& trackerStateEstimatorType ){
	// the modes available now: BOOSTING, SVM

	/**
	if( trackerStateEstimatorType.find( "SVM" ) == 0 ){
		return cv::Ptr<TrackerStateEstimatorSVM>(new TrackerStateEstimatorSVM() );
	}
	
	if( trackerStateEstimatorType.find( "MIL" ) == 0 ){
		return cv::Ptr<TrackerStateEstimatorMILBoosting>(new TrackerStateEstimatorMILBoosting() );
	}
	*/

	if( trackerStateEstimatorType.find( "BOOSTING" ) == 0 ){
		return cv::Ptr<TrackerStateEstimatorAdaBoosting>(new TrackerStateEstimatorAdaBoosting() );
	}

	CV_Error( -1, "Tracker state estimator type not supported" );
	return cv::Ptr<TrackerStateEstimator>();
}

std::string TrackerStateEstimator::getClassName() const{
	return className;
}


/*************************** TrackerStateEstimatorAdaBoosting ********************************/

/**
 * TrackerStateEstimatorAdaBoosting
 */

TrackerStateEstimatorAdaBoosting::TrackerStateEstimatorAdaBoosting() {
	className = "ADABOOSTING";
	numBaseClassifier = 100;
	iterationInit = 50;
	numFeatures = ( numBaseClassifier * 10 ) + iterationInit;
	initPatchSize = cv::Size();
	trained = false;
	sampleROI = cv::Rect();
}

TrackerStateEstimatorAdaBoosting::TrackerStateEstimatorAdaBoosting( int numClassifer, int initIterations, int nFeatures, cv::Size patchSize, const cv::Rect& ROI ){
	className = "ADABOOSTING";
	numBaseClassifier = numClassifer;
	numFeatures = nFeatures;
	iterationInit = initIterations;
	initPatchSize = patchSize;
	trained = false;
	sampleROI = ROI;
}

cv::Rect TrackerStateEstimatorAdaBoosting::getSampleROI() const{
  	return sampleROI;
}

void TrackerStateEstimatorAdaBoosting::setSampleROI( const cv::Rect& ROI ){
  	sampleROI = ROI;
}

/**
 * TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState
 */
TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::TrackerAdaBoostingTargetState( const cv::Point2f& position, int width, int height,
                                                                                                bool foreground, const cv::Mat& responses ){
	setTargetPosition( position );
	setTargetWidth( width );
	setTargetHeight( height );

	setTargetFg( foreground );
	setTargetResponses( responses );
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetFg( bool foreground ){
  	isTarget = foreground;
}

bool TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::isTargetFg() const{
  	return isTarget;
}

void TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::setTargetResponses( const cv::Mat& responses ){
  	targetResponses = responses;
}

cv::Mat TrackerStateEstimatorAdaBoosting::TrackerAdaBoostingTargetState::getTargetResponses() const{
  	return targetResponses;
}

TrackerStateEstimatorAdaBoosting::~TrackerStateEstimatorAdaBoosting(){

}

void TrackerStateEstimatorAdaBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap ){
  	currentConfidenceMap.clear();
  	currentConfidenceMap = confidenceMap;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeReplacedClassifier(){
  	return replacedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSwappedClassifier(){
  	return swappedClassifier;
}

std::vector<int> TrackerStateEstimatorAdaBoosting::computeSelectedWeakClassifier(){
  	return boostClassifier->getSelectedWeakClassifier();
}

cv::Ptr<TrackerTargetState> TrackerStateEstimatorAdaBoosting::estimateImpl( const std::vector<ConfidenceMap>& /*confidenceMaps*/ ){
	//run classify in order to compute next location

	if( currentConfidenceMap.empty() )
		return cv::Ptr<TrackerTargetState>();

	std::vector<cv::Mat> respColSet;

	// typedef std::vector<std::pair<Ptr<TrackerTargetState>, float> > cv::ConfidenceMap
	for ( size_t i = 0; i < currentConfidenceMap.size(); i++ ){
		cv::Ptr<TrackerAdaBoostingTargetState> currentTargetState = currentConfidenceMap.at( i ).first.staticCast<TrackerAdaBoostingTargetState>();
		respColSet.push_back( currentTargetState->getTargetResponses() );
	}

	int bestIndex;
	// 定义: Ptr<StrongClassifierDirectSelection> boostClassifier;
	boostClassifier->classifySmooth( respColSet, sampleROI, bestIndex );

	// get bestIndex from classifySmooth
	return currentConfidenceMap.at( bestIndex ).first;
}

void TrackerStateEstimatorAdaBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps ){
	if( !trained ){
		// this is the first time that the classifier is built
		int numWeakClassifier = numBaseClassifier * 10;
		bool useFeatureExchange = true;
		boostClassifier = cv::Ptr<StrongClassifierDirectSelection>( 
			new StrongClassifierDirectSelection( numBaseClassifier, numWeakClassifier, initPatchSize, sampleROI, useFeatureExchange, iterationInit ) );
		// init base classifiers
		boostClassifier->initBaseClassifiers();

		trained = true;
	}

	ConfidenceMap lastConfidenceMap = confidenceMaps.back();
	bool featureEx = boostClassifier->getUseFeatureExchange();

	/* each training sample will produce one bad weakclassifier to be replaced */
	replacedClassifier.clear();
	replacedClassifier.resize( lastConfidenceMap.size(), -1 ); 
	swappedClassifier.clear();
	swappedClassifier.resize( lastConfidenceMap.size(), -1 );

	for( size_t i = 0; i < lastConfidenceMap.size(); i++ ){
	// for each training sample

		cv::Ptr<TrackerAdaBoostingTargetState> currentTargetState = lastConfidenceMap.at( i ).first.staticCast<TrackerAdaBoostingTargetState>();

		int currentFg = 1;
		if( !currentTargetState->isTargetFg() )
			currentFg = -1;

		cv::Mat respCol = currentTargetState->getTargetResponses();

		boostClassifier->update( respCol, currentFg ); // for each training sample, update all weakclassifiers and strongClassifier, Algorithm 2.1

		if( featureEx ){
			replacedClassifier[i] = boostClassifier->getReplacedClassifier(); // each traing sample will produce one bad weakclassifier to be replaced
			swappedClassifier[i] = boostClassifier->getSwappedClassifier();
			if( replacedClassifier[i] >= 0 && swappedClassifier[i] >= 0 )
				boostClassifier->replaceWeakClassifier( replacedClassifier[i] );
		}
		else{
			replacedClassifier[i] = -1;
			swappedClassifier[i] = -1;
		}

	}
}



} /* namespace BOOSTING */