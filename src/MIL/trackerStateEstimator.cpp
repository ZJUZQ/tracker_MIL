#include "MIL/trackerStateEstimator.hpp"

namespace MIL
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
	*/

	if( trackerStateEstimatorType.find( "MIL" ) == 0 ){
		return cv::Ptr<TrackerStateEstimatorMILBoosting>(new TrackerStateEstimatorMILBoosting() );
	}

	CV_Error( -1, "Tracker state estimator type not supported" );
	return cv::Ptr<TrackerStateEstimator>();
}

std::string TrackerStateEstimator::getClassName() const{
	return className;
}


/*************************** TrackerStateEstimatorMILBoosting ********************************/

/**
 * TrackerStateEstimatorMILBoosting::TrackerMILTargetState
 */
TrackerStateEstimatorMILBoosting::TrackerMILTargetState::TrackerMILTargetState( const cv::Point2f& position, int width, int height, bool foreground, const cv::Mat& features ){
	setTargetPosition( position );
	setTargetWidth( width );
	setTargetHeight( height );
	setTargetFg( foreground );
	setFeatures( features );
}	

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setTargetFg( bool foreground ){
	// Set label: true for target foreground, false for background. 
	isTarget = foreground;
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setFeatures( const cv::Mat& features ){
	targetFeatures = features;
}

bool TrackerStateEstimatorMILBoosting::TrackerMILTargetState::isTargetFg() const{
  	return isTarget;
}

cv::Mat TrackerStateEstimatorMILBoosting::TrackerMILTargetState::getFeatures() const{
  return targetFeatures;
}

/**
 * TrackerStateEstimatorMILBoosting
 */

TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting( int nFeatures ){
	className = "BOOSTING";
	trained = false;
	numFeatures = nFeatures;  // nFeatures:  Number of features for each sample 
}

TrackerStateEstimatorMILBoosting::~TrackerStateEstimatorMILBoosting(){

}

void TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap ){
	currentConfidenceMap.clear();
	currentConfidenceMap = confidenceMap;
}

uint TrackerStateEstimatorMILBoosting::max_idx( const std::vector<float> &v ){
	const float* findPtr = & ( *std::max_element( v.begin(), v.end() ) );
	const float* beginPtr = & ( *v.begin() );
	return (uint) ( findPtr - beginPtr );
}

void TrackerStateEstimatorMILBoosting::prepareData( const ConfidenceMap& confidenceMap, cv::Mat& positive, cv::Mat& negative ){
  // 从confidenceMap中提取正样本及其特征，负样本及其特征

	int posCounter = 0;
	int negCounter = 0;

	for ( size_t i = 0; i < confidenceMap.size(); i++ ){
		cv::Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at( i ).first.staticCast<TrackerMILTargetState>();
		if( currentTargetState->isTargetFg() )
			posCounter++;
		else
		    negCounter++;
	}

	positive.create( posCounter, numFeatures, CV_32FC1 );
	negative.create( negCounter, numFeatures, CV_32FC1 );

	//TODO change with mat fast access
	//initialize trainData (positive and negative)

	int pc = 0;
	int nc = 0;
	for ( size_t i = 0; i < confidenceMap.size(); i++ ){
		cv::Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at( i ).first.staticCast<TrackerMILTargetState>();
		cv::Mat stateFeatures = currentTargetState->getFeatures();  // Get the features extracted. 

		if( currentTargetState->isTargetFg() ){
			for ( int j = 0; j < stateFeatures.rows; j++ ){
				//fill the positive trainData with the value of the feature j for sample i
				positive.at<float>( pc, j ) = stateFeatures.at<float>( j, 0 );
			}
			pc++;
		}

		else{
			for ( int j = 0; j < stateFeatures.rows; j++ ){
				//fill the negative trainData with the value of the feature j for sample i
				negative.at<float>( nc, j ) = stateFeatures.at<float>( j, 0 );
			}
			nc++;
		}
	}
}

cv::Ptr<TrackerTargetState> TrackerStateEstimatorMILBoosting::estimateImpl( const std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
	//run ClfMilBoost classify in order to compute next location
	// Estimate the most likely target state, return the estimated state. 

	if( currentConfidenceMap.empty() )
		return cv::Ptr<TrackerTargetState>();

	cv::Mat positiveStates;
	cv::Mat negativeStates;

	prepareData( currentConfidenceMap, positiveStates, negativeStates ); // 从confidenceMap中提取正样本及其特征，负样本及其特征

	std::vector<float> prob = boostMILModel.classify( positiveStates ); // 定义： ClfMilBoost boostMILModel;

	int bestind = max_idx( prob );
	//float resp = prob[bestind];

	//选择得分score最高的作为当前估计的状态
	return currentConfidenceMap.at( bestind ).first;  // 定义： typedef std::vector<std::pair<Ptr<TrackerTargetState>, float> > ConfidenceMap;
}

void TrackerStateEstimatorMILBoosting::updateImpl( std::vector<ConfidenceMap>& confidenceMaps ){
  // Update the ConfidenceMap with the scores

	if( !trained ){
		//this is the first time that the classifier is built
		//init MIL
		boostMILModel.init();
		trained = true;
	}

	ConfidenceMap lastConfidenceMap = confidenceMaps.back(); //  Returns a reference to the last element in the vector.
	cv::Mat positiveStates;
	cv::Mat negativeStates;

	prepareData( lastConfidenceMap, positiveStates, negativeStates );
	//update MIL
	boostMILModel.update( positiveStates, negativeStates );
}


} /* namespace MIL */