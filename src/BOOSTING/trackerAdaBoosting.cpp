#include "BOOSTING/trackerAdaBoosting.hpp"

namespace BOOSTING
{

class TrackerBoostingImpl : public TrackerBoosting{

public:
	TrackerBoostingImpl( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() );
	void read( const cv::FileNode& fn );
	void write( cv::FileStorage& fs ) const;

 protected:

	bool initImpl( const cv::Mat& image, const cv::Rect2d& boundingBox );
	bool updateImpl( const cv::Mat& image, cv::Rect2d& boundingBox );

	TrackerBoosting::Params params;
};

/***************************** Tracker **********************************/
Tracker::~Tracker(){

}

bool Tracker::init( cv::InputArray image, const cv::Rect2d& boundingBox ){
	// Initialize the tracker with a know bounding box that surrounding the target. 
	if( isInit )
		return false;

	if( image.empty() )
		return false;

	sampler = cv::Ptr<TrackerSampler>( new TrackerSampler() );
	featureSet = cv::Ptr<TrackerFeatureSet>( new TrackerFeatureSet() );
	model = cv::Ptr<TrackerModel>();

	bool initTracker = initImpl( image.getMat(), boundingBox );

	// check if the model component is initialized
	if( model == 0 ){
		CV_Error( -1, "The model is not initialized" );
		return false;
	}

	if( initTracker )
		isInit = true;

	return initTracker;
}

bool Tracker::update( cv::InputArray image, cv::Rect2d& boundingBox ){
	if( !isInit )
		return false;

	if( image.empty() )
		return false;

	return updateImpl( image.getMat(), boundingBox );
}

/************************ TrackerBoosting **********************/
/*
 * Parameters
 */
TrackerBoosting::Params::Params(){
	numClassifiers = 100;
	samplerOverlap = 0.99f;
	samplerSearchFactor = 1.8f;
	iterationInit = 50;
	featureSetNumFeatures = ( numClassifiers * 10 ) + iterationInit;
}

void TrackerBoosting::Params::read( const cv::FileNode& fn ){
	numClassifiers = fn["numClassifiers"];
	samplerOverlap = fn["overlap"];
	samplerSearchFactor = fn["samplerSearchFactor"];
	iterationInit = fn["iterationInit"];
	samplerSearchFactor = fn["searchFactor"];
}

void TrackerBoosting::Params::write( cv::FileStorage& fs ) const{
	fs << "numClassifiers" << numClassifiers;
	fs << "overlap" << samplerOverlap;
	fs << "searchFactor" << samplerSearchFactor;
	fs << "iterationInit" << iterationInit;
	fs << "samplerSearchFactor" << samplerSearchFactor;
}

cv::Ptr<TrackerBoosting> TrackerBoosting::create( const TrackerBoosting::Params& parameters ){
	return cv::Ptr<TrackerBoostingImpl>( new TrackerBoostingImpl( parameters ) );
}

cv::Ptr<TrackerBoosting> TrackerBoosting::create(){
	return cv::Ptr<TrackerBoostingImpl>( new TrackerBoostingImpl() );
}

/********************************* TrackerBoostingImpl ***************************************/

TrackerBoostingImpl::TrackerBoostingImpl( const TrackerBoosting::Params& parameters ) : params( parameters ){
	isInit = false;
}

void TrackerBoostingImpl::read( const cv::FileNode& fn ){
	params.read( fn );
}

void TrackerBoostingImpl::write( cv::FileStorage& fs ) const{
	params.write( fs );
}

bool TrackerBoostingImpl::initImpl( const cv::Mat& image, const cv::Rect2d& boundingBox ){
	std::srand( 1 );

	// sampling
	cv::Mat_<int> intImage;
	cv::Mat_<double> intSqImage;
	cv::Mat image_;
	cv::cvtColor( image, image_, CV_RGB2GRAY );
	cv::integral( image_, intImage, intSqImage, CV_32S );

	TrackerSamplerCS::Params CSparameters;
	CSparameters.overlap = params.samplerOverlap;
	CSparameters.searchFactor = params.samplerSearchFactor;
	cv::Ptr<TrackerSamplerAlgorithm> CSSampler = cv::Ptr<TrackerSamplerCS>( new TrackerSamplerCS( CSparameters ) );

	if( !sampler->addTrackerSamplerAlgorithm( CSSampler ) )
		return false;

	CSSampler.staticCast<TrackerSamplerCS>()->setMode( TrackerSamplerCS::MODE_POSITIVE );
	sampler->sampling( intImage, boundingBox );
	const std::vector<cv::Mat> posSamples = sampler->getSamples();

	CSSampler.staticCast<TrackerSamplerCS>()->setMode( TrackerSamplerCS::MODE_NEGATIVE );
	sampler->sampling( intImage, boundingBox );
	const std::vector<cv::Mat> negSamples = sampler->getSamples();

	if( posSamples.empty() || negSamples.empty() )
		return false;

	cv::Rect ROI = CSSampler.staticCast<TrackerSamplerCS>()->getsampleROI();

	// compute HAAR features
	TrackerFeatureHAAR::Params HAARparameters;
	HAARparameters.numFeatures = params.featureSetNumFeatures;
	HAARparameters.isIntegral = true;
	HAARparameters.rectSize = cv::Size( static_cast<int>(boundingBox.width), static_cast<int>(boundingBox.height) );
	cv::Ptr<TrackerFeature> trackerFeature = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters ) );

	if( !featureSet->addTrackerFeature( trackerFeature ) )
		return false;

	featureSet->extraction( posSamples );
	const std::vector<cv::Mat> posResponse = featureSet->getResponses();
	featureSet->extraction( negSamples );
	const std::vector<cv::Mat> negResponse = featureSet->getResponses();

	// Model
	model = cv::Ptr<TrackerBoostingModel>( new TrackerBoostingModel( boundingBox ) );
	cv::Ptr<TrackerStateEstimatorAdaBoosting> stateEstimator = cv::Ptr<TrackerStateEstimatorAdaBoosting>( 
		new TrackerStateEstimatorAdaBoosting( params.numClassifiers, params.iterationInit, params.featureSetNumFeatures, 
											  cv::Size( (int)boundingBox.width, (int)boundingBox.height ), ROI ) );
	model->setTrackerStateEstimator( stateEstimator );

	// Run model estimation and update for iterationInit iterations
	for( int i = 0; i < params.iterationInit; i++ ){
		// compute temp features
		TrackerFeatureHAAR::Params HAARparameters2;
		HAARparameters2.numFeatures = static_cast<int>( posSamples.size() + negSamples.size() );
		HAARparameters2.isIntegral = true;
		HAARparameters2.rectSize = cv::Size( (int)boundingBox.width, (int)boundingBox.height );

		cv::Ptr<TrackerFeatureHAAR> trackerFeature2 = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters2 ) );

		model.staticCast<TrackerBoostingModel>()->setMode( TrackerBoostingModel::MODE_NEGATIVE, negSamples );
		model->evalCurrentConfidenceMap( negResponse );
		model.staticCast<TrackerBoostingModel>()->setMode( TrackerBoostingModel::MODE_POSITIVE, posSamples );
		model->evalCurrentConfidenceMap( posResponse );
		model->modelUpdate();

		// get replaced classifier and change the features
		std::vector<int> replacedClassifier = stateEstimator->computeReplacedClassifier();
		std::vector<int> swappedClassifier = stateEstimator->computeSwappedClassifier();
		for( size_t j = 0; j < replacedClassifier.size(); j++ ){
			if( replacedClassifier[j] != -1 && swappedClassifier[j] != -1 ){
				trackerFeature.staticCast<TrackerFeatureHAAR>()->swapFeature( replacedClassifier[j], swappedClassifier[j] );
				trackerFeature.staticCast<TrackerFeatureHAAR>()->swapFeature( swappedClassifier[j], trackerFeature2->getFeatureAt( (int)j ) );
			}
		}
	}
	return true;
}

bool TrackerBoostingImpl::updateImpl( const cv::Mat& image, cv::Rect2d& boundingBox ){
	cv::Mat_<int> intImage;
	cv::Mat_<double> intSqImage;
	cv::Mat image_;
	cv::cvtColor( image, image_, CV_RGB2GRAY );
	cv::integral( image_, intImage, intSqImage, CV_32S );

	// get the last location X(k-1)
	cv::Ptr<TrackerTargetState> lastLocation = model->getLastTargetState();
	cv::Rect lastBoundingBox( (int)lastLocation->getTargetPosition().x, (int)lastLocation->getTargetPosition().y, 
							  lastLocation->getTargetWidth(), lastLocation->getTargetHeight() );

	// sampling new frame based on last location
	( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCS>()->setMode( TrackerSamplerCS::MODE_CLASSIFY );
	sampler->sampling( intImage, lastBoundingBox );
	const std::vector<cv::Mat> detectSamples = sampler->getSamples();
	cv::Rect ROI = ( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCS>()->getsampleROI();

	if( detectSamples.empty() )
		return false;

	/*//TODO debug samples
	Mat f;
	image.copyTo( f );

	for ( size_t i = 0; i < detectSamples.size(); i = i + 10 )
	{
	Size sz;
	Point off;
	detectSamples.at( i ).locateROI( sz, off );
	rectangle( f, Rect( off.x, off.y, detectSamples.at( i ).cols, detectSamples.at( i ).rows ), Scalar( 255, 0, 0 ), 1 );
	}*/

	std::vector<cv::Mat> responses;
	cv::Mat response;

	std::vector<int> classifiers = model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorAdaBoosting>()->computeSelectedWeakClassifier();
	cv::Ptr<TrackerFeatureHAAR> extractor = featureSet->getTrackerFeature()[0].second.staticCast<TrackerFeatureHAAR>();
	extractor->extractSelected( classifiers, detectSamples, response );
	responses.push_back( response );

	// predict new location
	ConfidenceMap cmap;
	model.staticCast<TrackerBoostingModel>()->setMode( TrackerBoostingModel::MODE_CLASSIFY, detectSamples );
	model.staticCast<TrackerBoostingModel>()->responseToConfidenceMap( responses, cmap );
	model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorAdaBoosting>()->setCurrentConfidenceMap( cmap );
	model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorAdaBoosting>()->setSampleROI( ROI );

	if( !model->runStateEstimator() )
		return false;

	cv::Ptr<TrackerTargetState> currentState = model->getLastTargetState();
	boundingBox = cv::Rect( (int)currentState->getTargetPosition().x, (int)currentState->getTargetPosition().y, 
							currentState->getTargetWidth(), currentState->getTargetHeight() );

	/*//TODO debug
	rectangle( f, lastBoundingBox, Scalar( 0, 255, 0 ), 1 );
	rectangle( f, boundingBox, Scalar( 0, 0, 255 ), 1 );
	imshow( "f", f );
	//waitKey( 0 );*/

	/************************************* 更新 model **************************************/

	// sampling new frame based on new location

	// Positive sampling
	( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCS>()->setMode( TrackerSamplerCS::MODE_POSITIVE );
	sampler->sampling( intImage, boundingBox );
	const std::vector<cv::Mat> posSamples = sampler->getSamples();

	// Negative sampling
	( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCS>()->setMode( TrackerSamplerCS::MODE_NEGATIVE );
	sampler->sampling( intImage, boundingBox );
	const std::vector<cv::Mat> negSamples = sampler->getSamples();

	if( posSamples.empty() || negSamples.empty() )
		return false;

	// extract features
	featureSet->extraction( posSamples );
	const std::vector<cv::Mat> posResponse = featureSet->getResponses();
	featureSet->extraction( negSamples );
	const std::vector<cv::Mat> negResponse = featureSet->getResponses();

	// compute temp features
	TrackerFeatureHAAR::Params HAARparameters2;
	HAARparameters2.numFeatures = static_cast<int>( posSamples.size() + negSamples.size() );
	HAARparameters2.isIntegral = true;
	HAARparameters2.rectSize = cv::Size( static_cast<int>(boundingBox.width), static_cast<int>(boundingBox.height) );
	cv::Ptr<TrackerFeatureHAAR> trackerFeature2 = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters2 ) );

	// model estimate
	model.staticCast<TrackerBoostingModel>()->setMode( TrackerBoostingModel::MODE_NEGATIVE, negSamples );
	model->evalCurrentConfidenceMap( negResponse );
	model.staticCast<TrackerBoostingModel>()->setMode( TrackerBoostingModel::MODE_POSITIVE, posSamples );
	model->evalCurrentConfidenceMap( posResponse );

	// model update
	model->modelUpdate();

	// get replaced classifier and change the features
	std::vector<int> replacedClassifier = model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorAdaBoosting>()->computeReplacedClassifier();
	std::vector<int> swappedClassifier = model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorAdaBoosting>()->computeSwappedClassifier();
	for( size_t j = 0; j < replacedClassifier.size(); j++ ){
		if( replacedClassifier[j] != -1 && swappedClassifier[j] != -1 ){
			featureSet->getTrackerFeature().at( 0 ).second.staticCast<TrackerFeatureHAAR>()->swapFeature( replacedClassifier[j], swappedClassifier[j] );
			featureSet->getTrackerFeature().at( 0 ).second.staticCast<TrackerFeatureHAAR>()->swapFeature( swappedClassifier[j], trackerFeature2->getFeatureAt( (int)j ) );
		}
	}

	return true;
}



} /* namespace BOOSTING */