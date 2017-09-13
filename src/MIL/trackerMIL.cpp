#include "MIL/trackerMIL.hpp"

namespace MIL
{

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


/***************************** TrackerMIL *************************************/
TrackerMIL::Params::Params(){
	samplerInitInRadius = 3;
	samplerSearchWinSize = 25;
	samplerInitMaxNegNum = 65;
	samplerTrackInRadius = 4;
	samplerTrackMaxPosNum = 100000;
	samplerTrackMaxNegNum = 65;
	featureSetNumFeatures = 250;
}

void TrackerMIL::Params::read( const cv::FileNode& fn ){
	samplerInitInRadius = fn["samplerInitInRadius"];
	samplerSearchWinSize = fn["samplerSearchWinSize"];
	samplerInitMaxNegNum = fn["samplerInitMaxNegNum"];
	samplerTrackInRadius = fn["samplerTrackInRadius"];
	samplerTrackMaxPosNum = fn["samplerTrackMaxPosNum"];
	samplerTrackMaxNegNum = fn["samplerTrackMaxNegNum"];
	featureSetNumFeatures = fn["featureSetNumFeatures"];
}

void TrackerMIL::Params::write( cv::FileStorage& fs ) const{
	fs << "samplerInitInRadius" << samplerInitInRadius;
	fs << "samplerSearchWinSize" << samplerSearchWinSize;
	fs << "samplerInitMaxNegNum" << samplerInitMaxNegNum;
	fs << "samplerTrackInRadius" << samplerTrackInRadius;
	fs << "samplerTrackMaxPosNum" << samplerTrackMaxPosNum;
	fs << "samplerTrackMaxNegNum" << samplerTrackMaxNegNum;
	fs << "featureSetNumFeatures" << featureSetNumFeatures;

}

cv::Ptr<TrackerMIL> TrackerMIL::create( const TrackerMIL::Params& parameters ){
	return cv::Ptr<TrackerMILImpl>( new TrackerMILImpl( parameters ) );
}

cv::Ptr<TrackerMIL> TrackerMIL::create(){
	return cv::Ptr<TrackerMILImpl>( new TrackerMILImpl() );
}

/************************** TrackerMILImpl *********************************/
TrackerMILImpl::TrackerMILImpl( const TrackerMIL::Params& parameters ) : params( parameters ){
	isInit = false;
}

void TrackerMILImpl::read( const cv::FileNode& fn ){
	params.read( fn );
}

void TrackerMILImpl::write( cv::FileStorage& fs ) const{
	params.write( fs );
}

void TrackerMILImpl::compute_integral( const cv::Mat& img, cv::Mat& ii_img ){
	cv::Mat ii;
	std::vector<cv::Mat> ii_imgs;
	cv::integral( img, ii, CV_32F );	// 计算img的累积图像ii

	cv::split( ii, ii_imgs ); 	// Divides a multi-channel array into several single-channel arrays. 
                        		//  the number of arrays (ii_imgs) must match ii.channels()
	ii_img = ii_imgs[0];
}

bool TrackerMILImpl::initImpl( const cv::Mat& image, const cv::Rect2d& boundingBox ){
	// Initialize the tracker with a know bounding box that surrounding the target

	std::srand( 1 ); // Initialize random number generator

	cv::Mat intImage;
	compute_integral( image, intImage );

	TrackerSamplerCSC::Params CSCparameters;
	CSCparameters.initInRad = params.samplerInitInRadius;
	CSCparameters.searchWinSize = params.samplerSearchWinSize;
	CSCparameters.initMaxNegNum = params.samplerInitMaxNegNum;
	CSCparameters.trackInPosRad = params.samplerTrackInRadius;
	CSCparameters.trackMaxPosNum = params.samplerTrackMaxPosNum;
	CSCparameters.trackMaxNegNum = params.samplerTrackMaxNegNum;

	cv::Ptr<TrackerSamplerAlgorithm> CSCSampler = cv::Ptr<TrackerSamplerCSC>( new TrackerSamplerCSC( CSCparameters ) );
	if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
		return false;

	// or add CSC sampler with default parameters
  	//sampler->addTrackerSamplerAlgorithm( "CSC" );

	// Positive sampling
	CSCSampler.staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_POS );
	sampler->sampling( intImage, boundingBox );
	std::vector<cv::Mat> posSamples = sampler->getSamples();

	// Negative sampling
	CSCSampler.staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_NEG );
	sampler->sampling( intImage, boundingBox );
	std::vector<cv::Mat> negSamples = sampler->getSamples();

	if( posSamples.empty() || negSamples.empty() )
		return false;

	// compute HAAR features
	TrackerFeatureHAAR::Params HAARparameters;
	HAARparameters.numFeatures = params.featureSetNumFeatures;
	HAARparameters.rectSize = cv::Size( (int)boundingBox.width, (int)boundingBox.height );
	HAARparameters.isIntegral = true;
	cv::Ptr<TrackerFeature> trackerFeature = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters ) );
	featureSet->addTrackerFeature( trackerFeature );

	featureSet->extraction( posSamples );
	const std::vector<cv::Mat> posResponse = featureSet->getResponses();

	featureSet->extraction( negSamples );
	const std::vector<cv::Mat> negResponse = featureSet->getResponses();

	model = cv::Ptr<TrackerMILModel>( new TrackerMILModel( boundingBox ) );
	cv::Ptr<TrackerStateEstimatorMILBoosting> stateEstimator = cv::Ptr<TrackerStateEstimatorMILBoosting>( 
			new TrackerStateEstimatorMILBoosting( params.featureSetNumFeatures ) );
	model->setTrackerStateEstimator( stateEstimator );

	// Run model estimation and update
	model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_POSITIVE, posSamples );
	model->modelEstimation( posResponse );
	model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_NEGATIVE, negSamples );
	model->modelEstimation( negResponse );
	model->modelUpdate();

	return true;
}

bool TrackerMILImpl::updateImpl( const cv::Mat& image, cv::Rect2d& boundingBox ){
	// 1) Find the new most likely bounding box for the target;
	// 2) Update the tracker, 

	cv::Mat intImage;
	compute_integral( image, intImage );

	/************* step 1: estimate the new most likely boundingbox for the target ********************/

	// get the last location [AAM] X(k-1)
	cv::Ptr<TrackerTargetState> lastLocation = model->getLastTargetState();
	cv::Rect lastBoundingBox( (int)lastLocation->getTargetPosition().x, (int)lastLocation->getTargetPosition().y, 
			(int)lastLocation->getTargetWidth(), (int)lastLocation->getTargetHeight() );

	// sampling new frame based on last location
	( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_DETECT );
	sampler->sampling( intImage, lastBoundingBox );
	std::vector<cv::Mat> detectSamples = sampler->getSamples();
	if( detectSamples.empty() )
		return false;

	/*//TODO debug samples
	Mat f;
	image.copyTo(f);

	for( size_t i = 0; i < detectSamples.size(); i=i+10 )
	{
		Size sz;
		Point off;
		detectSamples.at(i).locateROI(sz, off);
		rectangle(f, Rect(off.x,off.y,detectSamples.at(i).cols,detectSamples.at(i).rows), Scalar(255,0,0), 1);
	}*/

	// extract features from new samples
	featureSet->extraction( detectSamples );
	std::vector<cv::Mat> response = featureSet->getResponses();

	// predict new location
	ConfidenceMap cmap;
	model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_ESTIMATION, detectSamples );
	model.staticCast<TrackerMILModel>()->responseToConfidenceMap( response, cmap );
	model->getTrackerStateEstimator().staticCast<TrackerStateEstimatorMILBoosting>()->setCurrentConfidenceMap( cmap );
	if( !model->runStateEstimator() )
		return false;

	/*************************** step 2: Update the tracker appearance model ********************/

	// update the MIL appearance model
	cv::Ptr<TrackerTargetState> currentState = model->getLastTargetState();
	boundingBox = cv::Rect( (int)currentState->getTargetPosition().x, (int)currentState->getTargetPosition().y, 
				currentState->getTargetWidth(), currentState->getTargetHeight() );

	/*//TODO debug
	rectangle(f, lastBoundingBox, Scalar(0,255,0), 1);
	rectangle(f, boundingBox, Scalar(0,0,255), 1);
	imshow("f", f);
	//waitKey( 0 );*/

	// sampling new frame based on new location
	// positive smapling
	( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_POS );
	sampler->sampling( intImage, boundingBox );
	std::vector<cv::Mat> posSamples = sampler->getSamples();

	// negative sampling
	( sampler->getSamplers().at( 0 ).second ).staticCast<TrackerSamplerCSC>()->setMode( TrackerSamplerCSC::MODE_INIT_NEG );
	sampler->sampling( intImage, boundingBox );
	std::vector<cv::Mat> negSamples = sampler->getSamples();

	if( posSamples.empty() || negSamples.empty() )
		return false;

	// extract features
	featureSet->extraction( posSamples );
	std::vector<cv::Mat> posResponse = featureSet->getResponses();

	featureSet->extraction( negSamples );
	std::vector<cv::Mat> negResponse = featureSet->getResponses();

	// model estimation
	model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_POSITIVE, posSamples );
	model->modelEstimation( posResponse );
	model.staticCast<TrackerMILModel>()->setMode( TrackerMILModel::MODE_NEGATIVE, negSamples );
	model->modelEstimation( negResponse );

	// model update
	model->modelUpdate();

	return true;
}


} /**/