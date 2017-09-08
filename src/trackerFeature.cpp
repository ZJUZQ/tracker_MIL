#include "MIL/trackerFeature.hpp"

namespace MIL
{

/********************************** TrackerFeature **********************************/

TrackerFeature::~TrackerFeature(){

}

void TrackerFeature::compute( const std::vector<cv::Mat>& images, cv::Mat& response ){
	if( images.empty() )
		return;

	computeImpl( images, response );
}

cv::Ptr<TrackerFeature> TrackerFeature::create( const std::string& trackerFeatureType ){
	/*
	if( trackerFeatureType.find( "FEATURE2D" ) == 0 )
	{
		size_t firstSep = trackerFeatureType.find_first_of( "." );
		size_t secondSep = trackerFeatureType.find_last_of( "." );

		String detector = trackerFeatureType.substr( firstSep, secondSep - firstSep );
		String descriptor = trackerFeatureType.substr( secondSep, trackerFeatureType.length() - secondSep );

		return Ptr<TrackerFeatureFeature2d>( new TrackerFeatureFeature2d( detector, descriptor ) );
	}

	if( trackerFeatureType.find( "HOG" ) == 0 )
	{
		return Ptr<TrackerFeatureHOG>( new TrackerFeatureHOG() );
	}

	if( trackerFeatureType.find( "LBP" ) == 0 )
	{
		return Ptr<TrackerFeatureLBP>( new TrackerFeatureLBP() );
	}
	*/

	if( trackerFeatureType.find( "HAAR" ) == 0 ){
		return cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR() );
	}

	cv::CV_Error( -1, "Tracker feature type not supported" );
	return cv::Ptr<TrackerFeature>();
}

std::string TrackerFeature::getClassName() const{
	return className;
}

/************************* TrackerFeatureHAAR **************************/

TrackerFeatureHAAR::Params::Params(){
	numFeatures = 250;
	rectSize = cv::Size( 100, 100 );
	isIntegral = false;
}

TrackerFeatureHAAR::TrackerFeatureHAAR( const TrackerFeatureHAAR::Params& parameters ) : params( parameters ){
	className = "HAAR";

	CvHaarFeatureParams haarParams;
	haarParams.numFeatures = params.numFeatures;
	haarParams.isIntegral = params.isIntegral;
	featureEvaluator = CvFeatureEvaluator::create( CvFeatureParams::HAAR ).staticCast<CvHaarEvaluator>();
  	featureEvaluator->init( &haarParams, 1, params.rectSize );
}

TrackerFeatureHAAR::~TrackerFeatureHAAR(){

}

CvHaarEvaluator::FeatureHaar& TrackerFeatureHAAR::getFeatureAt( int id ){   // Get the feature in position id.
  	return featureEvaluator->getFeatures( id ); // // 在feature.hpp中有定义
}

bool TrackerFeatureHAAR::swapFeature( int id, CvHaarEvaluator::FeatureHaar& feature ){
	// Swap the feature in position id with the feature input. 

	featureEvaluator->getFeatures( id ) = feature; 
	return true;
}

bool TrackerFeatureHAAR::swapFeature( int source, int target ){ // 交换两个特征
	// Swap the feature in position source with the feature in position target. 

	CvHaarEvaluator::FeatureHaar feature = featureEvaluator->getFeatures( source );
	featureEvaluator->getFeatures( source ) = featureEvaluator->getFeatures( target );
	featureEvaluator->getFeatures( target ) = feature;
	return true;
}

bool TrackerFeatureHAAR::extractSelected( const std::vector<int> selFeatures, const std::vector<cv::Mat>& images, cv::Mat& response ){
	// Compute the features only for the selected indices in the images collection. 
	/*
	  selFeatures :   indices of selected features 
	  response    :   Collection of response for the specific TrackerFeature
	*/

	if( images.empty() )
		return false;

	int numFeatures = featureEvaluator->getNumFeatures();
	int numSelFeatures = (int) selFeatures.size();

	// Size_ (_Tp _width, _Tp _height)
	response.create( cv::Size( (int) images.size(), numFeatures ), CV_32F );
	response.setTo( 0 );

	//double t = getTickCount();
  	//for each sample compute #n_feature -> put each feature (n Rect) in response
  	for( size_t i = 0; i < images.size(); i++ ){
  		int c = images[i].cols;
  		int r = images[i].rows;
  		for( int j = 0; j < numSelFeatures; j++ ){
  			float res = 0;
  			CvHaarEvaluator::FeatureHaar& feature = featureEvaluator->getFeatures( selFeatures[j] );
  			feature.eval( images[i], cv::Rect( 0, 0, c, r ), &res );	
  			response.at<float>( selFeatures[j], (int)i ) = res; // _Tp & Mat.at (int row, int col)

  		}
  	}
  	//t = ( (double) getTickCount() - t ) / getTickFrequency();
  	//std::cout << "StrongClassifierDirectSelection time " << t << std::endl;

  	return true;
}

bool TrackerFeatureHAAR::computeImpl( const std::vector<cv::Mat>& images, cv::Mat& response ){
	// compute the features in the images collection

	if( images.empty() )
		return false;

	int numFeatures = featureEvaluator->getNumFeatures();

	response = cv::Mat_<float>( cv::Size( (int)images.size(), numFeatures ) ); // size (width, height)

	std::vector<CvHaarEvaluator::FeatureHaar> f = featureEvaluator->getFeatures();

	//for each sample compute #n_feature -> put each feature (n Rect) in response
	/*
  	parallel_for_( Range( 0, (int)images.size() ), Parallel_compute( featureEvaluator, images, response ) );
	*/
	for ( size_t i = 0; i < images.size(); i++ ){
		int c = images[i].cols;
		int r = images[i].rows;

		for ( int j = 0; j < numFeatures; j++ ){
			float res = 0;
			featureEvaluator->getFeatures( j ).eval( images[i], Rect( 0, 0, c, r ), &res );
			( cv::Mat_<float>( response ) )( j, i ) = res;
		}
	}

	return true;
}

void TrackerFeatureHAAR::selection( Mat& /*response*/, int /*npoints*/){

}


} /* namespace MIL */