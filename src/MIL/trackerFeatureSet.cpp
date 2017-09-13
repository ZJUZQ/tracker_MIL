#include "MIL/trackerFeatureSet.hpp"

namespace MIL
{

/*************************** TrackerFeatureSet **********************************/
TrackerFeatureSet::TrackerFeatureSet(){
	blockAddTrackerFeature = false;
}

TrackerFeatureSet::~TrackerFeatureSet(){

}

void TrackerFeatureSet::extraction( const std::vector<cv::Mat>& images ){
	// Extract features from the images collection. 

	clearResponses(); // std::vector< std::pair< cv::String, cv::Ptr<TrackerFeature> > > features; 不同种类(HAAR, LBP, ...)的特征
	responses.resize( features.size() ); // std::vector<cv::Mat> responses

	for( size_t i = 0; i < features.size(); i++ ){
		cv::Mat response;
		features[i].second->compute( images, response );
		responses[i] = response;
	}

	if( !blockAddTrackerFeature )
		blockAddTrackerFeature = true;
}

void TrackerFeatureSet::selection(){

}

void TrackerFeatureSet::removeOutliers(){

}

bool TrackerFeatureSet::addTrackerFeature( cv::String trackerFeatureType ){
	if( blockAddTrackerFeature )
		return false;

	cv::Ptr<TrackerFeature> feature = TrackerFeature::create( trackerFeatureType );
	
	if( feature == 0 )
		return false;

	features.push_back( std::make_pair( trackerFeatureType, feature ) );

	return true;
}

bool TrackerFeatureSet::addTrackerFeature( cv::Ptr<TrackerFeature>& feature ){
	if( blockAddTrackerFeature )
		return false;

	cv::String trackerFeatureType = feature->getClassName();
	features.push_back( std::make_pair( trackerFeatureType, feature ) );

	return true;
}

const std::vector< std::pair< cv::String, cv::Ptr<TrackerFeature> > >& TrackerFeatureSet::getTrackerFeature() const{
	return features;
}

const std::vector<cv::Mat>& TrackerFeatureSet::getResponses() const{
	return responses;
}

void TrackerFeatureSet::clearResponses(){
	responses.clear();
}

} /* namespace MIL */