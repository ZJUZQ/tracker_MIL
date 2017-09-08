#include "MIL/trackerSampler.hpp"

namespace MIL
{

/************************** TrackerSampler ****************************/

TrackerSampler::TrackerSampler(){
	blockAddTrackerSampler = false;
}

TrackerSampler::~TrackerSampler(){

}

void TrackerSampler::sampling( const cv::Mat& image, cv::Rect boundingBox ){
	clearSamples();

	for( size_t i = 0; i < samplers.size(); i++ ){
		std::vector<cv::Mat> current_samples;
		samplers[i].second->sampling( image, boundingBox, current_samples );

		// push in samples all current_samples
		for( size_t j = 0; j < current_samples.size(); j++ ){
			std::vector<cv::Mat>::iterator it = samples.end();
			samples.insert( it, current_samples.at( j ) );
		}
	}

	if( !blockAddTrackerSampler )
		blockAddTrackerSampler = true;
}

bool TrackerSampler::addTrackerSamplerAlgorithm( std::string trackerSamplerAlgorithmType ){
	if( blockAddTrackerSampler )
		return false;

	cv::Ptr<TrackerSamplerAlgorithm> sampler = TrackerSamplerAlgorithm::create( trackerSamplerAlgorithmType );

	if( sampler == 0 )
		return false;

	samplers.push_back( std::make_pair( trackerSamplerAlgorithmType, sampler ) );

	return true;
}

bool TrackerSampler::addTrackerSamplerAlgorithm( cv::Ptr<TrackerSamplerAlgorithm>& sampler ){
	if( blockAddTrackerSampler )
		return false;

	if( sampler == 0 )
		return false;

	std::string trackerSamplerAlgorithmType = sampler->getClassName();
	samplers.push_back( std::make_pair( trackerSamplerAlgorithmType, sampler ) );

	return true;
}

const std::vector< std::pair< std::string, cv::Ptr<TrackerSamplerAlgorithm> > >& TrackerSampler::getSamplers() const{
	return samplers;
}

const std::vector<cv::Mat>& TrackerSampler::getSamples() const{
	return samples;
}

void TrackerSampler::clearSamples(){
	samples.clear();
}


} /* namespace MIL */