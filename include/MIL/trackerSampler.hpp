#ifndef MIL_TRACKERSAMPLER_HPP
#define MIL_TRACKERSAMPLER_HPP

#include "MIL/common_includes.hpp"
#include "MIL/trackerSamplerAlgorithm.hpp"

namespace MIL
{

/************************** TrackerSampler ***************************/

/** @brief Class that manages the sampler in order to select regions for the update the model of the tracker

@cite AAM Sampling e Labeling. See table I and section III B

TrackerSampler is an aggregation of TrackerSamplerAlgorithm
 */
class TrackerSampler{
public:
	TrackerSampler();
	~TrackerSampler();

	/** @brief Computes the regions starting from a position in an image
    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated
     */
	void sampling( const cv::Mat& image, cv::Rect boundingBox );

	/** @brief Return the collection of the TrackerSamplerAlgorithm
    */
    const std::vector< std::pair< std::string, cv::Ptr<TrackerSamplerAlgorithm> > >& getSamplers() const;

    /** @brief Return the samples from all TrackerSamplerAlgorithm, @cite AAM Fig. 1 variable Sk
    */
    const std::vector<cv::Mat>& getSamples() const;

    /** @brief Add TrackerSamplerAlgorithm in the collection. Return true if sampler is added, false otherwise
    @param trackerSamplerAlgorithmType The TrackerSamplerAlgorithm name

    The modes available now:
    -   "CSC" -- Current State Center
    -   "CS" -- Current State
    -   "PF" -- Particle Filtering

    Example TrackerSamplerAlgorithm::addTrackerSamplerAlgorithm : :
    @code
         TrackerSamplerCSC::Params CSCparameters;
         Ptr<TrackerSamplerAlgorithm> CSCSampler = new TrackerSamplerCSC( CSCparameters );

         if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
           return false;

         //or add CSC sampler with default parameters
         //sampler->addTrackerSamplerAlgorithm( "CSC" );
    @endcode
    @note If you use the second method, you must initialize the TrackerSamplerAlgorithm
     */
    bool addTrackerSamplerAlgorithm( std::string trackerSamplerAlgorithmType );

    /** @overload
    @param sampler The TrackerSamplerAlgorithm
    */
    bool addTrackerSamplerAlgorithm( cv::Ptr<TrackerSamplerAlgorithm>& sampler );

private:
	std::vector< std::pair< std::string, cv::Ptr<TrackerSamplerAlgorithm> > > samplers;
	std::vector<cv::Mat> samples;
	bool blockAddTrackerSampler;

	void clearSamples();
};



} /* namespace MIL */


#endif