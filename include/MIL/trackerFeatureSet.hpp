#ifndef MIL_TRACKERFEATURESET_HPP
#define MIL_TRACKERFEATURESET_HPP

#include "MIL/common_includes.hpp"
#include "MIL/trackerFeature.hpp"

namespace MIL
{

/*************************** TrackerFeatureSet ******************************/

/** @brief Class that manages the extraction and selection of features

@cite AAM Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection).
See table I and section III C @cite AMVOT Appearance modelling -\> Visual representation (Table II,
section 3.1 - 3.2)

TrackerFeatureSet is an aggregation of TrackerFeature

@sa
   TrackerFeature

 */
class TrackerFeatureSet{
public:
	TrackerFeatureSet();
	~TrackerFeatureSet();

	/** @brief Extract features from the images collection
    @param images The input images
     */
	void extraction( const std::vector<cv::Mat>& images );

	/** @brief Identify most effective features for all feature types (optional)
     */
	void selection();

	/** @brief Remove outliers for all feature types (optional)
     */
	void removeOutliers();

	/** @brief Add TrackerFeature in the collection. Return true if TrackerFeature is added, false otherwise
    @param trackerFeatureType The TrackerFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D

    Example TrackerFeatureSet::addTrackerFeature : :
    @code
        //sample usage:

        Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
        featureSet->addTrackerFeature( trackerFeature );

        //or add CSC sampler with default parameters
        //featureSet->addTrackerFeature( "HAAR" );
    @endcode
    @note If you use the second method, you must initialize the TrackerFeature
     */
	bool addTrackerFeature( cv::String trackerFeatureType );

	/** @overload
    @param feature The TrackerFeature class
    */
    bool addTrackerFeature( cv::Ptr<TrackerFeature>& feature );

    /** @brief Get the TrackerFeature collection (TrackerFeature name, TrackerFeature pointer)
     */
    const std::vector< std::pair< cv::String, cv::Ptr<TrackerFeature> > >& getTrackerFeature() const;

    /** @brief Get the responses

    @note Be sure to call extraction before getResponses Example TrackerFeatureSet::getResponses : :
     */
    const std::vector<cv::Mat>& getResponses() const;

private:
	void clearResponses();
	bool blockAddTrackerFeature;

	std::vector< std::pair< cv::String, cv::Ptr<TrackerFeature> > > features;
	std::vector<cv::Mat> responses;	// list of responses after compute

};

} /* namespace MIL */



#endif