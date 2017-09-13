#ifndef BOOSTING_TRACKERSAMPLERALGORITHM_HPP
#define BOOSTING_TRACKERSAMPLERALGORITHM_HPP

#include "BOOSTING/common_includes.hpp"

namespace BOOSTING
{

/********************** TrackerSamplerAlgorithm ************************/

/** @brief Abstract base class for TrackerSamplerAlgorithm that represents the algorithm for the specific
sampler.
 */
class TrackerSamplerAlgorithm{
public:
	virtual ~TrackerSamplerAlgorithm();

	/** @brief Create TrackerSamplerAlgorithm by tracker sampler type.
    @param trackerSamplerType The trackerSamplerType name

    The modes available now:

    -   "CSC" -- Current State Center
    -   "CS" -- Current State
     */
	static cv::Ptr<TrackerSamplerAlgorithm> create( const std::string& trackerSamplerType );

	/** @brief Computes the regions starting from a position in an image.

    Return true if samples are computed, false otherwise

    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated

    @param samples The computed samples @cite AAM Fig. 1 variable Sk
     */
	bool sampling( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& samples );

	/** @brief Get the name of the specific TrackerSamplerAlgorithm
    */
    std::string getClassName() const;

protected:
	std::string className;
	virtual bool samplingImpl( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& smaples ) = 0;

};

/************************************* TrackerSamplerCS *******************************************/

/** @brief TrackerSampler based on CS (current state), used by algorithm TrackerBoosting
 */
class TrackerSamplerCS : public TrackerSamplerAlgorithm{

public:
	enum{
		MODE_POSITIVE = 1, // mode for positive samples
		MODE_NEGATIVE = 2, // mode for negative samples
		MODE_CLASSIFY = 3 // mode for classify samples
	};

	struct Params{
		Params();
		float overlap; // overlapping for the search windows
		float searchFactor; // search region parameter
	};

	TrackerSamplerCS( const TrackerSamplerCS::Params& parameters = TrackerSamplerCS::Params() );

	/** @brief Set the sampling mode of TrackerSamplerCS
    @param samplingMode The sampling mode

    The modes are:

    -   "MODE_POSITIVE = 1" -- for the positive sampling
    -   "MODE_NEGATIVE = 2" -- for the negative sampling
    -   "MODE_CLASSIFY = 3" -- for the sampling in classification step
     */
	void setMode( int samplingMode );

	~TrackerSamplerCS();

	bool samplingImpl( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& samples );
	cv::Rect getsampleROI() const;

private:
	cv::Rect getTrackingROI( float searchFactor );
	cv::Rect RectMultiply( const cv::Rect& rect, float f );
	std::vector<cv::Mat> patchesRegularScan( const cv::Mat& image, cv::Rect trackingROI, cv::Size patchSize );
	void setsampleROI( cv::Rect imageROI );

	Params params;
	int mode;
	cv::Rect trackedPatch; // tracked boundingbox which represent the object
	cv::Rect validROI;	// the valid region in which the object may exist
	cv::Rect sampleROI; // the setted region in which the samples are sampled
};



} /* namespace BOOSTING */


#endif