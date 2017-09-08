#ifndef MIL_TRACKERSAMPLERALGORITHM_HPP
#define MIL_TRACKERSAMPLERALGORITHM_HPP

#include "MIL/common_includes.hpp"

namespace MIL
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

/**************************** TrackerSamplerCSC *******************************/

class TrackerSamplerCSC : public TrackerSamplerAlgorithm{
public:
	enum{
		MODE_INIT_POS = 1,  //!< mode for init positive samples
	    MODE_INIT_NEG = 2,  //!< mode for init negative samples
	    MODE_TRACK_POS = 3,  //!< mode for update positive samples
	    MODE_TRACK_NEG = 4,  //!< mode for update negative samples
	    MODE_DETECT = 5   //!< mode for detect samples
	};

	struct Params{
		Params();
		float initInRad;        //!< radius for gathering positive instances during init
		float trackInPosRad;    //!< radius for gathering positive instances during tracking
		float searchWinSize;  //!< size of search window
		int initMaxNegNum;      //!< # negative samples to use during init
		int trackMaxPosNum;     //!< # positive samples to use during training
		int trackMaxNegNum;     //!< # negative samples to use during training
	};

	/** @brief Constructor
    @param parameters TrackerSamplerCSC parameters TrackerSamplerCSC::Params
     */
	TrackerSamplerCSC( const TrackerSamplerCSC::Params& parameters = TrackerSamplerCSC::Params() );

	/** @brief Set the sampling mode of TrackerSamplerCSC
    @param samplingMode The sampling mode

    The modes are:

    -   "MODE_INIT_POS = 1" -- for the positive sampling in initialization step
    -   "MODE_INIT_NEG = 2" -- for the negative sampling in initialization step
    -   "MODE_TRACK_POS = 3" -- for the positive sampling in update step
    -   "MODE_TRACK_NEG = 4" -- for the negative sampling in update step
    -   "MODE_DETECT = 5" -- for the sampling in detection step
     */
	void setMode( int samplingMode );

	~TrackerSamplerCSC();

protected:
	bool samplingImpl( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& samples );

private:
	Params params;
	int mode;
	cv::RNG rng;	// Random Number Generator

	std::vector<cv::Mat> sampleImage( const cv::Mat& img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000 );	
};


} /* namespace MIL */


#endif