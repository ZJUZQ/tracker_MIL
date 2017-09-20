#ifndef BOOSTING_TRACKERBOOSTING_HPP
#define BOOSTING_TRACKERBOOSTING_HPP

#include "BOOSTING/common_includes.hpp"

#include "BOOSTING/trackerFeatureSet.hpp"
#include "BOOSTING/trackerSampler.hpp"
#include "BOOSTING/trackerAdaBoostingModel.hpp"

namespace BOOSTING
{

/********************************* Tracker Base Class ***************************/

/** @brief Base abstract class for the long-term tracker:
 */
class Tracker : public virtual cv::Algorithm{  // 虚基类，即只对此基类生成一块内存区域，这样最终派生类中就只会含有一个基类
	// cv::Algorithm <-- #include "core.hpp"

public:
	virtual ~Tracker();

	/** @brief Initialize the tracker with a know bounding box that surrounding the target
    @param image The initial frame
    @param boundingBox The initial boundig box

    @return True if initialization went succesfully, false otherwise
     */
	bool init( cv::InputArray image, const cv::Rect2d& boundingBox );

	/** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The boundig box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
     */
	bool update( cv::InputArray image, cv::Rect2d& boundingBox );

	/* Add by me: just estimate the new most likely boundingbox, without update the model */
	bool estimateOnly( cv::InputArray image, cv::Rect2d& boundingBox );

	// Add by me: update the strong classifier with a given sample which has class label
	bool updateWithSample( cv::InputArray sample, int labelFg = 1 );

	virtual void read( const cv::FileNode& fn )=0;
	virtual void write( cv::FileStorage& fs ) const=0;

protected:

	virtual bool initImpl( const cv::Mat& image, const cv::Rect2d& boundingBox ) = 0;
	virtual bool updateImpl( const cv::Mat& image, cv::Rect2d& boundingBox ) = 0;

	/* Add by me: just estimate the new most likely boundingbox, without update the model */
	virtual bool estimateOnlyImpl( cv::InputArray image, cv::Rect2d& boundingBox ) = 0;

	// Add by me: update the strong classifier with a given sample which has class label
	virtual bool updateWithSampleImpl( cv::InputArray sample, int labelFg ) = 0;

	bool isInit;

	cv::Ptr<TrackerFeatureSet> featureSet;
	cv::Ptr<TrackerSampler> sampler;
	cv::Ptr<TrackerModel> model;
};

/************************* TrackerBoosting *************************/

/** @brief This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.

The classifier uses the surrounding background as negative examples in update step to avoid the
drifting problem. The implementation is based on @cite OLB .
 */

class TrackerBoosting : public Tracker{

public:
	struct Params{
		Params();
		int numBaseClassifiers; // the number of base classifiers to use in a OnlineBoosting algorithm
		int iterationInit; // the initial iterations
		int featureSetNumFeatures; // features, feature pool

		float samplerOverlap; // search region paramters to use in a OnlineBoosting algorithm
		float samplerSearchFactor; // search region parameters

		/**
		* \brief Read parameters from file
		*/
		void read( const cv::FileNode& fn );

		/**
		* \brief Write parameters in a file
		*/
		void write( cv::FileStorage& fs ) const;
	};

	/** @brief Constructor
    @param parameters BOOSTING parameters TrackerBoosting::Params
     */
	static cv::Ptr<TrackerBoosting> create( const TrackerBoosting::Params& parameters );

	static cv::Ptr<TrackerBoosting> create();	// 静态成员函数

	virtual ~TrackerBoosting() {}
};


} /* namespace BOOSTING */


#endif