#ifndef MIL_TRACKERMIL_HPP
#define MIL_TRACKERMIL_HPP

#include "MIL/common_includes.hpp"

namespace MIL
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

	virtual void read( const cv::FileNode& fn )=0;
	virtual void write( cv::FileStorage& fs ) const=0;

protected:

	virtual bool initImpl( const cv::Mat& image, const cv::Rect2d& boundingBox ) = 0;
	virtual bool updateImpl( const cv::Mat& image, cv::Rect2d& boundingBox ) = 0;

	bool isInit;

	cv::Ptr<TrackerFeatureSet> featureSet;
	cv::Ptr<TrackerSampler> sampler;
	cv::Ptr<TrackerModel> model;
};

/********************** TrackerMIL : Specific Tracker Classes **********************/

/** @brief The MIL algorithm trains a classifier in an online manner to separate the object from the
background.

Multiple Instance Learning avoids the drift problem for a robust tracking. The implementation is
based on @cite MIL .

Original code can be found here <http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml>
 */
class TrackerMIL : public Tracker{
public:
	struct Params{
		Params();
		// parameters for sampler
		float samplerInitInRadius;  //!< radius for gathering positive instances during init
	    int samplerInitMaxNegNum;  //!< # negative samples to use during init
	    float samplerSearchWinSize;  //!< size of search window
	    float samplerTrackInRadius;  //!< radius for gathering positive instances during tracking
	    int samplerTrackMaxPosNum;  //!< # positive samples to use during tracking
	    int samplerTrackMaxNegNum;  //!< # negative samples to use during tracking
	    int featureSetNumFeatures;  //!< # features

	    void read( const cv::FileNode& fn );
	    void write( cv::FileStorage& fs ) const;
	};

	/** @brief Constructor
    @param parameters MIL parameters TrackerMIL::Params
     */
	static cv::Ptr<TrackerMIL> create( const TrackerMIL::Params& parameters );

	static cv::Ptr<TrackerMIL> create();

	virtual ~TrackerMIL() {}

};

/********************************** TrackerMILImpl ***********************************/
class TrackerMILImpl : public TrackerMIL{
public:
	TrackerMILImpl( const TrackerMIL::Params& parameters = TrackerMIL::Params() );
	void read( const cv::FileNode& fn );
  	void write( cv::FileStorage& fs ) const;

 protected:

	bool initImpl( const cv::Mat& image, const cv::Rect2d& boundingBox );
	bool updateImpl( const cv::Mat& image, cv::Rect2d& boundingBox );
	void compute_integral( const cv::Mat& img, cv::Mat& ii_img );

  	TrackerMIL::Params params;
};


} /* namespace MIL */


#endif