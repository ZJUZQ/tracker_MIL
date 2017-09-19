#ifndef BOOSTING_DIRECTION_TRACKERFEATURE_HPP
#define BOOSTING_DIRECTION_TRACKERFEATURE_HPP

#include "BOOSTING_DIRECTION/common_includes.hpp"
#include "BOOSTING_DIRECTION/feature.hpp"

namespace BOOSTING_DIRECTION
{

/**************************** TrackerFeature ****************************8/

/** @brief Abstract base class for TrackerFeature that represents the feature.
 */
class TrackerFeature{
public:
	virtual ~TrackerFeature();

	/** @brief Compute the features in the images collection
    @param images The images
    @param response The output response
     */
	void compute( const std::vector<cv::Mat>& images, cv::Mat& response );

	/** @brief Create TrackerFeature by tracker feature type
    @param trackerFeatureType The TrackerFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D
     */
	static cv::Ptr<TrackerFeature> create( const std::string& trackerFeatureType );

	/** @brief Identify most effective features
    @param response Collection of response for the specific TrackerFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
     */
	virtual void selection( cv::Mat& response, int npoints ) = 0;

	/** @brief Get the name of the specific TrackerFeature
     */
	std::string getClassName() const;

protected:
	virtual bool computeImpl( const std::vector<cv::Mat>& images, cv::Mat& response ) = 0;

	std::string className;
};

/***************************** TrackerFeatureHAAR ****************************/

/** @brief TrackerFeature based on HAAR features, used by TrackerMIL and many others algorithms
@note HAAR features implementation is copied from apps/traincascade and modified according to MIL
 */
class TrackerFeatureHAAR : public TrackerFeature{
public:
	struct Params{
		Params();
		int numFeatures;  //!< # of rects
	    cv::Size rectSize;    //!< rect size
	    bool isIntegral;  //!< true if input images are integral, false otherwise
	};

	/** @brief Constructor
    @param parameters TrackerFeatureHAAR parameters TrackerFeatureHAAR::Params
     */
	TrackerFeatureHAAR( const TrackerFeatureHAAR::Params& parameters = TrackerFeatureHAAR::Params() );

	~TrackerFeatureHAAR();

	/** @brief Compute the features only for the selected indices in the images collection
    @param selFeatures indices of selected features
    @param images The images
    @param response Collection of response for the specific TrackerFeature
     */
	bool extractSelected( const std::vector<int> selFeatures, const std::vector<cv::Mat>& images, cv::Mat& response );

	/** @brief Identify most effective features
    @param response Collection of response for the specific TrackerFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
     */
	void selection( cv::Mat& response, int npoints );

	/** @brief Swap the feature in position source with the feature in position target
	  @param source The source position
	  @param target The target position
	 */
	bool swapFeature( int source, int target );

	/** @brief   Swap the feature in position id with the feature input
	  @param id The position
	  @param feature The feature
	 */
	bool swapFeature( int id, CvHaarEvaluator::FeatureHaar& feature );

	/** @brief Get the feature in position id
    @param id The position
     */
	CvHaarEvaluator::FeatureHaar& getFeatureAt( int id );

protected:
	bool computeImpl( const std::vector<cv::Mat>& images, cv::Mat& response );

private:
	Params params;
	cv::Ptr<CvHaarEvaluator> featureEvaluator;

};

} /* namespace BOOSTING_DIRECTION */


#endif