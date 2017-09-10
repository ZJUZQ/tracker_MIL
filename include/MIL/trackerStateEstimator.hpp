#ifndef MIL_TRACKERSTATEESTIMATOR_HPP
#define MIL_TRACKERSTATEESTIMATOR_HPP

#include "MIL/common_includes.hpp"
#include "MIL/trackerTargetState.hpp"
// #include "MIL/trackerStateEstimatorMILBoosting.hpp"
#include "MIL/trackerMILClassifier.hpp"

namespace MIL
{
/********************************* TrackerStateEstimator ***************************/

//class TrackerStateEstimatorMILBoosting; //前向声明

/** @brief Abstract base class for TrackerStateEstimator that estimates the most likely target state.

See @cite AAM State estimator

See @cite AMVOT Statistical modeling (Fig. 3), Table III (generative) - IV (discriminative) - V (hybrid)
 */
class TrackerStateEstimator{
public:
	virtual ~TrackerStateEstimator();

	/** @brief Estimate the most likely target state, return the estimated state
    @param confidenceMaps The overall appearance model as a list of :cConfidenceMap
     */
	cv::Ptr<TrackerTargetState> estimate( const std::vector<ConfidenceMap>& confidenceMaps );

	/** @brief Update the ConfidenceMap with the scores
    @param confidenceMaps The overall appearance model as a list of :cConfidenceMap
     */
	void update( std::vector<ConfidenceMap>& confidenceMaps );

	/** @brief Create TrackerStateEstimator by tracker state estimator type
    @param trackeStateEstimatorType The TrackerStateEstimator name

    The modes available now:

    -   "BOOSTING" -- Boosting-based discriminative appearance models. See @cite AMVOT section 4.4

    The modes available soon:

    -   "SVM" -- SVM-based discriminative appearance models. See @cite AMVOT section 4.5
     */
	static cv::Ptr<TrackerStateEstimator> create( const std::string& trackerStateEstimatorType );

	/** @brief Get the name of the specific TrackerStateEstimator
     */
	std::string getClassName() const;

protected:
	virtual cv::Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps ) = 0;
	virtual void updateImpl( std::vector<ConfidenceMap>& confidenceMaps ) = 0;
	std::string className;
};


/******************************** TrackerStateEstimatorMILBoosting ***************************/

/** @brief TrackerStateEstimator based on Boosting
  */
class TrackerStateEstimatorMILBoosting : public TrackerStateEstimator{
public:
	/**
	 * Implementation of the target state for TrackerStateEstimatorMILBoosting
   	 */
	class TrackerMILTargetState : public TrackerTargetState{
	public:
		/**
	     * \brief Constructor
	     * \param position Top left corner of the bounding box
	     * \param width Width of the bounding box
	     * \param height Height of the bounding box
	     * \param foreground label for target or background
	     * \param features features extracted
	     */
		TrackerMILTargetState( const cv::Point2f& position, int width, int height, bool foreground, const cv::Mat& features );

		~TrackerMILTargetState(){}

		/** @brief Set label: true for target foreground, false for background
	    @param foreground Label for background/foreground
	     */
		void setTargetFg( bool foreground ); // // Set label: true for target foreground, false for background

		/** @brief Set the features extracted from TrackerFeatureSet
	    @param features The features extracted
	     */
		void setFeatures( const cv::Mat& features );

		/** @brief Get the label. Return true for target foreground, false for background
     	*/
     	bool isTargetFg() const;

     	/** @brief Get the features extracted
     	*/
     	cv::Mat getFeatures() const;

     private:
     	bool isTarget;
     	cv::Mat targetFeatures;
	};

	/** @brief Constructor
    @param nFeatures Number of features for each sample
     */
	TrackerStateEstimatorMILBoosting( int nFeatures = 250 );
	~TrackerStateEstimatorMILBoosting();

	/** @brief Set the current confidenceMap
    @param confidenceMap The current :cConfidenceMap
     */
	void setCurrentConfidenceMap( ConfidenceMap& confidenceMap );

protected:
	cv::Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps );
	void updateImpl( std::vector<ConfidenceMap>& confidenceMaps );

private:
	uint max_idx( const std::vector<float>& v );
	void prepareData( const ConfidenceMap& confidenceMap, cv::Mat& positive, cv::Mat& negative );

	ClfMilBoost boostMILModel;
	bool trained;
	int numFeatures;

	ConfidenceMap currentConfidenceMap;

};


} /* namespace MIL */

#endif