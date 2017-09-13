#ifndef BOOSTING_TRACKERSTATEESTIMATOR_HPP
#define BOOSTING_TRACKERSTATEESTIMATOR_HPP

#include "BOOSTING/common_includes.hpp"
#include "BOOSTING/trackerTargetState.hpp"
#include "BOOSTING/trackerAdaBoostingClassifier.hpp"

namespace BOOSTING
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


/******************************** TrackerStateEstimatorAdaBoosting **********************************/

/** @brief TrackerStateEstimatorAdaBoosting based on ADA-Boosting
 */
class TrackerStateEstimatorAdaBoosting : public TrackerStateEstimator{
public:

	/** @brief Implementation of the target state for TrackerAdaBoostingTargetState
    */
    class TrackerAdaBoostingTargetState : public TrackerTargetState{

    public:
    	/**
		* \brief Constructor
		* \param position Top left corner of the bounding box
		* \param width Width of the bounding box
		* \param height Height of the bounding box
		* \param foreground label for target or background
		* \param responses list of features
		*/
		TrackerAdaBoostingTargetState( const cv::Point2f& position, int width, int height, bool foreground, const cv::Mat& responses );

		~TrackerAdaBoostingTargetState() {}

		/** @brief Set the features extracted from TrackerFeatureSet
		@param responses The features extracted
		*/
		void setTargetResponses( const cv::Mat& responses );

		/** @brief Set label: true for target foreground, false for background
		@param foreground Label for background/foreground
		*/
		void setTargetFg( bool foreground );

		/** @brief Get the features extracted
		*/
		cv::Mat getTargetResponses() const;

		/** @brief Get the label. Return true for target foreground, false for background
		*/
		bool isTargetFg() const;

	private:
		bool isTarget;
		cv::Mat targetResponses;
    };

	/** @brief Constructor
		@param numClassifer Number of base classifiers
		@param initIterations Number of iterations in the initialization
		@param nFeatures Number of features/weak classifiers
		@param patchSize tracking rect
		@param ROI initial ROI
	*/
	TrackerStateEstimatorAdaBoosting( int numClassifer, int initIterations, int nFeatures, cv::Size patchSize, const cv::Rect& ROI );
	TrackerStateEstimatorAdaBoosting();
	/**
	* \brief Destructor
	*/
	~TrackerStateEstimatorAdaBoosting();

	/** @brief Get the sampling ROI
	*/
	cv::Rect getSampleROI() const;

	/** @brief Set the sampling ROI
	@param ROI the sampling ROI
	*/
	void setSampleROI( const cv::Rect& ROI );

	/** @brief Set the current confidenceMap
	@param confidenceMap The current :cConfidenceMap
	*/
	void setCurrentConfidenceMap( ConfidenceMap& confidenceMap );

	/** @brief Get the list of the selected weak classifiers for the classification step
	*/
	std::vector<int> computeSelectedWeakClassifier();

	/** @brief Get the list of the weak classifiers that should be replaced
	*/
	std::vector<int> computeReplacedClassifier();

	/** @brief Get the list of the weak classifiers that replace those to be replaced
	*/
	std::vector<int> computeSwappedClassifier();

protected:
	cv::Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps );
	void updateImpl( std::vector<ConfidenceMap>& confidenceMaps );

	cv::Ptr<StrongClassifierDirectSelection> boostClassifier;

private:
	int numBaseClassifier;
	int iterationInit;
	int numFeatures;
	bool trained;
	cv::Size initPatchSize;
	cv::Rect sampleROI;
	std::vector<int> replacedClassifier;
	std::vector<int> swappedClassifier;

	ConfidenceMap currentConfidenceMap;
};



} /* namespace BOOSTING */

#endif