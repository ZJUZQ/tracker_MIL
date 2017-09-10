#ifndef MIL_TRACKERSTATEESTIMATIONMILBOOSTING_HH
#define MIL_TRACKERSTATEESTIMATIONMILBOOSTING_HH

#include "MIL/common_includes.hpp"
#include "MIL/trackerStateEstimator.hpp"
#include "MIL/trackerMILClassifier.hpp"

namespace MIL
{

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
		TrackerMILTargetState( const cv::Point2f position, int width, int height, bool foreground, const cv::Mat& features );

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