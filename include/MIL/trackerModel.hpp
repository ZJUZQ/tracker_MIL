#ifndef MIL_TRACKERMODEL_HPP_
#define MIL_TRACKERMODEL_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>

#include "MIL/feature.hpp"
#include "MIL/trackerTargetState.hpp"

#include <iostream>
#include <vector>

namespace MIL
{

/** @brief Abstract class that represents the model of the target. It must be instantiated by specialized
tracker

Inherits this with your TrackerModel
 */
class TrackerModle{
public:
	TrackerModle();
	virtual ~TrackerModle();

	/** @brief Set TrackerEstimator, return true if the tracker state estimator is added, false otherwise

	    @param trackerStateEstimator The TrackerStateEstimator
	    @note You can add only one TrackerStateEstimator
     */
	bool setTrackerStateEstimator( cv::Ptr<TrackerStateEstimator> trackerStateEstimator );

	/** @brief Estimate the most likely target location

	    @cite AAM ME, Model Estimation table I
	    @param responses Features extracted from TrackerFeatureSet
     */
	void modelEstimation( const std::vector<cv::Mat>& responses );

	/** @brief Update the model

    	@cite AAM MU, Model Update table I
     */
	void modelUpdate();

	/** @brief Run the TrackerStateEstimator, return true if is possible to estimate a new state, false otherwise
    */
    bool runStateEstimator();

    /** @brief Set the current TrackerTargetState in the Trajectory
    	@param lastTargetState The current TrackerTargetState
     */
    void setLastTargetState( const cv::Ptr<TrackerTargetState>& lastTargetState );

    /** @brief Get the last TrackerTargetState from Trajectory
    */
    cv::Ptr<TrackerTargetState> getLastTargetState() const;

    /** @brief Get the list of the ConfidenceMap
    */
    const std::vector<ConfidenceMap>& getConfidenceMaps() const;

    /** @brief Get the last ConfidenceMap for the current frame
     */
    const ConfidenceMap& getLastConfidenceMap() const;

    /** @brief Get the TrackerStateEstimator
    */
    cv::Ptr<TrackerStateEstimator> getTrackerStateEstimator() const;

private:
	void clearCurrentConfidenceMap();

protected:
	std::vector<ConfidenceMap> confidenceMaps;
	cv::Ptr<TrackerStateEstimator> stateEstimator;
	ConfidenceMap currentConfidenceMap;
	Trajectory trajectory;
	int maxCMLength;

	virtual void modelEstimationImpl( const std::vector<cv::Mat>& responses ) = 0;
	virtual void modelUpdateImpl() = 0;

};


} /* namespace MIL */

#endif