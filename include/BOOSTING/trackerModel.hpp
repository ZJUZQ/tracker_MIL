#ifndef BOOSTING_TRACKERMODEL_HPP_
#define BOOSTING_TRACKERMODEL_HPP_

#include "BOOSTING/common_includes.hpp"
#include "BOOSTING/trackerStateEstimator.hpp"

namespace BOOSTING
{

/** @brief Abstract class that represents the model of the target. It must be instantiated by specialized
tracker

Inherits this with your TrackerModel
 */
class TrackerModel{
public:
	TrackerModel();
	virtual ~TrackerModel();

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

    // evaluate the currentConfidenceMap from a list of response
    virtual void evalCurrentConfidenceMap( const std::vector<cv::Mat>& responseSet ) = 0;

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


} /* namespace BOOSTING */

#endif