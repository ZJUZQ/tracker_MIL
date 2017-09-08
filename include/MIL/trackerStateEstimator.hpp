#ifndef MIL_TRACKERSTATEESTIMATOR_HPP
#define MIL_TRACKERSTATEESTIMATOR_HPP

#include "MIL/common_includes.hpp"
#include "MIL/trackerTargetState.hpp"

namespace MIL
{

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


} /* namespace MIL */

#endif