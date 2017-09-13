#ifndef BOOSTING_TRACKERTARGETSTATE_HPP
#define BOOSTING_TRACKERTARGETSTATE_HPP

#include "BOOSTING/common_includes.hpp"

namespace BOOSTING
{

class TrackerTargetState{
public:
	virtual ~TrackerTargetState(){}

	/**
	 * \brief Get the position
	 * \return The position
	 */
	cv::Point2f getTargetPosition() const{
		return targetPosition;
	}

	/**
	 * \brief Set the position
	 * \param position The position
	 */
	void setTargetPosition( const cv::Point2f& position ){
		targetPosition = position;
	}

	/**
	 * \brief Get the width of the target
	 * \return The width of the target
	 */
	int getTargetWidth() const{
		return targetWidth;
	}

	void setTargetWidth( int width ){
		targetWidth = width;
	}

	int getTargetHeight() const{
		return targetHeight;
	}

	void setTargetHeight( int height ){
		targetHeight = height;
	}

protected:
	cv::Point2f targetPosition;
	int targetWidth;
	int targetHeight;
};

/** @brief Represents the model of the target at frame \f$k\f$ (all states and scores)

See @cite AAM The set of the pair \f$\langle \hat{x}^{i}_{k}, C^{i}_{k} \rangle\f$
@sa TrackerTargetState
 */
typedef std::vector< std::pair<cv::Ptr<TrackerTargetState>, float> > ConfidenceMap;

/** @brief Represents the estimate states for all frames

@cite AAM \f$x_{k}\f$ is the trajectory of the target up to time \f$k\f$

@sa TrackerTargetState
 */
typedef std::vector< cv::Ptr<TrackerTargetState> > Trajectory;


} /* namespace BOOSTING */

#endif