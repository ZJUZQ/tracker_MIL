#include "MIL/trackerMILModel.hpp"

/* no finish */

namespace MIL
{

TrackerMILModel::TrackerMILModel( const cv::Rect& boundingBox ){
	currentSample.clear();
	mode = MODE_POSITIVE;
	width = boundingBox.width;
	height = boundingBox.height;

	cv::Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState> initState = cv::Ptr<TrackerStateEstimatorMILBoosting::TrackerMILTargetState>(
		new TrackerStateEstimatorMILBoosting::TrackerMILTargetState( cv::Point2f( (float)boundingBox.x, (float)boundingBox.y ), boundingBox.width, boundingBox.height, true, cv::Mat() ));

	trajectory.push_back(  initState );
}


} /* namespace MIL */