#ifndef MIL_TRACKERMILMODEL_H
#define MIL_TRACKERMILMODEL_H

#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>

namespace MIL
{

/**
 * \brief Implementation of TrackerModel for MIL algorithm
 */
class TrackerMILModel : public TrackerModel{
public:
	enum{
		MODE_POSITIVE = 1; // mode for positive features
		MODE_NEGATIVE = 2; // mode for negtive features
		MODE_ESTIMATION = 3; // mode for estimation step 
	};

	/**
   * \brief Constructor
   * \param boundingBox The first boundingBox
   */
	TrackerModel( const cv::Rect& boundingBox );

	~TrackerModel(){}

	/**
   * \brief Set the mode
   */
	void setMode( int trainingMode, const std::vector<cv::Mat>& samples );

	/**
   * \brief Create the ConfidenceMap from a list of responses
   * \param responses The list of the responses
   * \param confidenceMap The output
   */
	void responseToConfidenceMap( const std::vector<cv::Mat>& responses, ConfidenceMap& confidenceMap );

protected:
	void modelEstimationImpl( const std::vector<cv::Mat>& responses );
	void modelUpdateImpl();

private:
	int mode;
	std::vector<cv::Mat> currentSample;

	int width;	// initial width of the boundingBox
	int height;	// initial height of the boundingBox
};

} /* namespace MIL */

#endif