#ifndef BOOSTING_TRACKERBOOSTINGMODEL_HPP
#define BOOSTING_TRACKERBOOSTINGMODEL_HPP

#include "BOOSTING/common_includes.hpp"
#include "BOOSTING/trackerModel.hpp"

namespace BOOSTING
{

/***************************** TrackerBoostingModel *********************************/

/**
 * \brief Implementation of TrackerModel for BOOSTING algorithm
 */
class TrackerBoostingModel : public TrackerModel{
public:
	enum{
		MODE_POSITIVE = 1,    // mode for positive features
    	MODE_NEGATIVE = 2,    // mode for negative features
    	MODE_CLASSIFY = 3    // mode for classify step
	};

	/**
	 * \brief Constructor
	 * \param boundingBox The first boundingBox
	 */
	TrackerBoostingModel( const cv::Rect& boundingBox );

	~TrackerBoostingModel() {}

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

	/**
	* \brief return the selected weak classifiers for the detect
	* @return the selected weak classifiers
	*/
	std::vector<int> getSelectedWeakClassifier();

	// evaluate the currentConfidenceMap from a list of response
	void evalCurrentConfidenceMap( const std::vector<cv::Mat>& responseSet );

protected:
	void modelEstimationImpl( const std::vector<cv::Mat>& responses );
  	void modelUpdateImpl();

private:
	std::vector<cv::Mat> currentSamples;
	int mode;
};


} /* namespace BOOSTING */

#endif