#ifndef BOOSTING_DIRECTION_DIRECTIONBOOSTING_HPP
#define BOOSTING_DIRECTION_DIRECTIONBOOSTING_HPP

#include "BOOSTING_DIRECTION/common_includes.hpp"
#include "BOOSTING_DIRECTION/trackerAdaBoostingClassifier.hpp"
#include "BOOSTING_DIRECTION/trackerFeature.hpp"

namespace BOOSTING_DIRECTION{

/** 
	directionBoosting 分类器 : 判断图片中目标的方向：up or down
 */
class directionAdaBoosting : public virtual cv::Algorithm  // 虚基类，即只对此基类生成一块内存区域，这样最终派生类中就只会含有一个基类
{ 
public:
	directionAdaBoosting();

	~directionAdaBoosting();

	struct Params{
		Params();
		Params( int numBaseClfs_, int numWeakClfs_, int numAllWeakClfs_, cv::Size patchSize_, bool useFeatureExchange_ );
		int numBaseClfs; // number of base classifiers
		int numWeakClfs; // number of weak classifiers
		int numAllWeakClfs; // 多出来的是备份weak classifiers，用于替换前面表现不好的weak classfier

		cv::Size patchSize; // the object's size
		bool useFeatureExchange;
	};

	/** Initialize the direction boosting classifier with a know bounding box that surrounding the target
     */
	bool init( const cv::Mat& imageT, const cv::Rect2d& objectBB, Params params_ = Params() );

	/** Given a new object image with class label, update the strong classifier	
	*/
	bool updateWithOneSample( const cv::Mat& imgObject, const int labelUp );


	/** compute the given sample's class: 1 for down and -1 for up
	 */
	int classifierSample( const cv::Mat& sample);

	/** compute the given sample's confidence: poistive for down and negtive for up
	 */
	float evalSample( const cv::Mat& sample);

	void read( const cv::FileNode& fn ) {}
	void write( cv::FileStorage& fs ) const {}

private:
	bool isInit;
	Params params;

	cv::Ptr<StrongClassifierDirectSelection> strongClassifier;
	cv::Ptr<TrackerFeatureHAAR> trackerFeature;
};

}; /* namespace BOOSTING_DIRECTION */

#endif
