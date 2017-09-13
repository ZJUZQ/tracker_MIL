#include "BOOSTING/trackerSamplerAlgorithm.hpp"
#include <time.h>

namespace BOOSTING
{

/************* TrackerSamplerAlgorithm **********************/
 
TrackerSamplerAlgorithm::~TrackerSamplerAlgorithm(){

}

bool TrackerSamplerAlgorithm::sampling( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& samples ){
	if( image.empty() )
		return false;
	return samplingImpl( image, boundingBox, samples );
}

cv::Ptr<TrackerSamplerAlgorithm> TrackerSamplerAlgorithm::create( const std::string& trackerSamplerType ){
	/**
	if( trackerSamplerType.find( "CSC" ) == 0 )
		return cv::Ptr<TrackerSamplerCSC>( new TrackerSamplerCSC() );
	*/
	
	if( trackerSamplerType.find( "CS" ) == 0 )
	{
		return cv::Ptr<TrackerSamplerCS>( new TrackerSamplerCS() );
	}
	

	CV_Error( -1, "Tracker samper algorithm type not supported" );
	return cv::Ptr<TrackerSamplerAlgorithm>();
}

std::string TrackerSamplerAlgorithm::getClassName() const{
	return className;
}


/********************************* TrackerSamplerCS *************************************/

TrackerSamplerCS::Params::Params(){
	overlap = 0.99f;
	searchFactor = 2;
}

TrackerSamplerCS::TrackerSamplerCS( const TrackerSamplerCS::Params &parameters ) : params( parameters ){
	className = "CS";
	mode = MODE_POSITIVE;
}

void TrackerSamplerCS::setMode( int samplingMode ){
  	mode = samplingMode;
}

TrackerSamplerCS::~TrackerSamplerCS(){

}

bool TrackerSamplerCS::samplingImpl( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& samples ){

	trackedPatch = boundingBox;

	validROI = cv::Rect( 0, 0, image.cols, image.rows );

	cv::Rect trackingROI = getTrackingROI( params.searchFactor ); // 返回扩大searchFactor倍并且处于validROI有效范围内的searchRegion

	cv::Size trackedPatchSize( trackedPatch.width, trackedPatch.height );
	samples = patchesRegularScan( image, trackingROI, trackedPatchSize );

	return true;
}

cv::Rect TrackerSamplerCS::getTrackingROI( float searchFactor ){
	// 返回扩大searchFactor倍并且处于validROI有效范围内的searchRegion

	cv::Rect searchRegion;

	searchRegion = RectMultiply( trackedPatch, searchFactor ); // 返回向四周扩大searchFactor倍后的ROI

	/**
	// check
	if( searchRegion.y + searchRegion.height > validROI.height - 1 )
		searchRegion.height = validROI.height - searchRegion.y;
	if( searchRegion.x + searchRegion.width > validROI.width - 1 )
		searchRegion.width = validROI.width - searchRegion.x;
	 */

	// check
	searchRegion = searchRegion & validROI; // rect = rect1 & rect2 (rectangle intersection)

	return searchRegion;
}

cv::Rect TrackerSamplerCS::RectMultiply( const cv::Rect& rect, float f ){
  	// 返回向四周扩大searchFactor倍后的ROI

	cv::Rect r_tmp;
	r_tmp.y = (int) ( rect.y - ( (float) rect.height * f - rect.height ) / 2 );
	if( r_tmp.y < 0 )
		r_tmp.y = 0;
	r_tmp.x = (int) ( rect.x - ( (float) rect.width * f - rect.width ) / 2 );
	if( r_tmp.x < 0 )
		r_tmp.x = 0;
	r_tmp.height = (int) ( rect.height * f );
	r_tmp.width = (int) ( rect.width * f );

	return r_tmp;
}

cv::Rect TrackerSamplerCS::getsampleROI() const{
	return sampleROI;
}

void TrackerSamplerCS::setsampleROI( cv::Rect imageROI ){
	// 计算两个boundingbox的交集 --> imageROI & validROI

	sampleROI.y = imageROI.y < validROI.y ? validROI.y : imageROI.y;
	sampleROI.x = imageROI.x < validROI.x ? validROI.x : imageROI.x;

	sampleROI.height = ( imageROI.y + imageROI.height < validROI.y + validROI.height ) ? imageROI.y + imageROI.height - sampleROI.y : validROI.y + validROI.height - sampleROI.y;
	sampleROI.width = ( imageROI.x + imageROI.width < validROI.x + validROI.width ) ? imageROI.x + imageROI.width - sampleROI.x : validROI.x + validROI.width - sampleROI.x;
}

std::vector<cv::Mat> TrackerSamplerCS::patchesRegularScan( const cv::Mat& image, cv::Rect trackingROI, cv::Size patchSize ){
	// 根据采样模式mode，在trackingROI上移动trackedPatch，返回样本roi子图片

	std::vector<cv::Mat> samples;

	setsampleROI( trackingROI );  //设置sampleROI --> vaildROI和trackingROI的交集

	if( mode == MODE_POSITIVE ){
		int num = 4;
		samples.resize( 4 );
		for( int i = 0; i < num; i++ )
			samples[i] = image( trackedPatch );
		return samples;
	}

	int stepCol = (int) std::floor( ( 1.0f - params.overlap ) * (float)patchSize.width + 0.5f );
	int stepRow = (int) std::floor( ( 1.0f - params.overlap ) * (float)patchSize.height + 0.5f );
	stepCol = stepCol <= 0 ? 1 : stepCol;
	stepRow = stepRow <= 0 ? 1 : stepRow;

	cv::Size m_patchGrid;
	m_patchGrid.height = (int)( (float)( sampleROI.height - patchSize.height ) / stepRow + 1 );
	m_patchGrid.width = (int)( (float)( sampleROI.width - patchSize.width ) / stepCol + 1 );

	int num = m_patchGrid.height * m_patchGrid.width;
	samples.resize( num );

	cv::Rect m_rectUpperLeft;
	cv::Rect m_rectUpperRight;
	cv::Rect m_rectLowerLeft;
	cv::Rect m_rectLowerRight;

	m_rectUpperLeft = m_rectUpperRight = m_rectLowerLeft = m_rectLowerRight = cv::Rect( 0, 0, patchSize.width, patchSize.height );
	m_rectUpperLeft.y = sampleROI.y;
	m_rectUpperLeft.x = sampleROI.x;
	m_rectUpperRight.y = sampleROI.y;
	m_rectUpperRight.x = sampleROI.x + sampleROI.width - patchSize.width;
	m_rectLowerLeft.y = sampleROI.y + sampleROI.height - patchSize.height;
	m_rectLowerLeft.x = sampleROI.x;
	m_rectLowerRight.y = sampleROI.y + sampleROI.height - patchSize.height;
	m_rectLowerRight.x = sampleROI.x + sampleROI.width - patchSize.width;
 
 	if( mode == MODE_NEGATIVE ){
 		int negsamples = 4;
 		samples.resize( negsamples );
 		samples[0] = image( m_rectUpperLeft );
 		samples[1] = image( m_rectUpperRight );
 		samples[2] = image( m_rectLowerLeft );
 		samples[3] = image( m_rectLowerRight );
 		return samples;
 	}

 	// while mode == MODE_CLASSIFY

 	int curPatch = 0;

 	for( int curRow = 0; curRow < sampleROI.height - patchSize.height + 1; curRow += stepRow ){
 		for( int curCol = 0; curCol < sampleROI.width - patchSize.width + 1; curCol += stepCol ){
 			samples[curPatch] = image( cv::Rect( sampleROI.x + curCol, sampleROI.y + curRow, patchSize.width, patchSize.height ) );
 			curPatch++;
 		}
 	}
	
 	CV_Assert( curPatch == num ); // checks a condition at runtime and throws exception if it fails. 

 	return samples;
}


} /* namespace BOOSTING */