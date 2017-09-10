#include "MIL/trackerSamplerAlgorithm.hpp"
#include <time.h>

namespace MIL
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
	if( trackerSamplerType.find( "CSC" ) == 0 )
		return cv::Ptr<TrackerSamplerCSC>( new TrackerSamplerCSC() );

	/**
	if( trackerSamplerType.find( "CS" ) == 0 )
	{
		return cv::Ptr<TrackerSamplerCS>( new TrackerSamplerCS() );
	}
	 */

	CV_Error( -1, "Tracker samper algorithm type not supported" );
	return cv::Ptr<TrackerSamplerAlgorithm>();
}

std::string TrackerSamplerAlgorithm::getClassName() const{
	return className;
}


/*************************** TrackerSamplerCSC *********************************/

TrackerSamplerCSC::Params::Params(){
	initInRad = 3;  // inrad for initial pos samples
	initMaxNegNum = 65; 
	searchWinSize = 25; // // inrad for negtive samples
	trackInPosRad = 4;  // inrad for update positive samples
	trackMaxNegNum = 65;
	trackMaxPosNum = 100000;
}

TrackerSamplerCSC::TrackerSamplerCSC( const TrackerSamplerCSC::Params& parameters ) : params( parameters ){
	className = "CSC";
	mode = MODE_INIT_POS; //  for the positive sampling in initialization step
	rng = cv::RNG( uint64( time( 0 ) ) );
}

TrackerSamplerCSC::~TrackerSamplerCSC(){

}

bool TrackerSamplerCSC::samplingImpl( const cv::Mat& image, cv::Rect boundingBox, std::vector<cv::Mat>& samples ){
	// positive samples的左上角坐标在原boundingBox左上角的以inrad为半径的圆内滑动采样；
	// negtive samples的左上角坐标在原boundingBox左上角的以inrad为外半径，outrad为内半径的圆环内滑动采样；

	float inrad = 0;
	float outrad = 0;
	int maxnum = 0;

	switch ( mode ){
		case MODE_INIT_POS:	//  for the positive sampling in initialization step
			inrad = params.initInRad;
			samples = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height,  inrad );
			break;

		case MODE_INIT_NEG:	// for the negative sampling in initialization step
			inrad = 2.0f * params.searchWinSize;
			outrad = 1.5f * params.initInRad;
			maxnum = params.initMaxNegNum;
			samples = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum );
			break;

		case MODE_TRACK_POS:  // for the positive sampling in update step
			inrad = params.trackInPosRad;
			outrad = 0;
			maxnum = params.trackMaxPosNum;
			samples = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum );
			break;

		case MODE_TRACK_NEG: 
			inrad = 1.5f * params.searchWinSize;
			outrad = params.trackInPosRad + 5;
			maxnum = params.trackMaxNegNum;
			samples = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum );
			break;

		case MODE_DETECT:
			inrad = params.searchWinSize;
			samples = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad );
			break;	
		
		default:
			inrad = params.initInRad;
			samples = sampleImage( image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad );
			break;
	}
	return false; // ?
}

void TrackerSamplerCSC::setMode( int samplingMode ){
	mode = samplingMode;
}

std::vector<cv::Mat> TrackerSamplerCSC::sampleImage( const cv::Mat& img, int x, int y, int w, int h, float inrad, float outrad, int maxnum ){
	// 根据inrad,outrad计算boundingBox的左上角坐标的移动范围，然后进行采样子图片
	int rowsz = img.rows - h - 1;
	int colsz = img.cols - w - 1;

	float inradsq = inrad * inrad;
	float outradsq = outrad * outrad;
	int dist;

	// 确定在inrad范围内可移动boundingBox的左上角坐标(x, y)的最小和最大值；
	uint minrow = std::max( 0, (int) y - (int) inrad );
	uint maxrow = std::min( (int) rowsz - 1, (int) y + (int) inrad );
	uint mincol = std::max( 0, (int) x - (int) inrad );
	uint maxcol = std::min( (int) colsz - 1, (int) x + (int) inrad );

	//fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

	std::vector<cv::Mat> samples;
	samples.resize( ( maxrow - minrow + 1 ) * ( maxcol - mincol + 1 ) );
	int i = 0;

	float prob = ( (float) maxnum ) / samples.size(); // 默认值 maxnum = 1000000

	for( int r = int( minrow ); r <= int( maxrow ); r++ )
		for( int c = int( mincol ); c <= int( maxcol ); c++ ){
			dist = ( y - r ) * ( y - r ) + ( x - c ) * ( x - c );
			if( dist < inradsq && dist >= outradsq && float( rng.uniform( 0.f, 1.f ) ) < prob ){
				samples[i] = img( cv::Rect( c, r, w, h ) );
				i++;
			}
		}
	samples.resize( std::min( i, maxnum ) );
	return samples;
}


} /* namespace MIL */