#include "MIL/feature.hpp"

namespace MIL
{

#define INITSIGMA( numAreas ) ( static_cast<float>( sqrt( 256.0f*256.0f / 12.0f * (numAreas) ) ) );


/*
 * TODO This implementation is based on apps/traincascade/
 * TODO Changed CvHaarEvaluator based on ADABOOSTING implementation (Grabner et al.)
 */

/*********************** CvParams ************************/
CvParams::CvParams() : name( "params" ){

}

void CvParams::printDefaults() const{
	std::cout << "--" << name << "--" << std::endl;
}

void CvParams::printAttrs() const{

}

bool CvParams::scanAttr( const std::string, const std::string ){
  	return false;
}

/********************************** CvFeatureParams *************************************/
CvFeatureParams::CvFeatureParams() :
    maxCatCount( 0 ),
    featSize( 1 ),
    numFeatures( 1 )
{
  	name = "featureParams"; // 头文件中有如下定义: #define CC_FEATURE_PARAMS "featureParams"
}

void CvFeatureParams::init( const CvFeatureParams& fp ){
	maxCatCount = fp.maxCatCount;
	featSize = fp.featSize;
	numFeatures = fp.numFeatures;
}

void CvFeatureParams::write( cv::FileStorage &fs ) const{
	fs << "maxCatCount" << maxCatCount;
	fs << "featSize" << featSize;
	fs << "numFeat" << numFeatures;
}

bool CvFeatureParams::read( const cv::FileNode &node ){
	if( node.empty() )
		return false;
	maxCatCount = node["maxCatCount"];
	featSize = node["featureParams"];
	numFeatures = node["numFeat"];
	return ( maxCatCount >= 0 && featSize >= 1 );
}

cv::Ptr<CvFeatureParams> CvFeatureParams::create( int featureType ){
	/**
  	return featureType == HAAR ? Ptr<CvFeatureParams>( new CvHaarFeatureParams ) : featureType == LBP ? Ptr<CvFeatureParams>( new CvLBPFeatureParams ) :
         featureType == HOG ? Ptr<CvFeatureParams>( new CvHOGFeatureParams ) : Ptr<CvFeatureParams>();
    */
    return featureType == HAAR ? cv::Ptr<CvFeatureParams>( new CvHaarFeatureParams ) : cv::Ptr<CvFeatureParams>();
}


/*********************************** CvHaarFeatureParams **********************************/
CvHaarFeatureParams::CvHaarFeatureParams(){
	name = "haarFeatureParams";
	isIntegral = false;
}

void CvHaarFeatureParams::init( const CvFeatureParams& fp ){
	CvFeatureParams::init( fp );
	isIntegral = ( (const CvHaarFeatureParams&) fp ).isIntegral;
}

void CvHaarFeatureParams::write( cv::FileStorage &fs ) const{
	CvFeatureParams::write( fs );
	fs << "isIntegral" << isIntegral;
}

bool CvHaarFeatureParams::read( const cv::FileNode &node )
{
	if( !CvFeatureParams::read( node ) )
		return false;

	FileNode rnode = node["isIntegral"];
	if( !rnode.isString() )
		return false;
	cv::String intStr;
	rnode >> intStr;
	isIntegral = !intStr.compare( "0" ) ? false : !true;
	return true;
}

void CvHaarFeatureParams::printDefaults() const{
	CvFeatureParams::printDefaults();
	std::cout << "isIntegral: false" << std::endl;
}

void CvHaarFeatureParams::printAttrs() const{
	CvFeatureParams::printAttrs();
	std::string int_str = isIntegral == true ? "true" : "false";
	std::cout << "isIntegral: " << int_str << std::endl;
}

bool CvHaarFeatureParams::scanAttr( const std::string /*prmName*/, const std::string /*val*/){

  return true;
}

/********************************** CvFeatureEvaluator *********************************/
void CvFeatureEvaluator::init( const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize ){
	CV_Assert( _maxSampleCount > 0 );
	featureParams = (CvFeatureParams *) _featureParams; 
	winSize = _winSize;
	numFeatures = _featureParams->numFeatures;
	// Mat.create (int rows, int cols, int type)
	cls.create( (int) _maxSampleCount, 1, CV_32FC1 );
	generateFeatures();
}

void CvFeatureEvaluator::setImage( const Mat &img, uchar clsLabel, int idx ){
	winSize.width = img.cols; // 注： Size  winSize
	winSize.height = img.rows;
	//CV_Assert( img.cols == winSize.width );
	//CV_Assert( img.rows == winSize.height );
	CV_Assert( idx < cls.rows );
	cls.ptr<float>( idx )[0] = clsLabel; 
}

cv::Ptr<CvFeatureEvaluator> CvFeatureEvaluator::create( int type ){
	/**
  	return type == CvFeatureParams::HAAR ? Ptr<CvFeatureEvaluator>( new CvHaarEvaluator ) :
         type == CvFeatureParams::LBP ? Ptr<CvFeatureEvaluator>( new CvLBPEvaluator ) :
         type == CvFeatureParams::HOG ? Ptr<CvFeatureEvaluator>( new CvHOGEvaluator ) : Ptr<CvFeatureEvaluator>();
	*/
   	return type == CvFeatureParams::HAAR ? cv::Ptr<CvFeatureEvaluator>( new CvHaarEvaluator ) : cv::Ptr<CvFeatureEvaluator>();
}


/*************************** CvHaarEvaluator *******************************/
void CvHaarEvaluator::init( const CvFeatureParams *_featureParams, int /*_maxSampleCount*/, Size _winSize ){
	int cols = ( _winSize.width + 1 ) * ( _winSize.height + 1 );
	sum.create( (int) 1, cols, CV_32SC1 ); // Mat sum; --> sum images (each row represents image)
	isIntegral = ( (CvHaarFeatureParams*) _featureParams )->isIntegral;
	CvFeatureEvaluator::init( _featureParams, 1, _winSize ); // virtual void  init (const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
}

void CvHaarEvaluator::setImage( const Mat& img, uchar /*clsLabel*/, int /*idx*/){
	cv::CV_DbgAssert( !sum.empty() );

	winSize.width = img.cols;
	winSize.height = img.rows;

	CvFeatureEvaluator::setImage( img, 1, 0 ); // virtual void  setImage (const Mat &img, uchar clsLabel, int idx)
	if( !isIntegral ){
		std::vector<cv::Mat_<float> > ii_imgs;
		compute_integral( img, ii_imgs );
		_ii_img = ii_imgs[0];
	}
	else{
		_ii_img = img;
	}
}

void CvHaarEvaluator::writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const{
  	_writeFeatures( features, fs, featureMap ); // template function: _writeFeatures
}

void CvHaarEvaluator::writeFeature( cv::FileStorage &fs ) const{
  	cv::String modeStr = isIntegral == true ? "1" : "0";
  	CV_Assert( !modeStr.empty() );
  	fs << "isIntegral" << modeStr;
}

void CvHaarEvaluator::generateFeatures(){
  	generateFeatures( featureParams->numFeatures );
}

void CvHaarEvaluator::generateFeatures( int nFeatures ){  //  随机生成numFeatures个Haar特征
	for ( int i = 0; i < nFeatures; i++ ){
		CvHaarEvaluator::FeatureHaar feature( Size( winSize.width, winSize.height ) );
		features.push_back( feature );  // 注： std::vector<FeatureHaar> features;
	}
}

const std::vector<CvHaarEvaluator::FeatureHaar>& CvHaarEvaluator::getFeatures() const{
  	return features;
}

float CvHaarEvaluator::operator()( int featureIdx, int /*sampleIdx*/){
	/* TODO Added from MIL implementation */
	//return features[featureIdx].calc( _ii_img, Mat(), 0 );
	float res;
	features.at( featureIdx ).eval( _ii_img, Rect( 0, 0, winSize.width, winSize.height ), &res );
	return res;
}

void CvHaarEvaluator::setWinSize( cv::Size patchSize ){
	winSize.width = patchSize.width;
	winSize.height = patchSize.height;
}

cv::Size CvHaarEvaluator::setWinSize() const{
  	return cv::Size( winSize.width, winSize.height );
}

/*********************** CvHaarEvaluator::FeatureHaar ********************************/
CvHaarEvaluator::FeatureHaar::FeatureHaar( Size patchSize ){
	try{
		generateRandomFeature( patchSize );
	}
	catch ( ... ){
		throw;
	}
}

float CvHaarEvaluator::FeatureHaar::getInitMean() const{
  	return m_initMean;
}



} /* namespace MIL */