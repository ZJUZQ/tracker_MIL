#include "BOOSTING_DIRECTION/feature.hpp"

namespace BOOSTING_DIRECTION
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

	cv::FileNode rnode = node["isIntegral"];
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
void CvFeatureEvaluator::init( const CvFeatureParams *_featureParams, int _maxSampleCount, cv::Size _winSize ){
	CV_Assert( _maxSampleCount > 0 );
	featureParams = (CvFeatureParams *) _featureParams; 
	winSize = _winSize;
	numFeatures = _featureParams->numFeatures;
	// Mat.create (int rows, int cols, int type)
	cls.create( (int) _maxSampleCount, 1, CV_32FC1 ); // classes? 
	generateFeatures();
}

void CvFeatureEvaluator::setImage( const cv::Mat &img, uchar clsLabel, int idx ){
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
void CvHaarEvaluator::init( const CvFeatureParams *_featureParams, int /*_maxSampleCount*/, cv::Size _winSize ){
	int cols = ( _winSize.width + 1 ) * ( _winSize.height + 1 );
	sum.create( (int) 1, cols, CV_32SC1 ); // cv::Mat sum; --> sum images (each row represents image)
	isIntegral = ( (CvHaarFeatureParams*) _featureParams )->isIntegral;
	CvFeatureEvaluator::init( _featureParams, 1, _winSize ); // virtual void  init (const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
}

void CvHaarEvaluator::setImage( const cv::Mat& img, uchar /*clsLabel*/, int /*idx*/){
	CV_DbgAssert( !sum.empty() );

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
		CvHaarEvaluator::FeatureHaar feature( cv::Size( winSize.width, winSize.height ) );
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
	features.at( featureIdx ).eval( _ii_img, cv::Rect( 0, 0, winSize.width, winSize.height ), &res );
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
CvHaarEvaluator::FeatureHaar::FeatureHaar( cv::Size patchSize ){
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

float CvHaarEvaluator::FeatureHaar::getInitSigma() const{
  	return m_initSigma;
}

void CvHaarEvaluator::FeatureHaar::generateRandomFeature( cv::Size patchSize ){
	cv::Point2i position;
	cv::Size baseDim;
	cv::Size sizeFactor;
	int area;

	// Size minSize = cv::Size( 3, 3 );
	int minArea = 9;

	bool valid = false;
	while( !valid ){
		//choose position and scale
    	// 在patchSize子图中随机选择Haar特征左上角的位置position和单个小方格的尺寸baseDim
    	position.y = std::rand() % ( patchSize.height );
    	position.x = std::rand() % ( patchSize.width );

    	// baseDim: 表示Haar特征中单个小方格的尺寸
    	baseDim.width = (int) ( ( 1 - std::sqrt( 1 - (float) std::rand() / RAND_MAX ) ) * patchSize.width );
    	baseDim.height = (int) ( ( 1 - std::sqrt( 1 - (float) std::rand() / RAND_MAX ) ) * patchSize.height );

    	// select FeatureHaar types
    	//float probType[11] = {0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0950f};
	    float probType[11] =
	    	{ 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	    // 这样设置probType，说明只选择前5种中的Haar特征类型

	    float prob = (float) std::rand() / RAND_MAX;	// 根据随机产生的prob，选择将要生成的Haar特征类型

	    if( prob < probType[0] ){
	    	// check if the feature is valid
	    	sizeFactor.height = 2;
	    	sizeFactor.width = 1;
	    	if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
	    		continue;
	    	area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
	    	if( area < minArea )
	    		continue;

	    	m_type = 1;   //  特征类型-1
      		m_numAreas = 2; // 该类型Haar特征由上下两个方框构成
      		m_weights.resize( m_numAreas );
      		m_weights[0] = 1;
      		m_weights[1] = -1;
      		m_areas.resize( m_numAreas );

      		m_areas[0].x = position.x;
      		m_areas[0].y = position.y;
      		m_areas[0].height = baseDim.height;
      		m_areas[0].width = baseDim.width;

      		m_areas[1].x = position.x;
      		m_areas[1].y = position.x + baseDim.height;
      		m_areas[1].height = baseDim.height;
      		m_areas[1].width = baseDim.width;

      		m_initMean = 0;
      		m_initSigma = INITSIGMA( m_numAreas );

      		valid = true;
	    }
	    else if( prob < probType[0] + probType[1] ){
	    	//check if feature is valid
			sizeFactor.height = 1;
			sizeFactor.width = 2;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 2; // 特征类型-2, 该类型Haar特征由左右两个方框构成
			m_numAreas = 2;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -1;
			m_areas.resize( m_numAreas );

			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;

			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;

			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
	    }
	    else if( prob < probType[0] + probType[1] + probType[2] ){
			//check if feature is valid
			sizeFactor.height = 4;
			sizeFactor.width = 1;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 3;
			m_numAreas = 3;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -2;
			m_weights[2] = 1;
			m_areas.resize( m_numAreas );

			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;

			m_areas[1].x = position.x;
			m_areas[1].y = position.y + baseDim.height;
			m_areas[1].height = 2 * baseDim.height;
			m_areas[1].width = baseDim.width;

			m_areas[2].y = position.y + 3 * baseDim.height;
			m_areas[2].x = position.x;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;

			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
	    }
	    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] ){
			//check if feature is valid
			sizeFactor.height = 1;
			sizeFactor.width = 4;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 3;
			m_numAreas = 3;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -2;
			m_weights[2] = 1;
			m_areas.resize( m_numAreas );

			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;

			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = 2 * baseDim.width;

			m_areas[2].y = position.y;
			m_areas[2].x = position.x + 3 * baseDim.width;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;

			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
	    }
	    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] ){
			//check if feature is valid
			sizeFactor.height = 2;
			sizeFactor.width = 2;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 5;
			m_numAreas = 4;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -1;
			m_weights[2] = -1;
			m_weights[3] = 1;
			m_areas.resize( m_numAreas );

			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;

			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;

			m_areas[2].y = position.y + baseDim.height;
			m_areas[2].x = position.x;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;

			m_areas[3].y = position.y + baseDim.height;
			m_areas[3].x = position.x + baseDim.width;
			m_areas[3].height = baseDim.height;
			m_areas[3].width = baseDim.width;

			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
	    }
	    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] ){
	    	//check if feature is valid
			sizeFactor.height = 3;
			sizeFactor.width = 3;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 6;
			m_numAreas = 2;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -9;
			m_areas.resize( m_numAreas );

			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = 3 * baseDim.height;
			m_areas[0].width = 3 * baseDim.width;

			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y + baseDim.height;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;

			m_initMean = -8 * 128;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
	    }
	    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] ){
			//check if feature is valid
			sizeFactor.height = 3;
			sizeFactor.width = 1;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 7;
			m_numAreas = 3;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -2;
			m_weights[2] = 1;
			m_areas.resize( m_numAreas );
			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;
			m_areas[1].x = position.x;
			m_areas[1].y = position.y + baseDim.height;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;
			m_areas[2].y = position.y + baseDim.height * 2;
			m_areas[2].x = position.x;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;
			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
    	}
    	else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] ){
			//check if feature is valid
			sizeFactor.height = 1;
			sizeFactor.width = 3;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;

			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;

			if( area < minArea )
				continue;

			m_type = 8;
			m_numAreas = 3;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -2;
			m_weights[2] = 1;
			m_areas.resize( m_numAreas );
			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;
			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;
			m_areas[2].y = position.y;
			m_areas[2].x = position.x + 2 * baseDim.width;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;
			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
    	}
    	else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] ){
			//check if feature is valid
			sizeFactor.height = 3;
			sizeFactor.width = 3;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 9;
			m_numAreas = 2;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -2;
			m_areas.resize( m_numAreas );
			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = 3 * baseDim.height;
			m_areas[0].width = 3 * baseDim.width;
			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y + baseDim.height;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;
			m_initMean = 0;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
    	}
    	else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] + probType[9] ){
			//check if feature is valid
			sizeFactor.height = 3;
			sizeFactor.width = 1;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
				continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
				continue;

			m_type = 10;
			m_numAreas = 3;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -1;
			m_weights[2] = 1;
			m_areas.resize( m_numAreas );
			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;
			m_areas[1].x = position.x;
			m_areas[1].y = position.y + baseDim.height;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;
			m_areas[2].y = position.y + baseDim.height * 2;
			m_areas[2].x = position.x;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;
			m_initMean = 128;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
    	}
    	else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] + probType[9] + probType[10] ){
			//check if feature is valid
			sizeFactor.height = 1;
			sizeFactor.width = 3;
			if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
			continue;
			area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
			if( area < minArea )
			continue;

			m_type = 11;
			m_numAreas = 3;
			m_weights.resize( m_numAreas );
			m_weights[0] = 1;
			m_weights[1] = -1;
			m_weights[2] = 1;
			m_areas.resize( m_numAreas );
			m_areas[0].x = position.x;
			m_areas[0].y = position.y;
			m_areas[0].height = baseDim.height;
			m_areas[0].width = baseDim.width;
			m_areas[1].x = position.x + baseDim.width;
			m_areas[1].y = position.y;
			m_areas[1].height = baseDim.height;
			m_areas[1].width = baseDim.width;
			m_areas[2].y = position.y;
			m_areas[2].x = position.x + 2 * baseDim.width;
			m_areas[2].height = baseDim.height;
			m_areas[2].width = baseDim.width;
			m_initMean = 128;
			m_initSigma = INITSIGMA( m_numAreas );
			valid = true;
    	}
    	else
    		CV_Error( CV_StsAssert, "" );
	}

	m_initSize = patchSize;
	m_curSize = m_initSize;
	m_scaleFactorWidth = m_scaleFactorHeight = 1.0f;
	m_scaleAreas.resize( m_numAreas ); // std::vector<cv::Rect> m_scaleAreas
	m_scaleWeights.resize( m_numAreas );
	for( int curArea = 0; curArea < m_numAreas; curArea++ ){
		m_scaleAreas[curArea] = m_areas[curArea]; // std::vector<cv::Rect> m_areas
		m_scaleWeights[curArea] = (float) m_weights[curArea] / (float) ( m_areas[curArea].width * m_areas[curArea].height );
		//  std::vector<float> m_scaleWeights;		weights after scaling, 单个像素的权重
	}
}

bool CvHaarEvaluator::FeatureHaar::eval( const cv::Mat& i_image, cv::Rect /*ROI*/, float* result ) const{
	*result = 0.0f;
	for( int curArea = 0; curArea < m_numAreas; curArea++ ){
		*result += (float) getSum( i_image, cv::Rect( m_areas[curArea].x, m_areas[curArea].y, m_areas[curArea].width, m_areas[curArea].height ) ) * m_scaleWeights[curArea];
	}

	/*
	if( image->getUseVariance() )
	{
		float variance = (float) image->getVariance( ROI );
		*result /= variance;
	}
	*/

  return true;
}

float CvHaarEvaluator::FeatureHaar::getSum( const cv::Mat& image, cv::Rect imageROI ) const{
	// 计算imageROI中像素的和

	// left upper Origin
	int OriginX = imageROI.x;
	int OriginY = imageROI.y;

	// chect and fix width and height
	int Width = imageROI.width;
	int Height = imageROI.height;

	if( OriginX + Width >= image.cols - 1 )
		Width = ( image.cols - 1 ) - OriginX;
	if( OriginY + Height >= image.rows - 1 )
		Height = ( image.rows - 1 ) - OriginY;

	float value = 0;
	int depth = image.depth();

	//  这里用的integral_image计算Haar特征
	if( depth == CV_8U || depth == CV_32S )
		value = static_cast<float>( image.at<int>( OriginY + Height, OriginX + Width ) + image.at<int>( OriginY, OriginX ) - image.at<int>(OriginY, OriginX + Width)
				- image.at<int>(OriginY + Height, OriginX) );

	else if( depth == CV_64F )
		value = static_cast<float>( image.at<double>( OriginY + Height, OriginX + Width ) + image.at<double>( OriginY, OriginX ) - image.at<double>(OriginY, OriginX + Width)
				- image.at<double>(OriginY + Height, OriginX) );

	else if( depth == CV_32F )
		value = static_cast<float>( image.at<float>( OriginY + Height, OriginX + Width ) + image.at<float>( OriginY, OriginX ) - image.at<float>(OriginY, OriginX + Width)
				- image.at<float>(OriginY + Height, OriginX) );

	return value;
}

int CvHaarEvaluator::FeatureHaar::getNumAreas(){
	return m_numAreas;
}

const std::vector<float>& CvHaarEvaluator::FeatureHaar::getWeights() const{
	return m_weights;
}

const std::vector<cv::Rect>& CvHaarEvaluator::FeatureHaar::getAreas() const{
	return m_areas;
}



} /* namespace BOOSTING_DIRECTION */