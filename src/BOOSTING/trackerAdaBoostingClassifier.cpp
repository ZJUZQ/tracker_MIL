#include "BOOSTING/trackerAdaBoostingClassifier.hpp"

namespace BOOSTING
{

/********************************* StrongClassifierDirectSelection **********************************/
StrongClassifierDirectSelection::StrongClassifierDirectSelection( int numBaseClf, int numWeakClf, cv::Size patchSz, const cv::Rect& sampleROI,
                                                                  bool useFeatureEx, int iterationInit ){
	//StrongClassifier
	numBaseClassifier = numBaseClf;
	numWeakClassifier = numWeakClf;
	numAllWeakClassifier = numWeakClf + iterationInit;
	iterInit = iterationInit;

	alpha.assign( numBaseClf, 0 ); // vector::assign --> Assigns new contents to the vector, replacing its current contents, and modifying its size accordingly.
	// 构成strong_classifier的各个base_classifier的权重

	patchSize = patchSz;
	useFeatureExchange = useFeatureEx;

	m_errorMask.resize( numAllWeakClassifier );
	m_errors.resize( numAllWeakClassifier );
	m_sumErrors.resize( numAllWeakClassifier );

	ROI = sampleROI;
	detector = new Detector( this );
}

void StrongClassifierDirectSelection::initBaseClassifier(){
  	baseClassifier = new BaseClassifier*[numBaseClassifier];
  	baseClassifier[0] = new BaseClassifier( numWeakClassifier, iterInit );

  	for ( int curBaseClassifier = 1; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    	baseClassifier[curBaseClassifier] = new BaseClassifier( numWeakClassifier, iterInit, baseClassifier[0]->getReferenceWeakClassifier() );
}

StrongClassifierDirectSelection::~StrongClassifierDirectSelection(){
  	for ( int curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    	delete baseClassifier[curBaseClassifier];
  	delete[] baseClassifier;
  	alpha.clear();
  	delete detector;
}

cv::Size StrongClassifierDirectSelection::getPatchSize() const{
  	return patchSize;
}

cv::Rect StrongClassifierDirectSelection::getROI() const{
  	return ROI;
}

float StrongClassifierDirectSelection::classifySmooth( const std::vector<cv::Mat>& images, const cv::Rect& sampleROI, int& idx ){
	ROI = sampleROI;
	idx = 0;
	float confidence = 0;
	//detector->classify (image, patches);
	detector->classifySmooth( images );

	//move to best detection
	if( detector->getNumDetections() <= 0 ){
		confidence = 0;
		return confidence;
	}
	idx = detector->getPatchIdxOfBestDetection();
	confidence = detector->getConfidenceOfBestDetection();

	return confidence;
}

bool StrongClassifierDirectSelection::getUseFeatureExchange() const{
  	return useFeatureExchange;
}

int StrongClassifierDirectSelection::getReplacedClassifier() const{
  	return replacedClassifier;
}

int StrongClassifierDirectSelection::getSwappedClassifier() const{
  	return swappedClassifier;
}

bool StrongClassifierDirectSelection::update( const cv::Mat& image, int target, float importance ){
	m_errorMask.assign( (size_t)numAllWeakClassifier, false );
	m_errors.assign( (size_t)numAllWeakClassifier, 0.0f );
	m_sumErrors.assign( (size_t)numAllWeakClassifier, 0.0f );

  	baseClassifier[0]->trainClassifier( image, target, importance, m_errorMask ); // BaseClassifier** baseClassifier;
  	
  	for ( int curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ ){
		int selectedClassifier = baseClassifier[curBaseClassifier]->selectBestClassifier( m_errorMask, importance, m_errors );

		if( m_errors[selectedClassifier] >= 0.5 )
			alpha[curBaseClassifier] = 0;
		else
			alpha[curBaseClassifier] = std::log( ( 1.0f - m_errors[selectedClassifier] ) / m_errors[selectedClassifier] );

		if( m_errorMask[selectedClassifier] )
			importance *= (float) std::sqrt( ( 1.0f - m_errors[selectedClassifier] ) / m_errors[selectedClassifier] );
		else
			importance *= (float) std::sqrt( m_errors[selectedClassifier] / ( 1.0f - m_errors[selectedClassifier] ) );

	    //weight limitation
	    //if (importance > 100) importance = 100;

	    //sum up errors
	    for ( int curWeakClassifier = 0; curWeakClassifier < numAllWeakClassifier; curWeakClassifier++ ){
	      	if( m_errors[curWeakClassifier] != FLT_MAX && m_sumErrors[curWeakClassifier] >= 0 )
	        		m_sumErrors[curWeakClassifier] += m_errors[curWeakClassifier];
	    }

	    //mark feature as used
	    m_sumErrors[selectedClassifier] = -1;
	    m_errors[selectedClassifier] = FLT_MAX;
 	}

	if( useFeatureExchange ){
		replacedClassifier = baseClassifier[0]->computeReplaceWeakestClassifier( m_sumErrors );
		swappedClassifier = baseClassifier[0]->getIdxOfNewWeakClassifier();
	}

  	return true;
}

void StrongClassifierDirectSelection::replaceWeakClassifier( int idx ){
	if( useFeatureExchange && idx >= 0 ){
		baseClassifier[0]->replaceWeakClassifier( idx );
		for ( int curBaseClassifier = 1; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
	  		baseClassifier[curBaseClassifier]->replaceClassifierStatistic( baseClassifier[0]->getIdxOfNewWeakClassifier(), idx );
	}
}

std::vector<int> StrongClassifierDirectSelection::getSelectedWeakClassifier(){
	std::vector<int> selected;
	int curBaseClassifier = 0;
	for ( curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ ){
		selected.push_back( baseClassifier[curBaseClassifier]->getSelectedClassifier() );
	}
	return selected;
}

float StrongClassifierDirectSelection::eval( const cv::Mat& response ){
	float value = 0.0f;
	int curBaseClassifier = 0;

	for ( curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
		value += baseClassifier[curBaseClassifier]->eval( response ) * alpha[curBaseClassifier];

	return value;
}

int StrongClassifierDirectSelection::getNumBaseClassifier(){
  	return numBaseClassifier;
}

/******************************* BaseClassifier ****************************************/

BaseClassifier::BaseClassifier( int numWeakClassifier, int iterationInit ){

	this->m_numWeakClassifier = numWeakClassifier;
	this->m_iterationInit = iterationInit;

	// 定义: WeakClassifierHaarFeature** weakClassifier
	weakClassifier = new WeakClassifierHaarFeature*[numWeakClassifier + iterationInit];
	m_idxOfNewWeakClassifier = numWeakClassifier;

	generateRandomClassifier(); // 生成 numWeakClassifier + iterationInit 个 weakclassifiers

	m_referenceWeakClassifier = false;
	m_selectedClassifier = 0;

	m_wCorrect.assign( numWeakClassifier + iterationInit, 0 ); // 记录每个弱分类器到目前为止正确分类的数目
	m_wWrong.assign( numWeakClassifier + iterationInit, 0 );

	for( int curWeakClassifier = 0; curWeakClassifier < numWeakClassifier + iterationInit; curWeakClassifier++ )
		m_wWrong[curWeakClassifier] = m_wCorrect[curWeakClassifier] = 1; //  初始化
}

BaseClassifier::BaseClassifier( int numWeakClassifier, int iterationInit, WeakClassifierHaarFeature** weakCls ){

	m_numWeakClassifier = numWeakClassifier;
	m_iterationInit = iterationInit;
	weakClassifier = weakCls;

	m_referenceWeakClassifier = true; //  引用了weakCls

	m_selectedClassifier = 0;
	m_idxOfNewWeakClassifier = numWeakClassifier;

	m_wCorrect.assign( numWeakClassifier + iterationInit, 0 );
	m_wWrong.assign( numWeakClassifier + iterationInit, 0 );

	for ( int curWeakClassifier = 0; curWeakClassifier < numWeakClassifier + iterationInit; curWeakClassifier++ )
		m_wWrong[curWeakClassifier] = m_wCorrect[curWeakClassifier] = 1;
}

BaseClassifier::~BaseClassifier(){

	if( !m_referenceWeakClassifier ){
		for( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
	  		delete weakClassifier[curWeakClassifier];
		delete[] weakClassifier;
	}
	m_wCorrect.clear();
	m_wWrong.clear();
}

void BaseClassifier::generateRandomClassifier(){
	for( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
		weakClassifier[curWeakClassifier] = new WeakClassifierHaarFeature();
}

int BaseClassifier::eval( const cv::Mat& image ){
  	return weakClassifier[m_selectedClassifier]->eval( image.at<float>( m_selectedClassifier ) );
}

int BaseClassifier::getSelectedClassifier() const{
  	return m_selectedClassifier;
}

void BaseClassifier::trainClassifier( const cv::Mat& image, int target, float importance, std::vector<bool>& errorMask ){

	//get poisson value
	double A = 1;
	int K = 0;
	int K_max = 10;

	for( ; ; )
	{
		double U_k = (double) std::rand() / RAND_MAX;
		A *= U_k;
		if( K > K_max || A < exp( -importance ) )
		  	break;
		K++;
	}

	for( int curK = 0; curK <= K; curK++ ){
		for( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ ){
	  		// errorMask: true表示分类器估计错误，false表示估计正确
	  		errorMask[curWeakClassifier] = weakClassifier[curWeakClassifier]->update( image.at<float>( curWeakClassifier ), target );
		}
	}
}

float BaseClassifier::getError( int curWeakClassifier ){
	// calculate e_{n, m}

  	if( curWeakClassifier == -1 )
    	curWeakClassifier = m_selectedClassifier;
  	return m_wWrong[curWeakClassifier] / ( m_wWrong[curWeakClassifier] + m_wCorrect[curWeakClassifier] );
}

int BaseClassifier::selectBestClassifier( std::vector<bool>& errorMask, float importance, std::vector<float> & errors ){
  	float minError = FLT_MAX;
  	int tmp_selectedClassifier = m_selectedClassifier;

  	for( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ ){
    	if( errorMask[curWeakClassifier] ) // 分类错误
      		m_wWrong[curWeakClassifier] += importance;
    	else
     		 m_wCorrect[curWeakClassifier] += importance;


    	if( errors[curWeakClassifier] == FLT_MAX )
      		continue;

    	errors[curWeakClassifier] = m_wWrong[curWeakClassifier] / ( m_wWrong[curWeakClassifier] + m_wCorrect[curWeakClassifier] );

		/*if(errors[curWeakClassifier] < 0.001 || !(errors[curWeakClassifier]>0.0))
		 {
		 	errors[curWeakClassifier] = 0.001;
		 }

		 if(errors[curWeakClassifier] >= 1.0)
		 	errors[curWeakClassifier] = 0.999;

		 assert (errors[curWeakClassifier] > 0.0);
		 assert (errors[curWeakClassifier] < 1.0);*/

    	if( curWeakClassifier < m_numWeakClassifier ){
      		if( errors[curWeakClassifier] < minError ){
        		minError = errors[curWeakClassifier];
        		tmp_selectedClassifier = curWeakClassifier;
      		}
    	}
  	}

  	m_selectedClassifier = tmp_selectedClassifier;
  	return m_selectedClassifier;
}

void BaseClassifier::getErrors( float* errors ){
  	for( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ ){
    	if( errors[curWeakClassifier] == FLT_MAX )
      		continue;

    	errors[curWeakClassifier] = m_wWrong[curWeakClassifier] / ( m_wWrong[curWeakClassifier] + m_wCorrect[curWeakClassifier] );

    	CV_Assert( errors[curWeakClassifier] > 0 );
  	}
}

void BaseClassifier::replaceWeakClassifier( int index ){
	delete weakClassifier[index];
	weakClassifier[index] = weakClassifier[m_idxOfNewWeakClassifier];
	m_wWrong[index] = m_wWrong[m_idxOfNewWeakClassifier];
	m_wWrong[m_idxOfNewWeakClassifier] = 1;
	m_wCorrect[index] = m_wCorrect[m_idxOfNewWeakClassifier];
	m_wCorrect[m_idxOfNewWeakClassifier] = 1;

	weakClassifier[m_idxOfNewWeakClassifier] = new WeakClassifierHaarFeature();
}

int BaseClassifier::computeReplaceWeakestClassifier( const std::vector<float> & errors ){
	float maxError = 0.0f;
	int index = -1;

  	//search the classifier with the largest error
	for ( int curWeakClassifier = m_numWeakClassifier - 1; curWeakClassifier >= 0; curWeakClassifier-- ){
		if( errors[curWeakClassifier] > maxError ){
			maxError = errors[curWeakClassifier];
			index = curWeakClassifier;
		}
	}

	CV_Assert( index > -1 );
	CV_Assert( index != m_selectedClassifier );

	//replace
  	m_idxOfNewWeakClassifier++;
  	if( m_idxOfNewWeakClassifier == m_numWeakClassifier + m_iterationInit )
    	m_idxOfNewWeakClassifier = m_numWeakClassifier;

  	if( maxError > errors[m_idxOfNewWeakClassifier] )
    	return index;
  	else
    	return -1;
}

void BaseClassifier::replaceClassifierStatistic( int sourceIndex, int targetIndex ){
	CV_Assert( targetIndex >= 0 );
	CV_Assert( targetIndex != m_selectedClassifier );
	CV_Assert( targetIndex < m_numWeakClassifier );

	//replace
	m_wWrong[targetIndex] = m_wWrong[sourceIndex];
	m_wWrong[sourceIndex] = 1.0f;
	m_wCorrect[targetIndex] = m_wCorrect[sourceIndex];
	m_wCorrect[sourceIndex] = 1.0f;
}

/******************************** EstimatedGaussDistribution *********************************/

EstimatedGaussDistribution::EstimatedGaussDistribution(){
	m_mean = 0;
	m_sigma = 1;
	this->m_P_mean = 1000;  // initial state for P, where P is the estimate error covariance
	this->m_R_mean = 0.01f;
	this->m_P_sigma = 1000;
	this->m_R_sigma = 0.01f;
}

EstimatedGaussDistribution::EstimatedGaussDistribution( float P_mean, float R_mean, float P_sigma, float R_sigma ){
	m_mean = 0;
	m_sigma = 1;
	this->m_P_mean = P_mean;  // P: estimate error covariance, for m_mean;
	this->m_R_mean = R_mean;  // P: estimate error covariance, for m_sigma;
	this->m_P_sigma = P_sigma; 
	this->m_R_sigma = R_sigma;
}

EstimatedGaussDistribution::~EstimatedGaussDistribution(){

}

void EstimatedGaussDistribution::update( float value ){
	//update distribution (mean and sigma) using a kalman filter for each

	float K;
	float minFactor = 0.001f;

	//mean

  	K = m_P_mean / ( m_P_mean + m_R_mean ); // P is the estimate error covariance
 	if( K < minFactor )
    	K = minFactor;

  	m_mean = K * value + ( 1.0f - K ) * m_mean;
  	m_P_mean = m_P_mean * m_R_mean / ( m_P_mean + m_R_mean );

  	K = m_P_sigma / ( m_P_sigma + m_R_sigma );
  	if( K < minFactor )
    	K = minFactor;

  	float tmp_sigma = K * ( m_mean - value ) * ( m_mean - value ) + ( 1.0f - K ) * m_sigma * m_sigma;
  	m_P_sigma = m_P_sigma * m_R_mean / ( m_P_sigma + m_R_sigma );

  	m_sigma = static_cast<float>( sqrt( tmp_sigma ) );
  	if( m_sigma <= 1.0f )
    	m_sigma = 1.0f;
}

void EstimatedGaussDistribution::setValues( float mean, float sigma ){
	this->m_mean = mean;
	this->m_sigma = sigma;
}

float EstimatedGaussDistribution::getMean(){
  	return m_mean;
}

float EstimatedGaussDistribution::getSigma(){
  	return m_sigma;
}

/****************************** WeakClassifierHaarFeature ****************************************/

WeakClassifierHaarFeature::WeakClassifierHaarFeature(){
	sigma = 1;
	mean = 0;

	EstimatedGaussDistribution* m_posSamples = new EstimatedGaussDistribution();
	EstimatedGaussDistribution* m_negSamples = new EstimatedGaussDistribution();
	generateRandomClassifier( m_posSamples, m_negSamples );

	getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( -1 ) );
	// set the negtive sample gaussion distribution's initial mean and sigma

	getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( 1 ) );
	// set the positive sample gaussion distribution's initial mean and sigma
}

WeakClassifierHaarFeature::~WeakClassifierHaarFeature(){
  	delete m_classifier;
}

void WeakClassifierHaarFeature::getInitialDistribution( EstimatedGaussDistribution* distribution ){
  	distribution->setValues( mean, sigma );
}

void WeakClassifierHaarFeature::generateRandomClassifier( EstimatedGaussDistribution* m_posSamples, EstimatedGaussDistribution* m_negSamples ){
  	m_classifier = new ClassifierThreshold( m_posSamples, m_negSamples );
}

bool WeakClassifierHaarFeature::update( float value, int target ){
  	m_classifier->update( value, target ); 	// update the gaussion distribution's parameter: mean and sigma
                                         	// update the classifierthreshold's parameter: threshold and parity

  	return ( m_classifier->eval( value ) != target );
}

int WeakClassifierHaarFeature::eval( float value ){
  	return m_classifier->eval( value );
}

/*********************************** Detector ***********************************/

Detector::Detector( StrongClassifierDirectSelection* classifier ) : m_sizeDetections( 0 ){
	this->m_classifier = classifier;

	m_sizeConfidences = 0;
	m_maxConfidence = -FLT_MAX;
	m_numDetections = 0;
	m_idxBestDetection = -1;
}

Detector::~Detector(){

}

void Detector::prepareConfidencesMemory( int numPatches ){
  	if( numPatches <= m_sizeConfidences )
    	return;

  	m_sizeConfidences = numPatches;
  	m_confidences.resize( numPatches );
}

void Detector::prepareDetectionsMemory( int numDetections ){
  	if( numDetections <= m_sizeDetections )
    	return;

  	m_sizeDetections = numDetections;
  	m_idxDetections.resize( numDetections );
}

void Detector::classifySmooth( const std::vector<cv::Mat>& images, float minMargin ){
	int numPatches = static_cast<int>(images.size());

	prepareConfidencesMemory( numPatches );

	m_numDetections = 0;
	m_idxBestDetection = -1;
	m_maxConfidence = -FLT_MAX;

	//compute grid
	//TODO 0.99 overlap from params
	cv::Size patchSz = m_classifier->getPatchSize();
	int stepCol = (int) floor( ( 1.0f - 0.99f ) * (float) patchSz.width + 0.5f );
	int stepRow = (int) floor( ( 1.0f - 0.99f ) * (float) patchSz.height + 0.5f );
	if( stepCol <= 0 )
		stepCol = 1;
	if( stepRow <= 0 )
		stepRow = 1;

	cv::Size patchGrid;
	cv::Rect ROI = m_classifier->getROI();
	patchGrid.height = ( (int) ( (float) ( ROI.height - patchSz.height ) / stepRow ) + 1 );
	patchGrid.width = ( (int) ( (float) ( ROI.width - patchSz.width ) / stepCol ) + 1 );

	if( ( patchGrid.width != m_confMatrix.cols ) || ( patchGrid.height != m_confMatrix.rows ) ){
		m_confMatrix.create( patchGrid.height, patchGrid.width );
		m_confMatrixSmooth.create( patchGrid.height, patchGrid.width );
		m_confImageDisplay.create( patchGrid.height, patchGrid.width ); // value convert to [0, 255] for display
  	}

  	int curPatch = 0;
  	// Eval and filter
  	for( int row = 0; row < patchGrid.height; row++ ){
    	for ( int col = 0; col < patchGrid.width; col++ ){
      		m_confidences[curPatch] = m_classifier->eval( images[curPatch] );

      		// fill matrix
			m_confMatrix( row, col ) = m_confidences[curPatch];
			curPatch++;
    	}
  	}

	// Filter
	//cv::GaussianBlur(m_confMatrix,m_confMatrixSmooth,cv::Size(3,3),0.8);
	cv::GaussianBlur( m_confMatrix, m_confMatrixSmooth, cv::Size( 3, 3 ), 0 );

	// Make display friendly
	double min_val, max_val;
	cv::minMaxLoc( m_confMatrixSmooth, &min_val, &max_val );

	for( int y = 0; y < m_confImageDisplay.rows; y++ ){
    	unsigned char* pConfImg = m_confImageDisplay[y];
    	const float* pConfData = m_confMatrixSmooth[y];
    	for( int x = 0; x < m_confImageDisplay.cols; x++, pConfImg++, pConfData++ ){
      		*pConfImg = static_cast<unsigned char>( 255.0 * ( *pConfData - min_val ) / ( max_val - min_val ) );
    	}
  	}

  	// Get best detection
  	curPatch = 0;
  	for( int row = 0; row < patchGrid.height; row++ ){
    	for ( int col = 0; col < patchGrid.width; col++ ){
      		// fill matrix
      		m_confidences[curPatch] = m_confMatrixSmooth( row, col );

      		if( m_confidences[curPatch] > m_maxConfidence ){
		        m_maxConfidence = m_confidences[curPatch];
		        m_idxBestDetection = curPatch;
      		}

     		if( m_confidences[curPatch] > minMargin )
        		m_numDetections++;
     
      		curPatch++;
    	}
  	}

  	prepareDetectionsMemory( m_numDetections );
  	int curDetection = -1;

  	for ( int currentPatch = 0; currentPatch < numPatches; currentPatch++ ){
    	if( m_confidences[currentPatch] > minMargin )
      		m_idxDetections[++curDetection] = currentPatch;
  	}
}

int Detector::getNumDetections(){
  	return m_numDetections;
}

float Detector::getConfidence( int patchIdx ){
  	return m_confidences[patchIdx];
}

float Detector::getConfidenceOfDetection( int detectionIdx ){
  	return m_confidences[getPatchIdxOfDetection( detectionIdx )];
}

int Detector::getPatchIdxOfBestDetection(){
  	return m_idxBestDetection;
}

int Detector::getPatchIdxOfDetection( int detectionIdx ){
  	return m_idxDetections[detectionIdx];
}

/********************************** ClassifierThreshold *********************************/

ClassifierThreshold::ClassifierThreshold( EstimatedGaussDistribution* posSamples, EstimatedGaussDistribution* negSamples ){
	m_posSamples = posSamples;
	m_negSamples = negSamples;
	m_threshold = 0.0f;
	m_parity = 0;
}

ClassifierThreshold::~ClassifierThreshold(){
  	if( m_posSamples != NULL )
    	delete m_posSamples;
  	if( m_negSamples != NULL )
    	delete m_negSamples;
}

void* ClassifierThreshold::getDistribution( int target ){
  	if( target == 1 )
    	return m_posSamples;
  	else
    	return m_negSamples;
}

void ClassifierThreshold::update( float value, int target ){
  	//update distribution
  	if( target == 1 )
    	m_posSamples->update( value ); // update the u+ and sigma+
  	else
    	m_negSamples->update( value ); // update the u- and sigma-

	//adapt threshold and parity
	m_threshold = ( m_posSamples->getMean() + m_negSamples->getMean() ) / 2.0f;
	m_parity = ( m_posSamples->getMean() > m_negSamples->getMean() ) ? 1 : -1;
}

int ClassifierThreshold::eval( float value ){
	// return the sample's label, 1 or -1

  	return ( ( ( m_parity * ( value - m_threshold ) ) > 0 ) ? 1 : -1 );
}


} /* namespace BOOSTING */