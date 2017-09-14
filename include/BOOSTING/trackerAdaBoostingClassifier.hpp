#ifndef BOOSTING_TRACKERADABOOSTINGCLASSIFIER_HPP
#define BOOSTING_TRACKERADABOOSTINGCLASSIFIER_HPP

#include "BOOSTING/common_includes.hpp"

namespace BOOSTING
{

//! @addtogroup tracking
//! @{

//TODO based on the original implementation
//http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

class BaseClassifier;
class WeakClassifierHaarFeature;
class EstimatedGaussDistribution;
class ClassifierThreshold;
class Detector;

/******************************* StrongClassifierDirectSelection ***********************************/
class StrongClassifierDirectSelection{

public:
	StrongClassifierDirectSelection( int numBaseClf, int numWeakClf, cv::Size patchSz, const cv::Rect& sampleROI, bool useFeatureEx = false, int iterationInit = 0 );

	virtual ~StrongClassifierDirectSelection();

	void initBaseClassifiers(); // Initialize baseClassifiers

	bool update( const cv::Mat& image, int target, float importance = 1.0 );

	// evaluate the strongclassifier's confidence measure, sum( alpha_i * h_i )
	float eval( const cv::Mat& response );

	std::vector<int> getSelectedWeakClassifier();
	float classifySmooth( const std::vector<cv::Mat>& images, const cv::Rect& sampleROI, int& idx );
	int getNumBaseClassifier();
	cv::Size getPatchSize() const;
	cv::Rect getROI() const;
	bool getUseFeatureExchange() const;
	int getReplacedClassifier() const;

	void replaceWeakClassifier( int idx );
	int getSwappedClassifier() const;

private:
	// StrongClassifier

	int numBaseClassifier; // num of selectors
	BaseClassifier** baseClassifiers; // selected weakclassifiers which form strong classfier
	std::vector<float> alpha; // weight of baseclassifier
	
	int numWeakClassifier; // features pool or weakclassifiers pool
	int iterInit;
	int numAllWeakClassifier; // == numWeakClassifier + iterInit;
	
	cv::Size patchSize;

	bool useFeatureExchange;

	std::vector<bool> m_errorMask; // True if weakclassifier classify wrong
	std::vector<float> m_errors;
	std::vector<float> m_sumErrors;

	Detector* detector;
	cv::Rect ROI;

	int replacedClassifier; // feature pool中分类误差最大，将要被替换掉的弱分类器索引
	int swappedClassifier;	// 候选弱分类器中用来替换replacedClassifier的新弱分类器索引

};

/*********************************** BaseClassifier *******************************/
// BaseClassifier: 构成StrongClassifier的基分类器 / selector, （从numWeakClassifier + iterationInit个弱分类器或特征池中进行选择）

class BaseClassifier{

public:
	BaseClassifier( int numWeakClassifier, int iterationInit );

	// WeakClassifierHaarFeature: 弱分类器
	BaseClassifier( int numWeakClassifier, int iterationInit, WeakClassifierHaarFeature** weakCls );

	WeakClassifierHaarFeature** getReferenceWeakClassifier(){
		return weakClassifiers;
	}

	// ?
	void trainClassifier( const cv::Mat& image, int target, float importance, std::vector<bool>& errorMask );
	
	// select the classifier with the smallest error, and return the index
	int selectBestClassifier( std::vector<bool>& errorMask, float importance, std::vector<float>& errors );
	
	// compute the weak classifier which has the largest error, return it's index
	int computeReplaceWeakestClassifier( const std::vector<float>& errors );

	// replace the targetIndex's wWrong and wCorrect with sourceIndex's, and initialize sourceIndx's wCorrect = wWrong = 1
	void replaceClassifierStatistic( int sourceIndex, int targetIndex );

	int getIdxOfNewWeakClassifier(){
		return m_idxOfNewWeakClassifier;
	}

	// 返回该基分类器(使用当前选择的误差最小的弱分类器)判断sample的label, 1 或者 -1
	int eval( const cv::Mat& response_ );

	virtual ~BaseClassifier();
	 
	// calculate the given weak classifier's error:  e_{m} = m_wWrong[m] / ( m_wCorrect[m] + m_wWrong[m] )
	float getError( int curWeakClassifier );
	
	// calculate all weak classifiers' errors
	void getErrors( float* errors ); 

	int getSelectedClassifier() const;

	// 使用候选弱分类器替换weakClassifiers[index]
	void replaceWeakClassifier( int index );

protected:
	void generateRandomClassifier(); // 生成 numWeakClassifier + iterationInit 个弱分类器WeakClassifierHaarFeature

	WeakClassifierHaarFeature** weakClassifiers; // 弱分类器集合, feature pool

	bool m_referenceWeakClassifier;
	int m_numWeakClassifier; // number of weak classifiers
	int m_selectedClassifier;
	int m_idxOfNewWeakClassifier;	// == m_numWeakClassifier, 下一个添加进入weakClassifiers的弱分类器的索引

	std::vector<float> m_wCorrect; 	// 记录每个弱分类器到目前为止正确分类sample的权重
	std::vector<float> m_wWrong;	// 记录每个弱分类器到目前为止错误分类sample的权重
	int m_iterationInit;
};

/********************************** EstimatedGaussDistribution *******************************/
/**
   Gauss distribution (mean, sigma);
   Incremently estimate the mean and sigma by a Kalman filtering approach.
 */

class EstimatedGaussDistribution{

public:

	EstimatedGaussDistribution();
	EstimatedGaussDistribution( float P_mean, float R_mean, float P_sigma, float R_sigma );
	virtual ~EstimatedGaussDistribution();
	void update( float value );  //, float timeConstant = -1.0);
	float getMean();
	float getSigma();
	void setValues( float mean, float sigma );

 private:

	float m_mean;		// constant --> mean_t = mean_(t-1) + N(0, R)
	float m_P_mean; 	// P: estimate error covariance, for m_mean;
	float m_R_mean; 	// R: random noise for m_mean, ~N(0, R)

	float m_sigma;		// constant --> sigma^2_t = sigma^2_(t-1) + N(0, R)
	float m_P_sigma; 	// P: estimate error covariance, for m_sigma^2;
	float m_R_sigma; 	// R: random noise for m_sigma^2, ~N(0, R)
};

/********************************* WeakClassifierHaarFeature ***********************************/

//  弱分类器: 含有一个ClassifierThreshold,其中包含( posGauss + negGauss )

class WeakClassifierHaarFeature
{

public:
	WeakClassifierHaarFeature();
	virtual ~WeakClassifierHaarFeature();

	bool update( float value, int target );

	// return the sample's label, 1 or -1
	int eval( float value );

private:
	float sigma;
	float mean;
	ClassifierThreshold* m_thresholdclassifier; // == posGauss + negGauss

	// set the gauss distribution's initial mean and sigma
	void setInitialDistribution( EstimatedGaussDistribution* gauss );

	void generateRandomClassifier( EstimatedGaussDistribution* m_posSamples, EstimatedGaussDistribution* m_negSamples );

};

/********************************** Detector ******************************************/
class Detector{

public:
	Detector( StrongClassifierDirectSelection* classifier );
	virtual ~Detector();

	void classifySmooth( const std::vector<cv::Mat>& samples, float minMargin = 0 );

	int getNumDetections();
	float getConfidence( int patchIdx );
	float getConfidenceOfDetection( int detectionIdx );

	float getConfidenceOfBestDetection(){
		return m_maxConfidence;
	}
  
	int getPatchIdxOfBestDetection();

	int getPatchIdxOfDetection( int detectionIdx );

	const std::vector<int>& getIdxDetections() const{
		return m_idxDetections;
	}

	const std::vector<float>& getConfidences() const{
		return m_confidences;
	}

	const cv::Mat& getConfImageDisplay() const{
		return m_confImageDisplay;
	}

private:

	void prepareConfidencesMemory( int numPatches );
	void prepareDetectionsMemory( int numDetections );

	StrongClassifierDirectSelection* m_strongClassifier;

	/**
		patch: 估计目标当前位置x_t时，TrackerSamplerCS 在x_(t-1)附近采样得到的一系列patches
	 */
	int m_sizeConfidences; // == number of search patches
	std::vector<float> m_confidences; // Store the confidence measures of all search patches by the strongClassifier

	int m_numDetections; // confidence measure超过阈值minMargin的patch的数目
	int m_sizeDetections; // == m_numDetections
	std::vector<int> m_idxDetections; // confidence measure超过阈值minMargin的patch在所有patches中的indexes

	int m_idxBestDetection; // the sample index with largest confidence measure
	float m_maxConfidence;	// largest confidence measure of all search smaples

	cv::Mat_<float> m_confMatrix; // Store the strongClassifier's confidence measure for all search samples
	cv::Mat_<float> m_confMatrixSmooth; // m_confMatrix after GaussianBlur
	cv::Mat_<unsigned char> m_confImageDisplay; // // confidence value convert to [0, 255] for display
};

/********************************* ClassifierThreshold *************************************/
class ClassifierThreshold{
	// Calculate the hypotheses of a weakClassifier hj(X) by using a simple threshold

public:
	ClassifierThreshold( EstimatedGaussDistribution* m_posGauss, EstimatedGaussDistribution* m_negGauss );
	virtual ~ClassifierThreshold();

	/** 
		update the m_posGauss and m_negGauss with the value ( fj(X) ) and label first;
		update the m_threshold and m_parity second;
	*/
	void update( float value, int label );

	int eval( float value ); // return the sample's label, 1 or -1

	void* getDistribution( int lable ); // if label == 1, return the m_posGauss; otherwise m_negGauss

private:
	EstimatedGaussDistribution* m_posGauss; 	// N(mean_pos, sigma_pos), gauss distribution for positive labeled samples
	EstimatedGaussDistribution* m_negGauss;	// N(mean_neg, sigma_neg), gauss distribution for negative labeled samples

	float m_threshold;	// ( mu_pos + mu_neg ) / 2
	int m_parity;		// sign( mu_pos - mu_neg )
};

} /* namespace BOOSTING */

#endif