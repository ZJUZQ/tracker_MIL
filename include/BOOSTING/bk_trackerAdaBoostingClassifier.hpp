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

	void initBaseClassifier();

	bool update( const cv::Mat& image, int target, float importance = 1.0 ); // when a new sample comes, the initial importance == 1.0

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
	int numBaseClassifier;
	int numAllWeakClassifier;
	int numWeakClassifier;
	int iterInit;
	BaseClassifier** baseClassifier;
	std::vector<float> alpha;
	cv::Size patchSize;

	bool useFeatureExchange;

	// StrongClassifierDirectSelection
	std::vector<bool> m_errorMask;
	std::vector<float> m_errors;
	std::vector<float> m_sumErrors;

	Detector* detector;
	cv::Rect ROI;

	int replacedClassifier;
	int swappedClassifier;

};

/*********************************** BaseClassifier *******************************/
class BaseClassifier{

public:
	BaseClassifier( int numWeakClassifier, int iterationInit );
	BaseClassifier( int numWeakClassifier, int iterationInit, WeakClassifierHaarFeature** weakCls );

	WeakClassifierHaarFeature** getReferenceWeakClassifier(){
		return weakClassifier;
	}

	void trainClassifier( const cv::Mat& image, int target, float importance, std::vector<bool>& errorMask );
	int selectBestClassifier( std::vector<bool>& errorMask, float importance, std::vector<float>& errors );
	int computeReplaceWeakestClassifier( const std::vector<float>& errors );
	void replaceClassifierStatistic( int sourceIndex, int targetIndex );

	int getIdxOfNewWeakClassifier(){
		return m_idxOfNewWeakClassifier;
	}

	int eval( const cv::Mat& image );
	virtual ~BaseClassifier();
	float getError( int curWeakClassifier );
	void getErrors( float* errors );
	int getSelectedClassifier() const;
	void replaceWeakClassifier( int index );

protected:
	void generateRandomClassifier();
	WeakClassifierHaarFeature** weakClassifier;
	bool m_referenceWeakClassifier;
	int m_numWeakClassifier;
	int m_selectedClassifier;
	int m_idxOfNewWeakClassifier;
	std::vector<float> m_wCorrect; // 记录每个弱分类器到目前为止正确分类的数目
	std::vector<float> m_wWrong;
	int m_iterationInit;
};

/********************************** EstimatedGaussDistribution *******************************/
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

	float m_mean;
	float m_sigma;
	float m_P_mean; // positive
	float m_P_sigma;
	float m_R_mean; //
	float m_R_sigma;
};

/********************************* WeakClassifierHaarFeature ***********************************/
class WeakClassifierHaarFeature
{

public:
	WeakClassifierHaarFeature();
	virtual ~WeakClassifierHaarFeature();

	bool update( float value, int target );
	int eval( float value );

private:
  float sigma;
  float mean;
  ClassifierThreshold* m_classifier;

  void getInitialDistribution( EstimatedGaussDistribution *distribution );
  void generateRandomClassifier( EstimatedGaussDistribution* m_posSamples, EstimatedGaussDistribution* m_negSamples );

};

/********************************** Detector ******************************************/
class Detector{

public:
	Detector( StrongClassifierDirectSelection* classifier );
	virtual ~Detector();

	void classifySmooth( const std::vector<cv::Mat>& images, float minMargin = 0 );

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

	StrongClassifierDirectSelection* m_classifier;
	std::vector<float> m_confidences;
	int m_sizeConfidences;
	int m_numDetections;
	std::vector<int> m_idxDetections;
	int m_sizeDetections;
	int m_idxBestDetection;
	float m_maxConfidence;
	cv::Mat_<float> m_confMatrix;
	cv::Mat_<float> m_confMatrixSmooth;
	cv::Mat_<unsigned char> m_confImageDisplay;
};

/********************************* ClassifierThreshold *************************************/
class ClassifierThreshold{

public:
	ClassifierThreshold( EstimatedGaussDistribution* posSamples, EstimatedGaussDistribution* negSamples );
	virtual ~ClassifierThreshold();

	void update( float value, int target );
	int eval( float value );

	void* getDistribution( int target );

private:
	EstimatedGaussDistribution* m_posSamples;
	EstimatedGaussDistribution* m_negSamples;

	float m_threshold;
	int m_parity;
};

} /* namespace BOOSTING */

#endif