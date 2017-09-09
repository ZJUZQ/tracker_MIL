#ifndef MIL_FEATURE_HPP
#define MIL_FEATURE_HPP

#include "MIL/common_includes.hpp"

// #include <opencv2/persistence.hpp> // FileStorage

namespace MIL{

template<class Feature>
void _writeFeatures( const std::vector<Feature> features, cv::FileStorage &fs, const cv::Mat& featureMap )
{
	fs << "features" << "[";
	const cv::Mat_<int>& featureMap_ = (const cv::Mat_<int>&) featureMap;
	for ( int fi = 0; fi < featureMap.cols; fi++ )
		if( featureMap_( 0, fi ) >= 0 ){
			fs << "{";
			features[fi].write( fs );
			fs << "}";
		}
	fs << "]";
}

/*************************** CvParams *********************************/
class CvParams{
public:
	CvParams();
	virtual ~CvParams(){}

	// from|to file
	virtual void write( cv::FileStorage &fs ) const = 0;
	virtual bool read( const cv::FileNode &node ) = 0;
	// from|to screen
	virtual void printDefaults() const;
	virtual void printAttrs() const;
	virtual bool scanAttr( const std::string prmName, const std::string val );

	std::string name;
};



/*************************** CvFeatureParams **************************/
class CvFeatureParams : public CvParams{
public:
	enum{
		HAAR = 0,
		LBP = 1,
		HOG = 1
	};
	CvFeatureParams();
	virtual void init( const CvFeatureParams& fp );
	virtual void write( cv::FileStorage &fs ) const;
	virtual bool read( const cv::FileNode &node );
	static cv::Ptr<CvFeatureParams> create( int featureType );
	int maxCatCount; // 0 in case of numerical features
	int featSize;	 //  1 in case of simple features (HAAR, LBP) and N_BINS(9)*N_CELLS(4) in case of Dalal's HOG features
	int numFeatures;
};


/**************************** CvHaarFeatureParams ****************************/
class CvHaarFeatureParams : public CvFeatureParams{
public:
	CvHaarFeatureParams();

	virtual void init( const CvFeatureParams& fp );
	virtual void write( cv::FileStorage &fs ) const;
	virtual bool read( const cv::FileNode &node );

	virtual void printDefaults() const;
	virtual void printAttrs() const;
	virtual bool scanAttr( const std::string prm, const std::string val );

	bool isIntegral;
};



/*************************** CvFeatureEvaluator ***********************/
class CvFeatureEvaluator{
public:
	virtual ~CvFeatureEvaluator(){}
	virtual void init( const CvFeatureParams* _featureParams, int _maxSampleCount, cv::Size _winSzie );
	virtual void setImage( const cv::Mat& img, uchar clsLabel, int idx );
	virtual void writeFeatures( cv::FileStorage& fs, const cv::Mat& featureMap ) const = 0;
	virtual float operator() ( int featureIdx, int sampleIdx ) = 0;
	static cv::Ptr<CvFeatureEvaluator> create( int type );

	int getNumFeatures() const{
		return numFeatures;
	}

	int getMaxCatCount() const{
		return featureParams->maxCatCount;
	}

	int getFeatureSize() const{
		return featureParams->featSize;
	}

	const cv::Mat& getCls() const{
		return cls;
	}

	float getCls( int si ) const{
		return cls.at<float>( si, 0 );
	}

protected:
	virtual void generateFeatures() = 0;

	int npos, nneg;
	int numFeatures;
	cv::Size winSize;
	CvFeatureParams* featureParams;
	cv::Mat cls;
};


/*************************** CvHaarEvaluator ***************************/
class CvHaarEvaluator : public CvFeatureEvaluator{
public:
	class FeatureHaar{
	public:
		FeatureHaar( cv::Size patchSize );
		bool eval( const cv::Mat& image, cv::Rect ROI, float* result ) const;
		int getNumAreas();
		const std::vector<float>& getWeights() const;
		const std::vector<cv::Rect>& getAreas() const;
		void write( cv::FileStorage ) const{

		}

		float getInitMean() const;
		float getInitSigma() const;

	private:
		int m_type;
		int m_numAreas;
		std::vector<float> m_weights;
		float m_initMean;
		float m_initSigma;
		void generateRandomFeature( cv::Size imageSize );
		float getSum( const cv::Mat& image, cv::Rect imgROI ) const;
		std::vector<cv::Rect> m_areas; // areas within the patch over which to compute the feature
		cv::Size m_initSize; // size of the patch used during training
		cv::Size m_curSize;	 // size of the patches currently under investigation
		float m_scaleFactorHeight; // scaling factor in vertical direction
		float m_scaleFactorWidth;  // scaling factor in horizontal direction
		std::vector<cv::Rect> m_scaleAreas; // areas after scaling
		std::vector<float> m_scaleWeights; 	// weights after scaling
	};

	virtual void init( const CvFeatureParams* _featureParams, int _maxSampleCount, cv::Size _winSize );
	virtual void setImage( const cv::Mat& img, uchar clsLable = 0, int idx = 1 );
	virtual float operator() (int featureIdx, int sampleIdx);
	virtual void writeFeatures( cv::FileStorage& fs, const cv::Mat& featureMap ) const;
	void writeFeature( cv::FileStorage& fs ) const; // for old file format
	const std::vector<CvHaarEvaluator::FeatureHaar>& getFeatures() const;
	inline CvHaarEvaluator::FeatureHaar& getFeatures( int idx ){
		return features[idx];
	}
	void setWinSize( cv::Size patchSize );
	cv::Size setWinSize() const;
	virtual void generateFeatures();

	/**
	* TODO new method
	* \brief Overload the original generateFeatures in order to limit the number of the features
	* @param numFeatures Number of the features
	*/
	virtual void generateFeatures( int numFeatures );

protected:
	bool isIntegral;

	/* TODO Added from MIL implementation */
	cv::Mat _ii_img;
	void compute_integral( const cv::Mat& img, std::vector< cv::Mat_<float> >& ii_imgs ){
		cv::Mat ii_img;
		cv::integral( img, ii_img, CV_32F );	// Calculates the integral of an image
		cv::split( ii_img, ii_imgs );	// Divides a multi-channel array into several single-channel arrays
	}

	std::vector<FeatureHaar> features;
	cv::Mat sum; /* sum images (each row represents image) */
};


} /* namespace MIL */


#endif