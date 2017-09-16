#include "ML/TrainData.hpp"

namespace ML{

static const float MISSED_VAL = TrainData::missingValue();
static const int VAR_MISSED = VAR_ORDERED;

TrainData::~TrainData(){

}

cv::Mat TrainData::getTestSamples() const{
	cv::Mat idx = getTestSampleIdx();
	cv::Mat samples = getSamples();
	return idx.empty() ? cv::Mat() : getSubVector( samples, idx );
}

cv::Mat TrainData::getSubVector( const cv::Mat& vec, const cv::Mat& idx ){
	if( idx.empty() )
		return vec;
	
	int n = idx.checkVector( 1, CV_32S ); // returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1)
	int type = vec.type();
    CV_Assert( type == CV_32S || type == CV_32F || type == CV_64F );
    int dims = 1；
    int m; // lenth fo vec

    if( vec.cols == 1 || vec.rows == 1 ){
    	dims = 1;
    	m = vec.cols + vec.rows - 1;
    }
    else{
    	dims = vec.cols; //  如果vec不是行矢量或列矢量，那么默认vec是按行存储的矩阵
    	m = vec.rows;
    }

    cv::Mat subvec;

    if( vec.cols == m )
    	subvec.create( dims, n, type ); // vec是行矢量
    else
    	subvec.create( n, dims, type );

    int i, j;
    if( type == CV_32S ){
    	for( i = 0; i < n; i++ ){
    		int k = idx.at<int>(i);
    		CV_Assert( 0 <= k && k < m );
    		if( dims == 1 )
    			subvec.at<int>(i) = vec.at<int>(k);
    		else{
    			for( j = 0; j < dims; j++ )
    				subvec.at<int>(i, j) = vec.at<int>(k, j);
    		}
    	}
    }
    else if( type == CV_32F ){
    	for( i = 0; i < n; i++ ){
    		int k = idx.at<int>(i);
    		CV_Assert( 0 <= k && k < m );
    		if( dims == 1 )
    			subvec.at<float>(i) = vec.at<float>(k);
    		else{
    			for( j = 0; j < dims; j++ )
    				subvec.at<float>(i, j) = vec.at<float>(k, j);
    		}
    	}
    }
    else{
    	for( i = 0; i < n; i++ ){
            int k = idx.at<int>(i);
            CV_Assert( 0 <= k && k < m );
            if( dims == 1 )
                subvec.at<double>(i) = vec.at<double>(k);
            else
                for( j = 0; j < dims; j++ )
                    subvec.at<double>(i, j) = vec.at<double>(k, j);
        }
    }

    return subvec;
}

/******************************** TrainDataImpl ***********************************/

class TrainDataImpl : public TrainData{
public:
	typedef std::map<cv::String, int> MapType;

	TrainDataImpl(){
		file = 0;
		clear();
	}

	virtual ~TrainDataImpl(){
		cv::closeFile();
	}

	int getLayout() const{
	 	return layout; 
	}

    int getNSamples() const{
        return !sampleIdx.empty() ? (int)sampleIdx.total() :
               layout == ROW_SAMPLE ? samples.rows : samples.cols;
    }

    int getNTrainSamples() const{
        return !trainSampleIdx.empty() ? (int)trainSampleIdx.total() : getNSamples();
    }

    int getNTestSamples() const{
        return !testSampleIdx.empty() ? (int)testSampleIdx.total() : 0;
    }

    int getNVars() const{
        return !varIdx.empty() ? (int)varIdx.total() : getNAllVars();
    }

    int getNAllVars() const{
        return layout == ROW_SAMPLE ? samples.cols : samples.rows;
    }
}


}; /* namespace ML */