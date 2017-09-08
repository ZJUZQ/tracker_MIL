#ifndef MIL_TRACKERMILCLASSIFIER_HPP
#define MIL_TRACKERMILCLASSIFIER_HPP

#include "MIL/common_includes.hpp"

namespace MIL
{
//! @addtogroup tracking
//! @{

//TODO based on the original implementation
//http://vision.ucsd.edu/~bbabenko/project_miltrack.html

class ClfOnlineStump;

class ClfMilBoost{
public:
	struct Params{
		Params();
		int _numSel;
		int _numFeat;
		float _lRate;
	};

	ClfMilBoost();
	~ClfMilBoost();
	void init( const ClfMilBoost::Params& parameters = ClfMilBoost::Params() );
	void update( const cv::Mat& posx, const cv::Mat& negx );
	std::vector<float> classify( const cv::Mat& x, bool logR = true );

	inline float sigmoid( float x ){ // the sigmoid function
		return 1.0f / ( 1.0f + std::exp( -x ) );
	}

private:
	uint _numsamples;
	ClfMilBoost::Params _myParams;
	std::vector<int> _selectors; // 组成强分类器的K个弱分类器序列，由候选的M个弱分类器的标签组成
	std::vector<ClfOnlineStump*> _weakclf;
	uint _counter;	// 算法(onlineBoost)累加计算次数

};

class ClfOnlineStump{
public:
	float _mu0, _mu1, _sig0, _sig1;
	float _q;
	int _s;
	float _log_n0, _log_n1;
	float _e0, _e1;
	float _lRate;	// learn rate

	ClfOnlineStump();
	ClfOnlineStump( int ind );
	void init();
	void update( const cv::Mat& posx, const cv::Mat& negx, const cv::Mat_<float>& posw = cv::Mat_<float>(), const cv::Mat_<float>& negw = cv::Mat_<float>() );
	bool classify( const cv::Mat& x, int i );
	float classifyF( const cv::Mat& x, int i );
	std::vector<float> classifySetF( const cv::Mat& x );

private:
	bool _trained;
	int _ind;
};

} /* namespace MIL */



#endif