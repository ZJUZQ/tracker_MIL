#include "MIL/trackerMILClassifier.hpp"

#define	 sign(s)  ( (s > 0) ? 1 : ( (s < 0) ? -1 : 0 ) )

template<class T> class SortableElementRev{
public:
	T 	_val;
	int _ind;
	SortableElementRev() : _ind( 0 ){

	}

	SortableElementRev( T val, int ind ){
		_val = val;
		_ind = ind;
	}

	bool operator <( SortableElementRev<T>& b ){
		return (_val < b._val );
	}
};

template<class T>
static bool CompareSortableElementRev( const SortableElementRev<T>& i, const SortableElementRev<T>& j ){
	return i._val < j._val;
}

template<class T>
void sort_order_ascending( std::vector<T>& v, std::vector<int>& order ) // 递增排序 Ascending sort
{
	uint n = (uint) v.size();
	std::vector< SortableElementRev<T> > v2;
	v2.resize( n );
	order.clear();
	order.resize( n );
	for( uint i = 0; i < n; i++ ){
		v2[i]._ind = i;
		v2[i]._val = v[i];
	}

	std::sort( v2.begin(), v2.end(), CompareSortableElementRev<T> );
	for( uint i = 0; i < n; i++ ){
		order[i] = v2[i]._ind;
		v[i] = v2[i]._val;
	}
}

namespace MIL
{
//----------------------------- ClfMilBoost -------------------------------//
/** implementations for strong classifier
*/
ClfMilBoost::Params::Params(){	// constructor of ClfMilBoost::Params
	_numSel = 50; // number of weak classifiers K, which composed of the strong classifier H
	_numFeat = 250; // a pool of M( > K) candidate weak stump classifiers
	_lRate = 0.85f;
}

ClfMilBoost::ClfMilBoost(){
	_myParams = ClfMilBoost::Params();
	_numsamples = 0;
}

ClfMilBoost::~ClfMilBoost(){
	_selectors.clear();
	for( size_t i = 0; i < _weakclf.size(); i++ )
		delete _weakclf.at( i );
}

void ClfMilBoost::init( const ClfMilBoost::Params& parameters ){
	_myParams = parameters;
	_numsamples = 0;

	//_ftrs = Ftr::generate( _myParams->_ftrParams, _myParams->_numFeat );
	// if( params->_storeFtrHistory )
	//  Ftr::toViz( _ftrs, "haarftrs" );

	_weakclf.resize( _myParams._numFeat );	// a pool of M( > K) candidate weak stump classifiers
	for( int k = 0; k < _myParams._numFeat; k++ ){
		_weakclf[k] = new ClfOnlineStump( k );
		_weakclf[k]->_lRate = _myParams._lRate;
	}
	_counter = 0;
}

void ClfMilBoost::update( const cv::Mat& posx, const cv::Mat& negx ){
	int numneg = negx.rows;
	int numpos = posx.rows;

	// compute ftrs
	//if( !posx.ftrsComputed() )
	//  Ftr::compute( posx, _ftrs );
	//if( !negx.ftrsComputed() )
	//  Ftr::compute( negx, _ftrs );

	// initialize strong classifier H
	static std::vector<float> Hpos, Hneg;
	Hpos.clear();
	Hneg.clear();
	Hpos.resize( posx.rows, 0.0f );
	Hneg.resize( negx.rows, 0.0f );

	_selectors.clear();
	std::vector<float> posw( posx.rows ), negw( negx.rows );
	std::vector< std::vector<float> > pospred( _weakclf.size() ), negpred( _weakclf.size() );

	// train all weak classifiers without weights
	for( int m = 0; m < _myParams._numFeat; m++ ){
		_weakclf[m]->update( posx, negx );
		pospred[m] = _weakclf[m]->classifySetF( posx );
		negpred[m] = _weakclf[m]->classifySetF( negx );
	}

	// pick the best features
	for( int s = 0; s < _myParams._numSel; s++ ){
		// compute errors/likl for all weak clfs
		std::vector<float> poslikl( _weakclf.size(), 1.0f ), neglikl( _weakclf.size() ), likl( _weakclf.size() );

		for( int w = 0; w < (int) _weakclf.size(); w++ ){
			float lll = 1.0f;
			for( int j = 0; j < numpos; j++ )
				lll *= ( 1 - sigmoid( Hpos[j] + pospred[w][j] ) );
			poslikl[w] = (float) -std::log( 1 - lll + 1e-5 ); // maximize the log likelihood L, 就是最小化 -L

			lll = 0.0f;
			for( int j = 0; j < numneg; j++ )
				lll += (float) -std::log( 1e-5f + 1 - sigmoid( Hneg[j] + negpred[w][j] ) );
			neglikl[w] = lll;

			likl[w] = poslikl[w] / numpos + neglikl[w] / numneg; // 正负样本平均分类 损失/错误
		}

		// pick best weak clf
		std::vector<int> order;
		sort_order_ascending<float>( likl, order ); // 对标签0--(M-1)的候选弱分类器，根据每个分类器的平均分类错误进行递增排序，order对应排序后的弱分类器的标签序列

		// find best weakclf that isn't already included
		/**
		  std::count (InputIterator first, InputIterator last, const T& val)
		     Returns the number of elements in the range [first,last) that compare equal to val.
		 */
		for( uint k = 0; k < order.size(); k++ ){
			if( std::count( _selectors.begin(), _selectors.end(), order[k] ) == 0 ){
				_selectors.push_back( order[k] ); 	// 根据每个候选弱分类器的标签判断是否已经添加过
                 break;                       		// _selectors 保存选择的弱分类器的标签
			}
		}

		// update H = H + h_m
		for( int k = 0; k < posx.rows; k++ )
			Hpos[k] += pospred[_selectors[s]][k]; // 对每个样本，计算已经选择出来的弱分类器在该样本上的某特征值的和

		for( int k = 0; k < negx.rows; k++ )
			Hneg[k] += negpred[_selectors[s]][k];	// 对每个样本，计算已经选择出来的弱分类器在该样本上的某特征值的和

	}

	//if( _myParams->_storeFtrHistory )
	//for ( uint j = 0; j < _selectors.size(); j++ )
	// _ftrHist( _selectors[j], _counter ) = 1.0f / ( j + 1 );

	_counter++;
	return;
}

std::vector<float> ClfMilBoost::classify( const cv::Mat& x, bool logR ){
	/* 矩阵x的列数表示特征(弱分类器数目)，矩阵x的行数表示样本数目；每一行表示一个样本对应的每个特征(Haar)的值 ? */

	int numsamples = x.rows;
	std::vector<float> res( numsamples );
	std::vector<float> tr;

	for( uint w = 0; w < _selectors.size(); w++ ){
		tr = _weakclf[_selectors[w]]->classifySetF( x );

		for(int j = 0; j < numsamples; j++)
			res[j] += tr[j];
	}

	// return probabilities or log odds ratio
	if( !logR ){
		for( int j = 0; j < (int) res.size(); j++ )
			res[j] = sigmoid( res[j] );
	}

	return res;
}



//----------------------------- ClfOnlineStump -------------------------------//
/** implementations for weak classifier
*/
ClfOnlineStump::ClfOnlineStump(){
	_trained = false;
	_ind = -1;
	init();
}

ClfOnlineStump::ClfOnlineStump( int ind ){
	_trained = false;
	_ind = ind;
	init();
}

void ClfOnlineStump::init(){
	_mu0 = 0;
	_mu1 = 0;
	_sig0 = 1;
	_sig1 = 1;
	_lRate = 0.85f;
	_trained = false;
}

void ClfOnlineStump::update( const cv::Mat& posx, const cv::Mat& negx, const cv::Mat_<float>& /*posw*/, const cv::Mat_<float>& /*negw*/){
	// std::cout << " ClfOnlineStump::update " << _ind << std::endl;
	float posmu = 0.0, negmu = 0.0;
	if( posx.cols > 0 )
		posmu = float( cv::mean( posx.col( _ind ) )[0] ); // The function cv::mean calculates the mean value M of array elements, independently for each channel, and return it: 
	if( negx.cols > 0 )
		negmu = float( cv::mean( negx.col( _ind ) )[0] );

	if( _trained ){
		if( posx.cols > 0 ){  // 有正样本，更新：_mu1, _sig1
			_mu1 = _lRate * _mu1 + ( 1 - _lRate ) * posmu;
			cv::Mat diff = posx.col( _ind ) - _mu1;
			_sig1 = _lRate * _sig1 + ( 1 - _lRate ) * float( cv::mean( diff.mul( diff ) )[0] ); // mul: Performs an element-wise multiplication
		} 
		if( negx.cols > 0 ){
			_mu0 = _lRate * _mu0 + ( 1 - _lRate ) * negmu;
			cv::Mat diff = negx.col( _ind ) - _mu0;
			_sig0 = _lRate * _sig0 + ( 1 - _lRate ) * float( cv::mean( diff.mul( diff ) )[0] );
		}

		_q = ( _mu1 - _mu0 ) / 2;
		_s = sign( _mu1 - _mu0 );
		_log_n0 = std::log( float( 1.0f / std::pow( _sig0, 0.5f ) ) );
		_log_n1 = std::log( float( 1.0f / std::pow( _sig1, 0.5f ) ) );
		
		//_e1 = -1.0f/(2.0f*_sig1+1e-99f);
    	//_e0 = -1.0f/(2.0f*_sig0+1e-99f);
    	_e0 = -1.0f / ( 2.0f * _sig0 + std::numeric_limits<float>::min() );	// template <class T> numeric_limits;
    	_e1 = -1.0f / ( 2.0f * _sig1 + std::numeric_limits<float>::min() );
	}
	else{	// _trained == false, 首次更新
		_trained = true;
		if( posx.cols > 0 ){
			_mu1 = posmu;	// 初始化均值
			cv::Scalar scal_mean, scal_std_dev;
			cv::meanStdDev( posx.col( _ind ), scal_mean, scal_std_dev );	// 计算数组元素的平均值和标准偏差
			_sig1 = std::pow( float( scal_std_dev[0] ), 2.0f ) + 1e-9f;	// 初始化标准方差 	
		}
		if( negx.cols > 0 ){
			_mu0 = negmu;
			cv::Scalar scal_mean, scal_std_dev;
			cv::meanStdDev( negx.col( _ind ), scal_mean, scal_std_dev );
			_sig0 = std::pow( float( scal_std_dev[0] ), 2.0f ) + 1e-9f;
		}

		_q = ( _mu1 - _mu0 ) / 2;
		_s = sign( _mu1 - _mu0 );
		_log_n0 = std::log( float( 1.0f / std::pow( _sig0, 0.5f ) ) );
		_log_n1 = std::log( float( 1.0f / std::pow( _sig1, 0.5f ) ) );
		
		//_e1 = -1.0f/(2.0f*_sig1+1e-99f);
    	//_e0 = -1.0f/(2.0f*_sig0+1e-99f);
    	_e0 = -1.0f / ( 2.0f * _sig0 + std::numeric_limits<float>::min() );	// template <class T> numeric_limits;
    	_e1 = -1.0f / ( 2.0f * _sig1 + std::numeric_limits<float>::min() );
	}
}

bool ClfOnlineStump::classify( const cv::Mat& x, int i ){
	//  return the log odds ratio: 布尔值
	float xx = x.at<float>(i, _ind);
	double log_p0 = ( xx - _mu0 ) * ( xx - _mu0 ) * _e0 + _log_n0;
	double log_p1 = ( xx - _mu1 ) * ( xx - _mu1 ) * _e1 + _log_n1;
	return log_p1 > log_p0;
}

float ClfOnlineStump::classifyF( const cv::Mat& x, int i ){
	//  return the log odds ratio: 浮点值
  	/* 矩阵x的列数表示特征(弱分类器数目)，矩阵x的行数表示样本数目；每一行表示一个样本对应的每个特征(Haar)的值 ? */

  	float xx = x.at<float>(i, _ind);
  	double log_p0 = ( xx - _mu0 ) * ( xx - _mu0 ) * _e0 + _log_n0;
	double log_p1 = ( xx - _mu1 ) * ( xx - _mu1 ) * _e1 + _log_n1;
	return float( log_p1 - log_p0 );
}

inline std::vector<float> ClfOnlineStump::classifySetF( const cv::Mat& x ){
	//  分类矩阵x中的所有特征矢量
	std::vector<float> res( x.rows );

	for(int k = 0; k < (int) res.size(); k++)
		res[k] = classifyF( x, k );

	return res;
}

} /* namespace MIL */