#ifndef ML_DATA_HPP
#define ML_DATA_HPP

#include "ML/common_includes.hpp"

namespace ML{

/** @brief Sample types */
enum SampleTypes
{
    ROW_SAMPLE = 0, //!< each training sample is a row of samples
    COL_SAMPLE = 1  //!< each training sample occupies a column of samples
};

enum VariableTypes
{
    VAR_NUMERICAL    =0, //!< same as VAR_ORDERED
    VAR_ORDERED      =0, //!< ordered variables
    VAR_CATEGORICAL  =1  //!< categorical variables, 分类变量
};

/** @brief Class encapsulating training data.

Please note that the class only specifies the interface of training data, but not implementation.
All the statistical model classes in _ml_ module accepts Ptr\<TrainData\> as parameter. In other
words, you can create your own class derived from TrainData and pass smart pointer to the instance
of this class into StatModel::train.

@sa @ref ml_intro_data
 */
class TrainData{
public:
	static inline float missingValue() { return FLT_MAX; }
	virtual ~TrainData();

	virtual int getLayout() const = 0;
	virtual int getNTrainSamples() const = 0;
	virtual int getNTestSamples() const = 0;
	virtual int getNSamples() const = 0;
	virtual int getNVars() const = 0;
	virtual int getNAllVars() const = 0;

	virtual void getSample( cv::InputArray varIdx, int sidx, float* buf ) const = 0;
	virtual cv::Mat getSamples() const = 0;
	virtual cv::Mat getMissing() const = 0;


	/** @brief Returns matrix of train samples

    @param layout The requested layout. If it's different from the initial one, the matrix is
        transposed. See ml::SampleTypes.
    @param compressSamples if true, the function returns only the training samples (specified by
        sampleIdx)
    @param compressVars if true, the function returns the shorter training samples, containing only
        the active variables.

    In current implementation the function tries to avoid physical data copying and returns the
    matrix stored inside TrainData (unless the transposition or compression is needed).
     */
	virtual cv::Mat getTrainSamples( int layout = ROW_SAMPLE, bool compressSamples = true, bool compressVars = true ) const = 0;


	/** @brief Returns the vector of responses

    The function returns ordered or the original categorical responses. Usually it's used in
    regression algorithms.
     */
	virtual cv::Mat getTrainResponses() const = 0;


	/** @brief Returns the vector of normalized categorical responses, 归一化分类响应

    The function returns vector of responses. Each response is integer from `0` to `<number of
    classes>-1`. The actual label value can be retrieved then from the class label vector, see
    TrainData::getClassLabels.
     */
	virtual cv::Mat getTrainNormCatResponses() const = 0;

	CV_WRAP virtual cv::Mat getTestResponses() const = 0;
    CV_WRAP virtual cv::Mat getTestNormCatResponses() const = 0;
    CV_WRAP virtual cv::Mat getResponses() const = 0;
    CV_WRAP virtual cv::Mat getNormCatResponses() const = 0;
    CV_WRAP virtual cv::Mat getSampleWeights() const = 0;
    CV_WRAP virtual cv::Mat getTrainSampleWeights() const = 0;
    CV_WRAP virtual cv::Mat getTestSampleWeights() const = 0;
    CV_WRAP virtual cv::Mat getVarIdx() const = 0;
    CV_WRAP virtual cv::Mat getVarType() const = 0;
    CV_WRAP cv::Mat getVarSymbolFlags() const;
    CV_WRAP virtual int getResponseType() const = 0;
    CV_WRAP virtual cv::Mat getTrainSampleIdx() const = 0;
    CV_WRAP virtual cv::Mat getTestSampleIdx() const = 0;
    CV_WRAP virtual void getValues(int vi, cv::InputArray sidx, float* values) const = 0;
    virtual void getNormCatValues(int vi, cv::InputArray sidx, int* values) const = 0;
    CV_WRAP virtual cv::Mat getDefaultSubstValues() const = 0;

    CV_WRAP virtual int getCatCount(int vi) const = 0;


    /** @brief Returns the vector of class labels

    The function returns vector of unique labels occurred in the responses.
     */
    CV_WRAP virtual cv::Mat getClassLabels() const = 0;

    CV_WRAP virtual cv::Mat getCatOfs() const = 0;
    CV_WRAP virtual cv::Mat getCatMap() const = 0;


    /** @brief Splits the training data into the training and test parts
    @sa TrainData::setTrainTestSplitRatio
     */
    CV_WRAP virtual void setTrainTestSplit( int count, bool shuffle = true ) = 0;


    /** @brief Splits the training data into the training and test parts

    The function selects a subset of specified relative size and then returns it as the training
    set. If the function is not called, all the data is used for training. Please, note that for
    each of TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test
    subset can be retrieved and processed as well.
    @sa TrainData::setTrainTestSplit
     */
    CV_WRAP virtual void setTrainTestSplitRatio(double ratio, bool shuffle=true) = 0;
    CV_WRAP virtual void shuffleTrainTest() = 0;


    /** @brief Returns matrix of test samples */
    CV_WRAP cv::Mat getTestSamples() const;

    /** @brief Returns vector of symbolic names captured in loadFromCSV() */
    CV_WRAP void getNames(std::vector<cv::String>& names) const;

    CV_WRAP static cv::Mat getSubVector( const cv::Mat& vec, const cv::Mat& idx );


    /** @brief Reads the dataset from a .csv file and returns the ready-to-use training data.

    @param filename The input file name
    @param headerLineCount The number of lines in the beginning to skip; besides the header, the
        function also skips empty lines and lines staring with `#`
    @param responseStartIdx Index of the first output variable. If -1, the function considers the
        last variable as the response
    @param responseEndIdx Index of the last output variable + 1. If -1, then there is single
        response variable at responseStartIdx.
    @param varTypeSpec The optional text string that specifies the variables' types. It has the
        format `ord[n1-n2,n3,n4-n5,...]cat[n6,n7-n8,...]`. That is, variables from `n1 to n2`
        (inclusive range), `n3`, `n4 to n5` ... are considered ordered and `n6`, `n7 to n8` ... are
        considered as categorical. The range `[n1..n2] + [n3] + [n4..n5] + ... + [n6] + [n7..n8]`
        should cover all the variables. If varTypeSpec is not specified, then algorithm uses the
        following rules:
        - all input variables are considered ordered by default. If some column contains has non-
          numerical values, e.g. 'apple', 'pear', 'apple', 'apple', 'mango', the corresponding
          variable is considered categorical.
        - if there are several output variables, they are all considered as ordered. Error is
          reported when non-numerical values are used.
        - if there is a single output variable, then if its values are non-numerical or are all
          integers, then it's considered categorical. Otherwise, it's considered ordered.
    @param delimiter The character used to separate values in each line.
    @param missch The character used to specify missing measurements. It should not be a digit.
        Although it's a non-numerical value, it surely does not affect the decision of whether the
        variable ordered or categorical.
    @note If the dataset only contains input variables and no responses, use responseStartIdx = -2
        and responseEndIdx = 0. The output variables vector will just contain zeros.
     */
    static cv::Ptr<TrainData> loadFromCSV( 	const cv::String& filename,
										int headerLineCount,
										int responseStartIdx=-1,
										int responseEndIdx=-1,
										const cv::String& varTypeSpec = cv::String(),
										char delimiter=',',
										char missch='?' );


    /** @brief Creates training data from in-memory arrays.

    @param samples matrix of samples. It should have CV_32F type.
    @param layout see ml::SampleTypes.
    @param responses matrix of responses. If the responses are scalar, they should be stored as a
        single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
        former case the responses are considered as ordered by default; in the latter case - as
        categorical)
    @param varIdx vector specifying which variables to use for training. It can be an integer vector
        (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
        active variables.
    @param sampleIdx vector specifying which samples to use for training. It can be an integer
        vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
        of training samples.
    @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
    @param varType optional vector of type CV_8U and size `<number_of_variables_in_samples> +
        <number_of_variables_in_responses>`, containing types of each input and output variable. See
        ml::VariableTypes.
     */
    CV_WRAP static cv::Ptr<TrainData> create( cv::InputArray samples, 
    										  int layout, 
    										  cv::InputArray responses,
                                 			  cv::InputArray varIdx = cv::noArray(), 
                                 			  cv::InputArray sampleIdx = cv::noArray(),
                                 			  cv::InputArray sampleWeights = cv::noArray(), 
                                 			  cv::InputArray varType = cv::noArray() );


};


}; /* namespace ML */

#endif