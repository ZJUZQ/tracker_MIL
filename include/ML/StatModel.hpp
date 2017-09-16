#ifndef ML_STATMODEL_HPP
#define ML_STATMODEL_HPP

#include "ML/common_includes.hpp"

namespace ML{

/** @brief Base class for statistical models in ML.
 */
class StatModel : public Algorithm{
public:
	enum Flags{
		UPDATE_MODEL = 1,
		RAW_OUTPUT = 1, //!< makes the method return the raw results (the sum), not the class label
		COMPRESSED_INPUT = 2,
		PREPROCESSED_INPUT = 4
	};

	/** @brief Returns the number of variables in training samples */
	virtual int getVarCount() const = 0;

	virtual bool empty() const;

	/** @brief Returns true if the model is trained */
	virtual bool isTrainded() const = 0;

	/** @brief Returns true if the model is classifier */
	virtual bool isClassifier() const = 0;


	/** @brief Trains the statistical model
    @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
        created with TrainData::create.
    @param flags optional flags, depending on the model. Some of the models can be updated with the
        new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
     */
	virtual bool train( const cv::Ptr<TrainData>& trainData, int flags = 0 );

	/** @brief Trains the statistical model
    @param samples training samples
    @param layout See ml::SampleTypes.
    @param responses vector of responses associated with the training samples.
    */
	virtual bool train( cv::InputArray samples, int layout, cv::InputArray responses );

	/** @brief Computes error on the training or test dataset
    @param data the training data
    @param test if true, the error is computed over the test subset of the data, otherwise it's
        computed over the training subset of the data. Please note that if you loaded a completely
        different dataset to evaluate already trained classifier, you will probably want not to set
        the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so
        that the error is computed for the whole new set. Yes, this sounds a bit confusing.
    @param resp the optional output responses.

    The method uses StatModel::predict to compute the error. For regression models the error is
    computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
     */
	virtual float calcError( const cv::Ptr<TrainData>& data, bool test, cv::OutputArray resp ) const;

	/** @brief Predicts response(s) for the provided sample(s)
    @param samples The input samples, floating-point matrix
    @param results The optional output matrix of results.
    @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
     */
	virtual float predict( cv::InputArray samples, cv::OutputArray results = cv::noArray(), int flags = 0 ) const = 0;


	/** @brief Create and train model with default parameters

    The class must implement static `create()` method with no parameters or with all default parameter values
    */
    template<typename _Tp> 
    static cv::Ptr<_Tp> train( const cv::Ptr<TrainData>& data, int flags = 0 )
    {
        cv::Ptr<_Tp> model = _Tp::create();
        return !model.empty() && model->train(data, flags) ? model : cv::Ptr<_Tp>();
    }

};

}; /* namespace ML */


#endif