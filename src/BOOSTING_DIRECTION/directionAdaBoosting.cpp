#include "BOOSTING_DIRECTION/directionAdaBoosting.hpp"

namespace BOOSTING_DIRECTION{

/********************* directionAdaBoosting **********************************/

directionAdaBoosting::Params::Params(){
	numBaseClfs = 100; // number of base classifiers
	numWeakClfs = numBaseClfs * 10; // number of weak classifiers
	numAllWeakClfs = numWeakClfs + 5;
	patchSize = cv::Size( 30, 30 );
}

directionAdaBoosting::Params::Params( int numBaseClfs_, int numWeakClfs_, int numAllWeakClfs_, cv::Size patchSize_, bool useFeatureExchange_ ){
	numBaseClfs = numBaseClfs_; // number of base classifiers
	numWeakClfs = numWeakClfs_; // number of weak classifiers
	numAllWeakClfs = numAllWeakClfs_;
	patchSize = patchSize_;
	useFeatureExchange = useFeatureExchange_;
}

directionAdaBoosting::directionAdaBoosting(){
	isInit = false;
}

directionAdaBoosting::~directionAdaBoosting(){

}

/** imageT: tamplate image
	objectBB: boundingBox of the tracking object
 */
bool directionAdaBoosting::init( const cv::Mat& imageT, const cv::Rect2d& objectBB, Params params_ ){
	if( isInit )
		return false;

	if( imageT.empty() )
		return false;

    params = params_;

	// tracker feature
	TrackerFeatureHAAR::Params HAARparameters;
    HAARparameters.numFeatures = params.numAllWeakClfs;
    HAARparameters.isIntegral = true;
    HAARparameters.rectSize = params.patchSize;

    trackerFeature = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters ) );   

	// strong classifier
	strongClassifier = cv::Ptr<StrongClassifierDirectSelection>( 
            new StrongClassifierDirectSelection( params.numBaseClfs, params.numWeakClfs, params.patchSize, cv::Rect(), params.useFeatureExchange, ( params.numAllWeakClfs - params.numWeakClfs ) ) );
    // init base classifiers
    strongClassifier->initBaseClassifiers();


    // training the strong classifier
    cv::Point2f centerObject( objectBB.x + objectBB.width / 2.0, objectBB.y + objectBB.height / 2.0 ); // TODO: change with imgT

 	int numTrainingSamples = 900; // number of training samples
    for( int i = 0; i < numTrainingSamples; i++ ){ 
 
        // 将随机生成新的TrackerFeatureHAAR，用来替换每次用一个样本训练时挑出的最差featurehaar
        TrackerFeatureHAAR::Params HAARparameters2;
        HAARparameters2.numFeatures = 1;
        HAARparameters2.isIntegral = true;
        HAARparameters2.rectSize = params.patchSize;
        cv::Ptr<TrackerFeatureHAAR> trackerFeature2 = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters2 ) );
 
        std::vector<cv::Mat> Samples;
 
        int rotationDegree = std::rand() % 360; // rotation angle of imageT in degrees 
        if( abs( rotationDegree - 90 ) <= 2 || abs( rotationDegree - 270 ) <= 2 )
            continue;
 
        cv::Mat image;
        cv::Mat rotateMat = cv::getRotationMatrix2D( centerObject, rotationDegree, 1 ); // rotation cunter_clockwise
        cv::warpAffine( imageT, image, rotateMat, imageT.size() );
 
        cv::Mat_<int> intImage;
        cv::Mat_<double> intSqImage;
        cv::Mat image_;

        if( image.channels() == 1 )
        	image_ = image;
        else
        	cv::cvtColor( image, image_, CV_RGB2GRAY );

        cv::integral( image_, intImage, intSqImage, CV_32S );
 
 		// Add some noise tranlate to imporve the classifier's generalization ability
 		Samples.push_back( intImage( cv::Rect( objectBB.x + std::rand() % 7 - 3,
 											   objectBB.y + std::rand() % 7 - 3, 
 											   objectBB.width, objectBB.height ) ) );

        /** In image coordinate, the y-axis towns down, so :
        		when the object's direction is up, the angle is negtive, [-180, 0],
        		when the object's direction is down, the angle is postive, [0, 180]
         */
        int labelPos = -1; // robot turns up, angle [-180, 0]

        if( rotationDegree > 90 && rotationDegree < 270 ){
            labelPos = 1; 	// robot turns down, angle [0, 180]
        }
       
        cv::Mat response;
        trackerFeature->compute( Samples, response );
        strongClassifier->update( response.col(0), labelPos ); // for each training sample, update all weakclassifiers and strongClassifier, Algorithm 2.1
 
        int replacedWeakClassifier, swappedWeakClassifier;

        if( params.useFeatureExchange ){
            replacedWeakClassifier = strongClassifier->getReplacedClassifier(); // each traing sample will produce one bad weakclassifier to be replaced
            swappedWeakClassifier = strongClassifier->getSwappedClassifier();
            if( replacedWeakClassifier >= 0 && swappedWeakClassifier >= 0 )
                strongClassifier->replaceWeakClassifier( replacedWeakClassifier );
        }
        else{
            replacedWeakClassifier = -1;
            swappedWeakClassifier = -1;
        }
 
        /*  因为weakclassifier是基于TrackerFeatureHAAR的，因此，在交换了weakClassifierHaarFeature(没有实际实现Haar feature)后，
            还需要实际交换实际的TrackerFeatureHAAR，两者之间索引是一致的。 
        */
        if( replacedWeakClassifier != -1 && swappedWeakClassifier != -1 ){
            trackerFeature.staticCast<TrackerFeatureHAAR>()->swapFeature( replacedWeakClassifier, swappedWeakClassifier );
            trackerFeature.staticCast<TrackerFeatureHAAR>()->swapFeature( swappedWeakClassifier, trackerFeature2->getFeatureAt( 0 ) );
        }   
    }

	isInit = true;

	return true;
}


/** Update the strong classifeir with one sample
 */
bool directionAdaBoosting::updateWithOneSample( const cv::Mat& imgObject, const int labelUp ){

	if( imgObject.empty() )
		return false;

	// 将随机生成新的TrackerFeatureHAAR，用来替换每次用一个样本训练时挑出的最差featurehaar
	TrackerFeatureHAAR::Params HAARparameters2;
	HAARparameters2.numFeatures = 1;
	HAARparameters2.isIntegral = true;
	HAARparameters2.rectSize = params.patchSize;
	cv::Ptr<TrackerFeatureHAAR> trackerFeature2 = cv::Ptr<TrackerFeatureHAAR>( new TrackerFeatureHAAR( HAARparameters2 ) );

	std::vector<cv::Mat> Samples;

	cv::Mat_<int> intImage;
	cv::Mat_<double> intSqImage;
	cv::Mat image;

	if( imgObject.channels() == 1 )
		image = imgObject;
	else
		cv::cvtColor( imgObject, image, CV_RGB2GRAY );

	cv::integral( image, intImage, intSqImage, CV_32S );

	// Add some noise tranlate to imporve the classifier's generalization ability
	Samples.push_back( intImage );


	cv::Mat response;
    trackerFeature->compute( Samples, response );
    strongClassifier->update( response.col(0), labelUp ); // for each training sample, update all weakclassifiers and strongClassifier, Algorithm 2.1

    int replacedWeakClassifier, swappedWeakClassifier;

    if( params.useFeatureExchange ){
        replacedWeakClassifier = strongClassifier->getReplacedClassifier(); // each traing sample will produce one bad weakclassifier to be replaced
        swappedWeakClassifier = strongClassifier->getSwappedClassifier();
        if( replacedWeakClassifier >= 0 && swappedWeakClassifier >= 0 )
            strongClassifier->replaceWeakClassifier( replacedWeakClassifier );
    }
    else{
        replacedWeakClassifier = -1;
        swappedWeakClassifier = -1;
    }

    /*  因为weakclassifier是基于TrackerFeatureHAAR的，因此，在交换了weakClassifierHaarFeature(没有实际实现Haar feature)后，
        还需要实际交换实际的TrackerFeatureHAAR，两者之间索引是一致的。 
    */
    if( replacedWeakClassifier != -1 && swappedWeakClassifier != -1 ){
        trackerFeature.staticCast<TrackerFeatureHAAR>()->swapFeature( replacedWeakClassifier, swappedWeakClassifier );
        trackerFeature.staticCast<TrackerFeatureHAAR>()->swapFeature( swappedWeakClassifier, trackerFeature2->getFeatureAt( 0 ) );
    }  

    return true; 
}

/** compute the given sample's class: 1 for down and -1 for up
	 */
int directionAdaBoosting::classifierSample( const cv::Mat& sample){
	cv::Mat imgGray;            
    cv::Mat_<int> intImage;
    cv::Mat_<double> intSqImage;

    if( sample.channels() == 1 )
    	imgGray = sample;
    else
    	cv::cvtColor( sample, imgGray, CV_RGB2GRAY );	

    cv::integral( imgGray, intImage, intSqImage, CV_32S );

    std::vector<cv::Mat> samples;
    samples.push_back( intImage );

    cv::Mat response;
    trackerFeature->compute( samples, response );
    float conf = strongClassifier->eval( response.col( 0 ) );

    if( conf > 0 ) // labelPos == 1
    	return 1;
    else
    	return -1;
}

/** compute the given sample's confidence: poistive for down and negtive for up
 */
float directionAdaBoosting::evalSample( const cv::Mat& sample){
	cv::Mat imgGray;            
    cv::Mat_<int> intImage;
    cv::Mat_<double> intSqImage;

    if( sample.channels() == 1 )
    	imgGray = sample;
    else
    	cv::cvtColor( sample, imgGray, CV_RGB2GRAY );	

    cv::integral( imgGray, intImage, intSqImage, CV_32S );

    std::vector<cv::Mat> samples;
    samples.push_back( intImage );

    cv::Mat response;
    trackerFeature->compute( samples, response );
    float conf = strongClassifier->eval( response.col( 0 ) );

    return conf;

}



}; /* namespace BOOSTING_DIRECTION */