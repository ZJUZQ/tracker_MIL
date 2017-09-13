add_library( BOOSTING SHARED
		feature.cpp
		trackerAdaBoosting.cpp
		trackerAdaBoostingClassifier.cpp
		trackerAdaBoostingModel.cpp
		trackerFeature.cpp
		trackerFeatureSet.cpp
		trackerModel.cpp
		trackerSampler.cpp
		trackerSamplerAlgorithm.cpp
		trackerStateEstimator.cpp
												)

target_link_libraries( BOOSTING ${THIRD_PARTY_LIBS} )