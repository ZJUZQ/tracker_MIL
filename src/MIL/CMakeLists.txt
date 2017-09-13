add_library( MIL SHARED
		feature.cpp
		trackerFeature.cpp
		trackerFeatureSet.cpp
		trackerMIL.cpp
		trackerMILClassifier.cpp
		trackerMILModel.cpp
		trackerModel.cpp
		trackerSampler.cpp
		trackerSamplerAlgorithm.cpp
		trackerStateEstimator.cpp
												)

target_link_libraries( MIL ${THIRD_PARTY_LIBS} )