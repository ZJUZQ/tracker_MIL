add_library( MIL SHARED
		feature.cpp
		#roiSelector.cpp
		trackerFeature.cpp
		trackerMIL.cpp
		trackerMILClassifier.cpp
		trackerMILModel.cpp
		trackerModel.cpp
		trackerSampler.cpp
		trackerSamplerAlgorithm.cpp
		trackerStateEstimator.cpp
		#trackerStateEstimatorMILBoosting.cpp 
												)

target_link_libraries( MIL ${THIRD_PARTY_LIBS} )