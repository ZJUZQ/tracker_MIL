add_library( BOOSTING_DIRECTION SHARED
		directionAdaBoosting.cpp
		feature.cpp
		trackerAdaBoostingClassifier.cpp
		trackerFeature.cpp )

target_link_libraries( BOOSTING_DIRECTION ${THIRD_PARTY_LIBS} )