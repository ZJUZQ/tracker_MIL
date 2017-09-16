add_executable( hough hough.cpp )
target_link_libraries( hough ${THIRD_PARTY_LIBS})

add_executable( trackerPose_AdaBoosting_Hough trackerPose_AdaBoosting_Hough.cpp )
target_link_libraries( trackerPose_AdaBoosting_Hough BOOSTING)