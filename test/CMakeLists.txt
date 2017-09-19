add_executable( tracker_AdaBoosting tracker_AdaBoosting.cpp )
target_link_libraries( tracker_AdaBoosting BOOSTING)

add_executable( tracker_MILBoosting tracker_MILBoosting.cpp )
target_link_libraries( tracker_MILBoosting MIL)

add_executable( trackerPosture_AdaBoosting trackerPosture_AdaBoosting.cpp )
target_link_libraries( trackerPosture_AdaBoosting BOOSTING)
