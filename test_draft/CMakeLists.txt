add_executable( hough hough.cpp )
target_link_libraries( hough ${THIRD_PARTY_LIBS})

add_executable( trackerPosture trackerPosture.cpp )
target_link_libraries( trackerPosture ${THIRD_PARTY_LIBS} BOOSTING BOOSTING_DIRECTION )

