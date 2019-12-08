cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(Outsize)

find_package(Boost REQUIRED)

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



add_executable (WashingMachine washing_machine.cpp classes/findTarget.cpp classes/entropyOut.cpp classes/pointposeOut.cpp classes/wmSegmentation.cpp) 
target_link_libraries (WashingMachine ${PCL_LIBRARIES})
target_link_libraries(WashingMachine ${Boost_LIBRARIES})

add_executable (Bin bin.cpp classes/binSegmentation.cpp classes/entropyOut.cpp classes/pointposeOut.cpp) 
target_link_libraries (Bin ${PCL_LIBRARIES})
target_link_libraries(Bin ${Boost_LIBRARIES})


add_executable (InsideDrum inside_drum.cpp classes/entropy.cpp classes/pointpose.cpp classes/segmentation.cpp)
target_link_libraries (InsideDrum ${PCL_LIBRARIES})
target_link_libraries(InsideDrum ${Boost_LIBRARIES})

add_executable (Paddles paddles.cpp classes/findTarget.cpp classes/drumModel.cpp)
target_link_libraries (Paddles ${PCL_LIBRARIES})
target_link_libraries(Paddles ${Boost_LIBRARIES})























