cmake_minimum_required(VERSION 2.8)
project(action)
find_package(OpenCV REQUIRED)
list(APPEND CMAKE_CXX_FLAGS "-Wno-deprecated -std=c++0x -fopenmp")
add_executable(action ActionAnalysis.cpp TrainAndTest.cpp DenseTrackStab.cpp FisherVector.cpp 
CmSaliencyGC.cpp CmGMM.cpp CmFile.cpp CmDefinition.cpp CmColorQua.cpp CmAPCluster.cpp)
target_link_libraries(action ${OpenCV_LIBS} libvl.so)