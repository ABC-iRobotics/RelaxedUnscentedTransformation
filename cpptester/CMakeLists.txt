cmake_minimum_required(VERSION 3.11)
PROJECT(RelaxedUT)

## ---------------------------------------------------------------------------------
## Set folders
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "" FORCE) #Nem convention instead of program files/SensorFusion...
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/Bin) #Or the program puts the application into different folders...

## ---------------------------------------------------------------------------------
## TODO clean this part...
set (CMAKE_CXX_STANDARD 11)

## ---------------------------------------------------------------------------------
## Find Eigen3
find_package(Eigen3 REQUIRED)

## ---------------------------------------------------------------------------------
## Get the content of subfolders

# Add all header and cpp files in the directory to the project
set (HEADERS
	../cppsource/index_selector.h
	../cppsource/UT.h
	../cppsource/SelUT.h
	../cppsource/RelaxedUT.h
	../cppsource/PartialCholevski.h
	)
	
set (SOURCES
	../cppsource/index_selector.cpp
	../cppsource/RelaxedUT.cpp
	../cppsource/PartialCholevski.cpp
	)

# Add the dinamic library target to the project
add_library (lib_relaxedUT STATIC ${SOURCES} ${HEADERS} )

target_link_libraries (lib_relaxedUT Eigen3::Eigen)

## Macro to setup archive targets

set_target_properties(lib_relaxedUT PROPERTIES IMPORTED_LOCATION lib)
if(WIN32)
	set_target_properties(lib_relaxedUT PROPERTIES
		RELEASE_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}"
		RELWITHDEBINFO_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}-relwithdebinfo"
		MINSIZEREL_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}-minsizerel"
		DEBUG_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}-debug")
endif()
target_include_directories(lib_relaxedUT INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cppsource/>)

add_executable(example_SLAM AUT.h RelaxedAUT.h SLAMmodels.cpp SLAMmodels.h
		Simulator.h Simulator.cpp readData.cpp readData.h SLAM_example.cpp)
target_link_libraries(example_SLAM lib_relaxedUT)
target_compile_definitions(example_SLAM
			PUBLIC DATASET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/")

add_executable(example_precision lin_example.cpp)
target_link_libraries(example_precision lib_relaxedUT)

add_executable(example_Norder Norder_example.cpp RMSE.cpp RMSE.h)
target_link_libraries(example_Norder lib_relaxedUT)

add_executable(example_nonlinear nonlinear_example.cpp)
target_link_libraries(example_nonlinear lib_relaxedUT)

