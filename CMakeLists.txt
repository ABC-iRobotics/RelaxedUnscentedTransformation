CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
cmake_policy(SET CMP0091 OLD) #to set mt/md via cmake_C/CXX_flags

PROJECT(RelaxedUT)

if (WIN32)
	## ---------------------------------------------------------------------------------
	## Use folders in IDEs supporting it
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "")
	## ---------------------------------------------------------------------------------
	## Set /MT or /MD compiler flags
	set(RUNTIME_LIBRARY_SETTINGS "/MT*" CACHE STRING "Code Generation: multi-threaded (/MT*) or multi threaded DLL (/MD*)")
	set_property(CACHE RUNTIME_LIBRARY_SETTINGS PROPERTY STRINGS "/MT*" "/MD*")
	set(CompilerFlags
		CMAKE_CXX_FLAGS
		CMAKE_CXX_FLAGS_DEBUG
		CMAKE_CXX_FLAGS_RELEASE
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		CMAKE_CXX_FLAGS_MINSIZEREL
		CMAKE_C_FLAGS
		CMAKE_C_FLAGS_DEBUG
		CMAKE_C_FLAGS_RELEASE
		CMAKE_C_FLAGS_RELWITHDEBINFO
		CMAKE_C_FLAGS_MINSIZEREL)
	set(EXTERNAL_FLAG_SETTER "")
	if (RUNTIME_LIBRARY_SETTINGS STREQUAL "/MT*")
		set(REPLACE_ARGS "/MD" "/MT")
		set(RUNTIME_INFO "mt")
	else()
		set(REPLACE_ARGS "/MT" "/MD")
		set(RUNTIME_INFO "md")
	endif()
	foreach(CompilerFlag ${CompilerFlags})
		string(REPLACE ${REPLACE_ARGS} ${CompilerFlag} "${${CompilerFlag}}")
		list(APPEND EXTERNAL_FLAG_SETTER -D${CompilerFlag}="${${CompilerFlag}}" ) #produces a list like -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
	endforeach()
	## ---------------------------------------------------------------------------------
	## Set configurations to be used
	set (CMAKE_CONFIGURATION_TYPES Debug;Release;MinSizeRel;RelWithDebInfo CACHE STRING "Configuration types" FORCE)
endif()

## ---------------------------------------------------------------------------------
## Set folders
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "" FORCE) #Nem convention instead of program files/SensorFusion...
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/Bin) #Or the program puts the application into different folders...

## ---------------------------------------------------------------------------------
## Draft settings - use SF_BUILD_DRAFT_API in the code for experimental API 
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  option(ENABLE_DRAFTS "Build and install draft classes and methods" ON)
else()
  option(ENABLE_DRAFTS "Build and install draft classes and methods" OFF)
endif()
if(ENABLE_DRAFTS)
  add_definitions(-DSF_BUILD_DRAFT_API)
endif()

## ---------------------------------------------------------------------------------
## Generally used variables for third parties
include(cmake/ThirdPartyHandling.cmake)
set(THIRD_PARTY_CONFIG Debug;Release)
set(ONLY_RELEASE Release)
set(GIT_NAMES GIT Git)
set(HG_NAMES HG Hg)

## ---------------------------------------------------------------------------------
## Find or get Eigen
set(EIGEN3_COMMAND "git clone https://gitlab.com/libeigen/eigen.git -b3.3.7")
set(EIGEN3_CACHE_SETTINGS -DBUILD_TESTING=FALSE;-DCMAKE_INSTALL_PREFIX="${3RDPARTY_INSTALL_PREFIX}";
	-DCMAKE_INSTALL_INCLUDEDIR="include";${EXTERNAL_FLAG_SETTER})
third_party_policy(Eigen3 ${EIGEN3_COMMAND} EIGEN3_CACHE_SETTINGS ONLY_RELEASE false GIT_NAMES)
macro(requires_eigen NAME)
	target_link_libraries (${NAME} Eigen3::Eigen)
endmacro(requires_eigen)

## ---------------------------------------------------------------------------------



add_subdirectory (cpptester)