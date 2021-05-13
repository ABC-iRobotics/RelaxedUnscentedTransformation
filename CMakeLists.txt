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
## TODO clean this part...
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
    #This is required to find sodium package for libzmq linking
    #set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()
# if(NOT LINUX) should work, too, if you need that
if(LINUX) 
	message(STATUS ">>> Linux")
	set (CMAKE_CXX_STANDARD 14)
elseif(WIN32)
        # for Windows operating system in general
        # on *nixes we need this to use C++14
        if (MSVC_VERSION GREATER_EQUAL "1900")
            include(CheckCXXCompilerFlag)
            CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
            if (_cpp_latest_flag_supported)
                add_compile_options("/std:c++latest")
            endif()
        endif()
endif()

## ---------------------------------------------------------------------------------
## Find or get Eigen
set (CMAKE_3RDPARTY_DIR "${CMAKE_BINARY_DIR}/ThirdParties")
if (NOT Eigen3_FOUND)
	find_package(Eigen3 QUIET)
	if (NOT Eigen3_FOUND)
		message(WARNING "Eigen3 Library is not found! Set its path or choose a download policy")
		set(Eigen_Policy "Choose" CACHE STRING "Where does Eigen must be installed?")
		set_property(CACHE Policy_${NAME} PROPERTY STRINGS "Choose" "Download and install into the default path" "Download and install into the replacable package")
		if (Eigen_Policy STREQUAL  "Download and install into the default path")
			EXEC_PROGRAM( install_eigen_submodule "${SRC_DIR}" ARGS "${CMAKE_3RDPARTY_DIR}")
			find_package(${NAME} REQUIRED)
		endif()
		if (Eigen_Policy STREQUAL  "Download and install into the replacable package")
			set (3RDPARTY_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/ThirdParties")
			EXEC_PROGRAM( install_eigen_submodule "${SRC_DIR}" ARGS "${CMAKE_3RDPARTY_DIR}" "${3RDPARTY_INSTALL_PREFIX}")
			find_package(${NAME} REQUIRED PATHS ${3RDPARTY_INSTALL_PREFIX} NO_DEFAULT_PATH )
		endif()
	endif()
endif()

## ---------------------------------------------------------------------------------
## Names used in the CMake package
set(export_config_name "RelaxedUT")
set(export_targets ${export_config_name}Targets)
if(WIN32)
	set(export_targets_name ${export_config_name}Targets-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO})
else()
	set(export_targets_name ${export_config_name}Targets)
endif()

## ---------------------------------------------------------------------------------
## Get the content of subfolders
add_subdirectory (cpp/lib)


















## ---------------------------------------------------------------------------------
## Additional install methods
install(EXPORT ${export_targets}
  FILE ${export_targets_name}.cmake
  DESTINATION "cmake"
  NAMESPACE ${export_config_name}::)
# Because git version cannot be returned in build-time, the install goes via install(CODE *)...
if(WIN32)
	set(ConfigInput "ConfigWindows.cmake.in")
else()
	set(ConfigInput "ConfigLinux.cmake.in")
endif()
install(CODE "
	set(LIBS_TO_INSTALL ${LIBS_TO_INSTALL})
	set(RUNTIME_INFO ${RUNTIME_INFO})
	set(export_config_name ${export_config_name})
	set(export_targets ${export_targets})

	configure_file(\"${CMAKE_CURRENT_LIST_DIR}/cmake/${ConfigInput}\"
	\"${CMAKE_INSTALL_PREFIX}/${export_config_name}Config.cmake\" @ONLY)
")

## ---------------------------------------------------------------------------------
## To generate doxygen documentation automatically during install
#option(WITH_DOCS "To generate documentation into the package during install (neeeds doxygen)" false)
#if(WITH_DOCS)
#	configure_file("${CMAKE_CURRENT_LIST_DIR}/cmake/docs/Doxyfile.in" "${CMAKE_BINARY_DIR}/Doxyfile" @ONLY)
#	find_program(DOXYGEN doxygen)
#	install(CODE "EXEC_PROGRAM(doxygen \"${CMAKE_BINARY_DIR}/Doxyfile\")
#		file(COPY \"${CMAKE_SOURCE_DIR}/cmake/docs/index.html\" DESTINATION \"${CMAKE_INSTALL_PREFIX}/Docs\" )
#	" )
#endif()


## ---------------------------------------------------------------------------------
## Tests
#option(BUILD_TESTING "To build the unit tests" TRUE)
#if (BUILD_TESTING)
	#enable_testing()
	#add_subdirectory(tests)

#endif()

#readme: tested on:
#versions, versions of third parties...

#TODO: update package title page, pictures...