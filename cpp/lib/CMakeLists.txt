# PROJECT(SensorFusion)

# Add all header and cpp files in the directory to the project
set (HEADERS
	index_selector.h
	UT.h
	RelUT.h
	RelAUT.h
	AUT.h
	PartialCholevski.h
	)
	
set (SOURCES
	index_selector.cpp
	RelUT.cpp
	PartialCholevski.cpp
	)

# Add the dinamic library target to the project
add_library (lib_relaxedUT STATIC ${SOURCES} ${HEADERS} )

target_link_libraries (lib_relaxedUT Eigen3::Eigen)

## Macro to setup archive targets
macro(install_lib NAME HEADERS)
	set(LIBS_TO_INSTALL ${LIBS_TO_INSTALL} "${NAME}" PARENT_SCOPE) #for configuring Config.cmake in installing.cmake
	set_target_properties(${NAME} PROPERTIES IMPORTED_LOCATION lib)
	if(WIN32)
		set_target_properties(${NAME} PROPERTIES
			RELEASE_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}"
			RELWITHDEBINFO_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}-relwithdebinfo"
			MINSIZEREL_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}-minsizerel"
			DEBUG_POSTFIX "-${CMAKE_VS_PLATFORM_TOOLSET}-${CMAKE_GENERATOR_PLATFORM}-${RUNTIME_INFO}-debug")
	endif()
	target_include_directories(${NAME} INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
		$<INSTALL_INTERFACE:include>)
	install(TARGETS ${NAME} EXPORT ${export_targets}
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		RUNTIME DESTINATION bin
		INCLUDES DESTINATION include)
	install(FILES ${HEADERS} DESTINATION include)
endmacro(install_lib)

install_lib(lib_relaxedUT "${HEADERS}")

