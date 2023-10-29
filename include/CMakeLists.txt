# Create an interface library
add_library(denoising INTERFACE)
add_library(dv::denoising ALIAS denoising)

# Specifies include directories
target_include_directories(
    denoising
    INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
	denoising
	INTERFACE ${OpenCV_LIBS}
			  ${BLAS_LIBRARIES}
			  fmt::fmt
			  Boost::boost
			  Eigen3::Eigen
			  dv::processing
)

if(ENABLE_TORCH)
	target_link_libraries(
		denoising
		INTERFACE ${TORCH_LIBRARIES}
	)
endif()

target_compile_features(denoising INTERFACE cxx_std_20)
