# # Compile module to dv runtime
# if (dv_FOUND)
#     add_library(m_template SHARED _module_dv.cpp)
#     set_target_properties(m_template PROPERTIES PREFIX "K")
#     target_link_libraries(m_template PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)
#     install(TARGETS m_template DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
# endif()

# # Convert module to python package
# pybind11_add_module(template _module_py.cpp)
# target_link_libraries(template PUBLIC ${BLAS_LIBRARIES} dv::sdk pybind11::module)