# Find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Define macro
function(pybind11_add_denoisor name)
    set(libs "")

    if(${ARGC} GREATER 1)
        math(EXPR last_index "${ARGC}-1")
        foreach(index RANGE 1 ${last_index})    
            set(libs ${libs} ${ARGV${index}})
        endforeach()
    endif()

    pybind11_add_module(
        ${name}
        MODULE ${name}.cpp)

    target_compile_features(
        ${name} 
        PRIVATE cxx_std_20
    )

    target_link_libraries(
        ${name}
        PRIVATE dv::processing 
                dv::toolkit
                dv::denoising
                ${libs}
    )

    if(EMLB_ENABLE_TORCH)
        target_link_libraries(
            ${name}
            PRIVATE ${TORCH_LIBRARIES}
        )
    endif()

    set_target_properties(
        ${name}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/modules/python
    )
endfunction()

# Register submodule
add_subdirectory(src)
