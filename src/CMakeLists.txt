# Define macro
function(register_module name)
    set(libs "")

    if(${ARGC} GREATER 1)
        math(EXPR last_index "${ARGC}-1")
        foreach(index RANGE 1 ${last_index})    
            set(libs ${libs} ${ARGV${index}})
        endforeach()
    endif()

    add_library(
        m_${name} 
        SHARED ${name}.cpp)

    set_target_properties(
        m_${name}
        PROPERTIES PREFIX "K")

    target_link_libraries(
        m_${name}
        PRIVATE dv::denoising
                dv::sdk
                ${libs}
    )

    if(EMLB_ENABLE_TORCH)
        target_link_libraries(
            m_${name}
            PRIVATE ${TORCH_LIBRARIES}
        )
    endif()

    install(
        TARGETS m_${name} 
        DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endfunction()

add_subdirectory(denoisors)
