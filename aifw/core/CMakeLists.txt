
# --- Core ---------------------------------------------------------------------

add_library(aifw_core)
add_library(aifw::core ALIAS aifw_core)

target_sources(aifw_core PRIVATE
  runtime/cpu/cpu_kernel_registry.cpp
)

target_include_directories(aifw_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

target_compile_definitions(aifw_core PUBLIC
  $<$<CONFIG:Debug>:AIFW_DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)

target_link_libraries(aifw_core PUBLIC aifw::warnings)

# --- CUDA backend -------------------------------------------------------------

if(AIFW_ENABLE_CUDA)
  target_sources(aifw_core PRIVATE
    ops/cuda_kernel_registry.cu
  )
  target_link_libraries(aifw_core PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(aifw_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()
