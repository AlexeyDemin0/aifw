
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(WARNING
    "Benchmark are built with ${CMAKE_BUILD_TYPE}."
    "Use -DCMAKE_BUILD_TYPE=Release for meaningful results."
  )
endif()

function(aifw_bench_module target)
  target_link_libraries(${target} PRIVATE aifw::aifw)
  target_compile_features(${target} PRIVATE cxx_std_20)
endfunction()

add_executable(bench_ops
  ops/main.cpp
  ops/bench_matmul.cpp
)
aifw_bench_module(bench_ops)
