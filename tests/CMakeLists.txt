
function(aifw_test_module target)
  target_link_libraries(${target} PRIVATE aifw::aifw)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_compile_definitions(${target} PRIVATE AIFW_DEBUG)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_executable(test_tensor
  tensor/main.cpp
  tensor/test_dtype.cpp
  tensor/test_shape.cpp
  tensor/test_stride.cpp
  tensor/test_tensor.cpp
  tensor/test_tensor_iterator.cpp
  tensor/test_tensor_views.cpp
)
aifw_test_module(test_tensor)

add_executable(test_ops
  ops/main.cpp
  ops/test_ops.cpp
)
aifw_test_module(test_ops)
