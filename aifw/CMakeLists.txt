
add_subdirectory(core)

add_library(aifw INTERFACE)
add_library(aifw::aifw ALIAS aifw)

target_link_libraries(aifw INTERFACE
  aifw::core
)
