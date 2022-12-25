add_library(omm-shared INTERFACE)
target_include_directories(omm-shared INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
