add_library(omm-shared INTERFACE)
target_include_directories(omm-shared INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
set_target_properties("omm-shared" PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}")