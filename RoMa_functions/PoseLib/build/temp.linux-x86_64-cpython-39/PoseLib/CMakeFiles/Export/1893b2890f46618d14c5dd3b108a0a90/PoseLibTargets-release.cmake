#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PoseLib::PoseLib" for configuration "Release"
set_property(TARGET PoseLib::PoseLib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(PoseLib::PoseLib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libPoseLib.a"
  )

list(APPEND _cmake_import_check_targets PoseLib::PoseLib )
list(APPEND _cmake_import_check_files_for_PoseLib::PoseLib "${_IMPORT_PREFIX}/lib64/libPoseLib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
