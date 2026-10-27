# After Effects plug-in. Configure from the repo root with
#   cmake -S . -B build-ae -G Xcode -DAE_SDK_ROOT=/path/to/AfterEffectsSDK
# AE_SDK_ROOT is the folder that contains Examples/Headers.

set(AE_HEADERS ${AE_SDK_ROOT}/Examples/Headers)
if(NOT EXISTS ${AE_HEADERS}/AE_Effect.h)
  message(FATAL_ERROR "AE_SDK_ROOT=${AE_SDK_ROOT} does not contain Examples/Headers/AE_Effect.h")
endif()

if(NOT APPLE)
  message(WARNING "Glow Profond: only the macOS build is wired up for now; skipping the AE plug-in.")
  return()
endif()

add_library(GlowProfond MODULE GlowProfond.cpp)
target_link_libraries(GlowProfond PRIVATE glow_core)
target_include_directories(GlowProfond PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${AE_HEADERS}
  ${AE_HEADERS}/SP
  ${AE_SDK_ROOT}/Examples/Util
  ${AE_SDK_ROOT}/Examples/Resources)

# The PiPL resource tells After Effects what the bundle contains.
execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(PIPL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/GlowProfondPiPL.r)
set(PIPL_RSRC ${CMAKE_CURRENT_BINARY_DIR}/GlowProfond.rsrc)
add_custom_command(
  OUTPUT ${PIPL_RSRC}
  COMMAND xcrun Rez -o ${PIPL_RSRC} -useDF -script Roman
          -d __MACH__=1 -arch x86_64 -arch arm64
          -i ${CMAKE_CURRENT_SOURCE_DIR}
          -i ${AE_HEADERS}
          -i ${AE_HEADERS}/SP
          -i ${AE_SDK_ROOT}/Examples/Resources
          -isysroot ${MACOS_SDK_PATH}
          ${PIPL_SRC}
  DEPENDS ${PIPL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/GlowProfond_Version.h
  COMMENT "Compiling PiPL resource")
target_sources(GlowProfond PRIVATE ${PIPL_RSRC})
set_source_files_properties(${PIPL_RSRC} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

set_target_properties(GlowProfond PROPERTIES
  BUNDLE TRUE
  BUNDLE_EXTENSION plugin
  OSX_ARCHITECTURES "arm64;x86_64"
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in
  MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION}
  CXX_VISIBILITY_PRESET hidden)
# The core is linked statically into the bundle, so it must be universal too.
set_target_properties(glow_core PROPERTIES OSX_ARCHITECTURES "arm64;x86_64")
