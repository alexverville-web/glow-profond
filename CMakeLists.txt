cmake_minimum_required(VERSION 3.20)
project(GlowProfond VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Path to the Adobe After Effects SDK (the folder that contains Examples/).
# Leave empty to build only the core, CLI and tests.
set(AE_SDK_ROOT "" CACHE PATH "Adobe After Effects SDK root")

find_package(Threads REQUIRED)

add_library(glow_core STATIC core/src/glow.cpp)
target_include_directories(glow_core PUBLIC core/include)
target_link_libraries(glow_core PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(glow_core PRIVATE /W4)
else()
  target_compile_options(glow_core PRIVATE -Wall -Wextra)
endif()

add_executable(glowprofond cli/main.cpp)
target_include_directories(glowprofond PRIVATE third_party/stb)
target_link_libraries(glowprofond PRIVATE glow_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(core_test tests/core_test.cpp)
  target_link_libraries(core_test PRIVATE glow_core)
  add_test(NAME core COMMAND core_test)
  # Timing, not a test: ./glow_bench [width height radius repeats]
  add_executable(glow_bench tests/bench.cpp)
  target_link_libraries(glow_bench PRIVATE glow_core)

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME parity_with_reference
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/parity.py $<TARGET_FILE:glowprofond>)
  endif()
endif()

if(AE_SDK_ROOT)
  add_subdirectory(ae)
endif()
