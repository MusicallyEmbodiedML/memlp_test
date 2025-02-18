# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_BUILD_TYPE "Debug")
set(PICO_DEOPTIMIZED_DEBUG 1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.0)
set(toolchainVersion 13_3_Rel1)
set(picotoolVersion 2.1.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(memlp_test C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1
file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
add_executable(memlp_test memlp_test.cpp ${SOURCES})

pico_set_program_name(memlp_test "memlp_test")
pico_set_program_version(memlp_test "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(memlp_test 0)
pico_enable_stdio_usb(memlp_test 1)

# Add the standard library to the build
target_link_libraries(memlp_test
        pico_stdlib)

# Add the standard include files to the build
target_include_directories(memlp_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/src
)

# Add any user requested libraries
target_link_libraries(memlp_test
        pico_cyw43_arch_none
        )

pico_add_extra_outputs(memlp_test)
