# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

# Build target selection options
option(FORCE_LINUX_BUILD "Force build for Linux even if Pico SDK detected" OFF)
option(FORCE_PICO_BUILD "Force build for Pico even without SDK detection" OFF)

# Check for conflicting options
if(FORCE_LINUX_BUILD AND FORCE_PICO_BUILD)
    message(FATAL_ERROR "Cannot force both Linux and Pico builds simultaneously")
endif()

# Determine build target
if(FORCE_LINUX_BUILD)
    set(BUILD_FOR_PICO OFF)
    message(STATUS "Forced Linux build")
elseif(FORCE_PICO_BUILD)
    set(BUILD_FOR_PICO ON)
    message(STATUS "Forced Pico build")
else()
    # Auto-detect based on environment
    if(DEFINED ENV{PICO_SDK_PATH} OR DEFINED PICO_SDK_PATH OR PICO_BOARD)
        set(BUILD_FOR_PICO ON)
        message(STATUS "Auto-detected Pico build (SDK found)")
    else()
        set(BUILD_FOR_PICO OFF)
        message(STATUS "Auto-detected Linux build (no Pico SDK)")
    endif()
endif()

# Platform-specific setup BEFORE project() call
if(BUILD_FOR_PICO)
    set(CMAKE_BUILD_TYPE "Debug")
    set(PICO_DEOPTIMIZED_DEBUG 1)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Pico SDK setup
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
    set(PICO_BOARD pico2_w CACHE STRING "Board type")

    # Pull in Raspberry Pi Pico SDK (must be before project)
    include(pico_sdk_import.cmake)
else()
    # Linux-specific setup
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type")
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Project declaration
if(BUILD_FOR_PICO)
    project(memlp_test_pico C CXX ASM)
    set(TARGET_NAME memlp_test_pico)
else()
    project(memlp_test_linux C CXX)
    set(TARGET_NAME memlp_test_linux)
endif()

# Platform-specific initialization
if(BUILD_FOR_PICO)
    # Initialise the Raspberry Pi Pico SDK
    pico_sdk_init()
else()
    # Linux-specific compiler flags
    add_compile_definitions(LINUX)
    add_compile_definitions(__force_inline=)  # GCC compatibility for inline hints
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wall -Wextra -O2)
    endif()
endif()

# Common source files
file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true
        ${CMAKE_CURRENT_LIST_DIR}/src/*.c
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# Create executable
add_executable(${TARGET_NAME} memlp_test.cpp ${SOURCES})

# Common include directories
target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/src
)

# Platform-specific configuration
if(BUILD_FOR_PICO)
    pico_set_program_name(${TARGET_NAME} "memlp_test")
    pico_set_program_version(${TARGET_NAME} "0.1")

    # Modify the below lines to enable/disable output over UART/USB
    pico_enable_stdio_uart(${TARGET_NAME} 0)
    pico_enable_stdio_usb(${TARGET_NAME} 1)

    # Add Pico-specific libraries
    target_link_libraries(${TARGET_NAME}
            pico_stdlib
            pico_cyw43_arch_none
    )

    pico_add_extra_outputs(${TARGET_NAME})
else()
    # Linux-specific libraries (if any needed)
    target_link_libraries(${TARGET_NAME}
            # Add Linux-specific libraries here if needed
            # For example: pthread, m (math library), etc.
    )
endif()

# Display build information
message(STATUS "=== Build Configuration ===")
message(STATUS "Target: ${TARGET_NAME}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(BUILD_FOR_PICO)
    message(STATUS "Pico Board: ${PICO_BOARD}")
else()
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()
message(STATUS "==========================")
