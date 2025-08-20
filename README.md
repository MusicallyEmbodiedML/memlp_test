# memlp_test

Unit test wrapper for the MEMLP library. Currently supported platforms: RP2350, Ubuntu Linux.

## Building

This project supports building for both Raspberry Pi Pico (RP2350) and Linux using CMake with automatic platform detection and manual override options.

### Prerequisites

#### For Pico builds:
- Raspberry Pi Pico SDK
- CMake 3.13+
- ARM GCC toolchain

#### For Linux builds:
- CMake 3.13+
- GCC or Clang compiler
- Standard C++ libraries

### Build Instructions

#### Auto-Detection (Recommended for most users)

The build system automatically detects your target platform:

```bash
# Creates build directory and builds for detected platform
mkdir build
cd build
cmake ..
make -j$(nproc)
```

- If `PICO_SDK_PATH` is set in your environment → builds for Pico
- If no Pico SDK detected → builds for Linux

#### Manual Platform Selection

You can force a specific target regardless of your environment:

##### Linux Build (Force)
```bash
# Force Linux build even if Pico SDK is available
mkdir build-linux
cd build-linux
cmake -DFORCE_LINUX_BUILD=ON ..
make -j$(nproc)

# Run the tests
./memlp_test_linux
```

##### Pico Build (Force)
```bash
# Force Pico build even without SDK environment
mkdir build-pico
cd build-pico
cmake -DFORCE_PICO_BUILD=ON ..
make -j$(nproc)

# Flash to Pico (when connected in BOOTSEL mode)
cp memlp_test_pico.uf2 /media/$(whoami)/RPI-RP2/
```

#### Building Both Targets

To build both Linux and Pico versions simultaneously:

```bash
# Method 1: Separate commands
cmake -DFORCE_LINUX_BUILD=ON -B build-linux && cmake --build build-linux
cmake -DFORCE_PICO_BUILD=ON -B build-pico && cmake --build build-pico

# Method 2: Using separate directories
mkdir build-linux build-pico

cd build-linux
cmake -DFORCE_LINUX_BUILD=ON ..
make -j$(nproc)
cd ..

cd build-pico
cmake -DFORCE_PICO_BUILD=ON ..
make -j$(nproc)
cd ..
```

### Build Outputs

- **Linux**: `build-linux/memlp_test_linux` (executable)
- **Pico**: `build-pico/memlp_test_pico.uf2` (firmware file)

### Build Options

| CMake Option | Description | Default |
|--------------|-------------|---------|
| `FORCE_LINUX_BUILD` | Force build for Linux platform | OFF |
| `FORCE_PICO_BUILD` | Force build for Pico platform | OFF |

### Troubleshooting

#### Pico SDK Issues
```bash
# Ensure PICO_SDK_PATH is set
export PICO_SDK_PATH=/path/to/pico-sdk

# Or force Linux build if you want to skip Pico
cmake -DFORCE_LINUX_BUILD=ON ..
```

#### Build Conflicts
```bash
# Clean build directories if switching targets
rm -rf build-*
# Then rebuild with desired target
```

### Example Development Workflow

```bash
# 1. Test quickly on Linux during development
cmake -DFORCE_LINUX_BUILD=ON -B build-linux && cmake --build build-linux
./build-linux/memlp_test_linux

# 2. Build for Pico when ready to test on hardware
cmake -DFORCE_PICO_BUILD=ON -B build-pico && cmake --build build-pico

# 3. Flash to Pico
cp build-pico/memlp_test_pico.uf2 /media/$(whoami)/RPI-RP2/
```
