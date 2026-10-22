# HelloWorld Sample

## Overview

This sample demonstrates the basic process of running and verifying a kernel function on the NPU side using the <<<>>> kernel launch operator. The kernel function prints output results via printf.

Debugging APIs such as `printf` are shared by the standard and C-style APIs.

The host calls `aclInit(nullptr)` to initialize the Runtime before setting the device, creating the stream, and launching the kernel. After execution, it destroys the stream, resets the device, and calls `aclFinalize()`. These Runtime APIs are independent of whether the kernel uses C-style or C++-style APIs.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── hello_world
│   ├── CMakeLists.txt      // Build project file
│   ├── hello_world.asc     // Ascend C sample implementation & sample call
│   └── README.md           // Sample documentation
```

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables

  Configure environment variables according to the [installation method](https://gitcode.com/cann/asc-devkit/blob/master/docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.
  ```bash
  source ${install_path}/cann/set_env.sh
  ```

  > **Note:** `${install_path}` is the CANN package installation directory. If no installation directory is specified, it defaults to `/usr/local/Ascend`.

- Run the Sample

  Execute the following commands in this sample directory.
  ```bash
  mkdir -p build && cd build;
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Build project (default npu mode)
  ./demo
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  For detailed information about CPU debugging, refer to [06_cpu_debug sample](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/01_utilities/06_cpu_debug).

  > **Note:** Clear the cmake cache before switching build modes. Execute `rm CMakeCache.txt` in the build directory, then run cmake again.

- Build Options

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Results

  The following execution results indicate successful execution.

  ```bash
  [Block (0/8)]: Hello World!!!
  [Block (1/8)]: Hello World!!!
  [Block (2/8)]: Hello World!!!
  [Block (3/8)]: Hello World!!!
  [Block (4/8)]: Hello World!!!
  [Block (5/8)]: Hello World!!!
  [Block (6/8)]: Hello World!!!
  [Block (7/8)]: Hello World!!!
  ```

  The output contains one message for each of the 8 blocks, in any order. The example shows the message body; the toolchain may add a core-identifying prefix.
