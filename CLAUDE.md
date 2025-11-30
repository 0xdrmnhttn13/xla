# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

XLA (Accelerated Linear Algebra) is an open-source machine learning compiler for GPUs, CPUs, and ML accelerators. It takes models from PyTorch, TensorFlow, and JAX, and optimizes them for high-performance execution across different hardware platforms.

## Build System

XLA uses **Bazel** as its build system. Install via [Bazelisk](https://github.com/bazelbuild/bazelisk) for automatic version management.

### Configuration

Before building, configure for your target backend:

```bash
# CPU backend
./configure.py --backend=CPU

# GPU backend (auto-detect compute capability - requires GPU)
./configure.py --backend=CUDA

# GPU backend (manual compute capability - for builds without GPU)
./configure.py --backend=CUDA --cuda_compute_capabilities="9.0"
```

This generates `xla_configure.bazelrc` with backend-specific settings.

### Building

```bash
# Build all XLA targets
bazel build --spawn_strategy=sandboxed --test_output=all //xla/...

# Build specific component
bazel build //xla/service/...
bazel build //xla/backends/gpu/...
```

### Common Bazel Configs

From `.bazelrc` and `tensorflow.bazelrc`:
- `--config=cuda` - CUDA/GPU support
- `--config=rocm` - AMD ROCm support
- `--config=cpu` - CPU-only build
- `--config=dbg` - Debug build with symbols
- `--config=avx_linux` - AVX instruction set

### Running Tests

```bash
# Single test
bazel test //xla/tests:axpy_simple_test

# Specific backend
bazel test //xla/tests:axpy_simple_test --test_arg=--xla_backend=cpu

# All tests in directory
bazel test //xla/tests/...

# GPU tests (requires GPU)
bazel test --config=cuda //xla/tests:some_gpu_test

# Filter tests by name
bazel test //xla/tests:some_test --test_filter=SpecificTestName
```

### Docker Development (Recommended)

```bash
# Start container (CPU/GPU support)
docker run -itd --rm \
  --name xla \
  -w /xla \
  -v $PWD:/xla \
  us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
  bash

# For GPU support, add: --gpus all

# Execute commands in container
docker exec xla ./configure.py --backend=CUDA
docker exec xla bazel build //xla/...
```

## Architecture

### Core Concepts

**HLO (High Level Operations)** is XLA's primary intermediate representation:
- **HloModule**: Top-level compilation unit (defined in `xla/hlo/ir/hlo_module.h`)
- **HloComputation**: Function-like unit with entry and nested computations (`xla/hlo/ir/hlo_computation.h`)
- **HloInstruction**: Individual operations in the computation DAG (`xla/hlo/ir/hlo_instruction.h`)
- **HloOpcode**: Operation types (Add, Dot, Convolution, etc.)

### Compilation Pipeline

1. **Frontend → StableHLO**: ML frameworks emit StableHLO (versioned portability layer)
2. **StableHLO → HLO**: Conversion to XLA's internal HLO representation
3. **Hardware-Independent Passes**: Optimizations in `xla/hlo/transforms/` (algebraic simplification, constant folding, etc.)
4. **Backend-Specific Passes**: Platform optimizations in `xla/service/gpu/transforms/`, `xla/service/cpu/`
5. **Code Generation**: LLVM IR generation (CPU/GPU) or Triton (GPU)
6. **Executable Creation**: Platform-specific executables with memory planning

### Directory Structure

**`xla/hlo/`** - HLO intermediate representation
- `ir/` - HLO instruction definitions, modules, computations
- `transforms/` - Hardware-independent optimization passes
- `analysis/` - Dataflow, alias analysis, etc.
- `pass/` - Pass framework infrastructure (`hlo_pass_interface.h`, `hlo_pass_pipeline.h`)
- `builder/` - HLO graph construction utilities
- `evaluator/` - HLO expression evaluation
- `parser/` - HLO text format parsing

**`xla/service/`** - Compiler infrastructure and shared components
- Core compiler interfaces (`compiler.h`, `llvm_compiler.h`)
- Contains 285+ BUILD files indicating extensive modularization
- Legacy backend code (being migrated to `xla/backends/`)
- Subdirectories: `llvm_ir/`, `memory_space_assignment/`, `spmd/`, `heap_simulator/`

**`xla/backends/`** - Hardware backend implementations (new structure)
- `cpu/` - CPU backend (runtime, codegen, autotuner, OneDNN integration)
- `gpu/` - GPU backend (CUDA/ROCm, Triton support, 100+ transforms in `xla/service/gpu/transforms/`)
- `interpreter/` - Reference interpreter backend
- `profiler/` - Profiling support
- `autotuner/` - Auto-tuning infrastructure

**`xla/client/`** - High-level client API
- `client.h` - Main client interface
- `local_client.h` - Local execution client
- `executable_build_options.h` - Compilation options

**`xla/pjrt/`** - PJRT (Pretty Just-in-Time Runtime) - Modern runtime API
- `c/` - C API for ABI stability (see `xla/pjrt/c/README.md`)
- `plugin/` - Plugin system for hardware backends (`xla_cpu/`, `xla_gpu/`, `xla_tpu/`, `example_plugin/`)
- `cpu/`, `gpu/`, `interpreter/` - Backend implementations
- `distributed/` - Multi-host distributed execution

**`xla/python/`** - Python bindings (primarily for JAX)
- `ifrt/` - IFRT (Interim Framework Runtime) API for framework portability
- `pjrt_ifrt/` - PJRT-based IFRT implementation
- `ifrt_proxy/` - Remote IFRT execution

**`xla/stream_executor/`** - Device abstraction layer (low-level device management)

**`xla/mlir/` and `xla/mlir_hlo/`** - MLIR integration
- XLA-specific MLIR framework and tools (bisect, interpreter, replay)
- MHLO/StableHLO dialect support

**`xla/ffi/`** - Foreign Function Interface (next-gen custom call mechanism)
- See `xla/ffi/README.md` for documentation
- Type-safe C API for custom operations

**`xla/codegen/`** - Unified code generation framework
- Kernel emitters and specifications
- LLVM and MLIR kernel sources

**`xla/tests/`** - Integration tests (150+ test files)
- Test base classes: `hlo_test_base.h`, `client_library_test_base.h`
- Backend-specific test suites (CPU, GPU variants)

**`xla/tsl/`** - TensorFlow Shared Libraries (vendored utilities)

### Directory Migration In Progress

The codebase is actively reorganizing:
- Moving from `xla/service/{cpu,gpu}` → `xla/backends/{cpu,gpu}`
- Separating HLO transforms into `xla/hlo/transforms/`

When working with backend-specific code, check both old (`xla/service/`) and new (`xla/backends/`) locations.

### Pass Framework

Passes transform HLO modules through the compilation pipeline:

- **Base classes**: `HloPassInterface` (abstract), `HloModulePass` (concrete base)
- **Pass execution**: `HloPassPipeline` (sequential), `HloPassFix` (fixed-point iteration)
- **Hardware-independent passes**: `xla/hlo/transforms/` (algebraic simplification, constant folding, etc.)
- **GPU-specific passes**: `xla/service/gpu/transforms/` (100+ transform files)
- **CPU-specific passes**: `xla/service/cpu/`

### Backend Architecture

- **Compiler interface**: `xla/service/compiler.h` provides abstract interface for platform-specific compilers
- **Registration**: Backends register via factory pattern
- **Types**: CPU (LLVM for multiple ISAs), GPU (NVIDIA/AMD/Intel), TPU (Google custom), Interpreter (reference)
- **Modern approach**: PJRT plugin system for dynamic backend loading (`xla/pjrt/plugin/`)

## Testing

### Test Organization

- **Unit tests**: Co-located with implementation files (`*_test.cc`)
- **Integration tests**: `xla/tests/` directory (150+ files)
- **Framework**: GoogleTest (gtest)
- **XLA-specific infrastructure**: `xla_test`, `xla_cc_test` wrappers in `xla/tests/build_defs.bzl`

### Test Base Classes

Located in `xla/tests/`:
- `HloTestBase` - HLO-level testing
- `ClientLibraryTestBase` - Client API testing
- `HloRunnerAgnosticTestBase` - Cross-platform HLO tests
- `LiteralTestUtil` - Result comparison utilities

### Backend Selection

Tests can target specific backends via tags:
- `cpu` - CPU backend
- `gpu`, `nvgpu_any`, `p100`, `v100`, `a100`, `h100`, `b200` - GPU variants
- `interpreter` - Reference interpreter
- Backend requirements: `requires-gpu-sm80`, etc.

## Code Standards

XLA follows [Google's C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and [Python Style Guide](https://google.github.io/styleguide/pyguide.html).

- **Compact changes**: Follow [Google's guide on small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)
- **Test coverage**: All changes require unit tests
- **Error handling**: Uses `absl::Status` and `absl::StatusOr<T>`
- **Naming conventions**:
  - HLO passes: `*Simplifier`, `*Rewriter`, `*Expander` suffixes
  - Tests: `*_test.cc` suffix
  - Build files: `BUILD` or `BUILD.bazel`

## Common Abbreviations

- **HLO**: High Level Operations
- **PJRT**: Pretty (much) Just-in-Time Runtime
- **IFRT**: Interim Framework Runtime
- **SPMD**: Single Program Multiple Data
- **TSL**: TensorFlow Shared Libraries
- **FFI**: Foreign Function Interface
- **MHLO**: Meta HLO (MLIR dialect)

## Important Documentation

- **Architecture**: `docs/architecture.md`
- **Developer Guide**: `docs/developer_guide.md`
- **HLO Passes**: `docs/hlo_passes.md`
- **GPU Architecture**: `docs/gpu_architecture.md`
- **PJRT**: `xla/pjrt/c/README.md`
- **FFI**: `xla/ffi/README.md`
- **IFRT**: `xla/python/ifrt/README.md`
- **Contributing**: `docs/contributing.md`
- **Tools**: `docs/tools.md`

## Review Process

- All changes require review via GitHub pull requests
- Code must follow style guides and all tests must pass
- Changes undergo internal testing on Google hardware/code
- Infrastructure improvements are expected as part of regular contributions
- Address all review comments in the same PR (follow-ups generally not accepted)
