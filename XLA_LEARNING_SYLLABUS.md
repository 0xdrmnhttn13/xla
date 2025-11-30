# XLA Compiler Learning Syllabus

A structured learning path for understanding XLA, designed for compiler beginners.

---

## How to Use This Syllabus

This syllabus integrates two complementary learning resources:

1. **MLC Course (https://mlc.ai)** - Teaches general ML compiler principles using Apache TVM
2. **XLA Codebase** - Hands-on experience with Google's production ML compiler

**Recommended Approach:**
- **MLC for Theory:** Read MLC chapters to understand "why" and "what" of ML compilation concepts
- **XLA for Practice:** Apply concepts by exploring XLA's implementation and writing code
- **Compare & Contrast:** Understanding both TVM (from MLC) and XLA gives you broader perspective on ML compiler design

**Example Learning Flow:**
```
Week 6 (HLO Introduction):
1. Read MLC chapter on "Tensor Program Abstraction" (theory)
2. Study XLA's HLO implementation in xla/hlo/ir/ (practice)
3. Generate HLO from JAX and analyze it (hands-on)
4. Compare: How does HLO differ from TVM's tensor programs?
```

This dual approach accelerates learning by grounding abstract concepts in concrete implementations.

### MLC-to-XLA Concept Mapping

| MLC Course Topic | XLA Implementation | Where to Find in XLA |
|-----------------|-------------------|---------------------|
| Tensor Program Abstraction | HLO (High Level Operations) | `xla/hlo/ir/` |
| Computational Graphs | HloModule, HloComputation | `xla/hlo/ir/hlo_module.h` |
| Automatic Optimization | HLO Pass Pipeline | `xla/hlo/transforms/`, `xla/hlo/pass/` |
| GPU Acceleration | GPU Backend & Transforms | `xla/service/gpu/transforms/` |
| Memory Optimization | Buffer Assignment | `xla/service/buffer_assignment.h` |
| Code Generation | LLVM IR Emitter | `xla/service/llvm_ir/` |
| Hardware Integration | PJRT Plugin System | `xla/pjrt/plugin/` |

---

## Phase 1: Compiler Fundamentals (2-3 weeks)

### Week 1: Basic Compiler Theory

**Concepts to Learn:**
- What is a compiler? (Source code → Machine code transformation)
- Compilation stages: Lexing, Parsing, Semantic Analysis, Optimization, Code Generation
- Abstract Syntax Trees (AST)
- Intermediate Representations (IR)
- Control Flow Graphs (CFG)

**Resources:**
- **MLC Course - Introduction:** https://mlc.ai/chapter_introduction/index.html (HIGHLY RECOMMENDED - intro to ML compilation)
- Book: "Crafting Interpreters" by Robert Nystrom (free online, very beginner-friendly)
- Stanford CS143: Compilers course (lectures on YouTube)
- Book: "Engineering a Compiler" by Cooper & Torczon (Chapters 1-5)

**Hands-on Exercise:**
- Read MLC Chapter 1: Introduction to understand what ML compilation is about
- Write a simple calculator that parses and evaluates expressions like `2 + 3 * 4`
- Understand operator precedence and tree-based evaluation

### Week 2-3: LLVM Basics

**Why LLVM?** XLA uses LLVM for CPU/GPU code generation, so understanding LLVM IR is crucial.

**Concepts to Learn:**
- LLVM IR structure (modules, functions, basic blocks, instructions)
- SSA form (Static Single Assignment)
- LLVM passes and pass managers
- Basic LLVM optimizations

**Resources:**
- LLVM Tutorial: "Kaleidoscope" (official LLVM tutorial)
- LLVM Language Reference Manual
- "Getting Started with LLVM Core Libraries" by Bruno Cardoso Lopes

**Hands-on Exercise:**
```bash
# Install LLVM and experiment
# Write simple C code and view LLVM IR
clang -S -emit-llvm hello.c -o hello.ll
cat hello.ll  # Study the IR

# Run LLVM optimizations
opt -O2 hello.ll -S -o hello_opt.ll
diff hello.ll hello_opt.ll
```

---

## Phase 2: Machine Learning Fundamentals (2 weeks)

### Week 4: ML Basics for Compilers

You don't need to be an ML expert, but understanding what you're compiling helps!

**Concepts to Learn:**
- Tensors (multi-dimensional arrays)
- Basic operations: matrix multiplication, convolution, activation functions
- Neural network layers: Dense, Conv2D, BatchNorm
- Training vs. Inference
- Automatic differentiation basics

**Resources:**
- **MLC Course - Tensor Program Abstraction:** https://mlc.ai (Chapters on tensor operations)
- "Neural Networks and Deep Learning" by Michael Nielsen (free online)
- PyTorch tutorials (just the basics)
- 3Blue1Brown's neural network video series (YouTube)

**Hands-on Exercise:**
```python
# Write a simple neural network in NumPy
# Understand what operations are being performed
import numpy as np

# Simple 2-layer network
def forward(X, W1, W2):
    hidden = np.maximum(0, X @ W1)  # ReLU activation
    output = hidden @ W2
    return output
```

### Week 5: ML Frameworks Overview

**Concepts to Learn:**
- How PyTorch, TensorFlow, and JAX work at a high level
- Graph-based vs. eager execution
- Why ML models need compilation

**Resources:**
- **MLC Course - Computational Graphs:** https://mlc.ai (Learn about graph-based compilation)
- JAX documentation (JAX is tightly integrated with XLA)
- "Deep Learning with PyTorch" (free chapter 1)

**Hands-on Exercise:**
```python
# Install JAX and see XLA in action
import jax.numpy as jnp
from jax import jit

# This function will be compiled by XLA
@jit
def matrix_multiply(A, B):
    return jnp.dot(A, B)

# JAX uses XLA under the hood!
```

---

## Phase 3: XLA Foundations (3-4 weeks)

### Week 6: HLO Introduction

**XLA Directory:** `xla/hlo/ir/`

**Concepts to Learn:**
- What is HLO (High Level Operations)?
- HloModule, HloComputation, HloInstruction hierarchy
- HLO opcodes (operations)
- Reading HLO text format

**Files to Study:**
```
xla/hlo/ir/hlo_instruction.h      # Core instruction class
xla/hlo/ir/hlo_opcode.h           # All operation types
xla/hlo/ir/hlo_module.h           # Top-level compilation unit
xla/hlo/ir/hlo_computation.h      # Function-like unit
```

**Hands-on Exercise:**
```python
# Use JAX to generate HLO
import jax
import jax.numpy as jnp

def simple_function(x):
    return jnp.sin(x) * 2.0 + 1.0

# Get the HLO representation
x = jnp.array([1.0, 2.0, 3.0])
lowered = jax.jit(simple_function).lower(x)
print(lowered.as_text())  # Study this HLO output!
```

**MLC Course Connection:**
Read MLC chapters on **Tensor Program Abstraction** - understand how tensor operations are represented in compilers. HLO is XLA's specific implementation of this concept.

**Learning Exercise:**
- Read HLO examples in `xla/tests/` directory
- Understand how operations compose into computations
- Learn HLO text format syntax
- Compare HLO representation with tensor program concepts from MLC course

### Week 7: HLO Parser and Builder

**XLA Directory:** `xla/hlo/parser/`, `xla/hlo/builder/`

**Concepts to Learn:**
- Parsing HLO text format
- Building HLO graphs programmatically
- XlaBuilder API

**Files to Study:**
```
xla/hlo/parser/hlo_parser.h
xla/hlo/builder/xla_builder.h
xla/client/xla_builder.h
```

**Hands-on Exercise:**
```bash
# Build and run HLO parser tool
bazel build //xla/tools:hlo-opt

# Parse an HLO file
./bazel-bin/xla/tools/hlo-opt \
  --platform=cpu \
  some_hlo_file.txt
```

**Practical Task:**
- Find HLO test files in `xla/tests/`
- Try modifying HLO text and re-parsing
- Write simple HLO programs by hand

### Week 8-9: XLA Compilation Pipeline

**Concepts to Learn:**
- Complete flow: StableHLO → HLO → Optimized HLO → LLVM IR → Machine Code
- Compiler class hierarchy
- Backend selection

**Files to Study:**
```
xla/service/compiler.h              # Abstract compiler interface
xla/service/llvm_compiler.h         # LLVM-based compiler
xla/service/cpu/cpu_compiler.h      # CPU backend
xla/service/gpu/gpu_compiler.h      # GPU backend
```

**Documentation to Read:**
- `docs/architecture.md`
- `docs/developer_guide.md`

**Hands-on Exercise:**
```bash
# Set up XLA development environment
./configure.py --backend=CPU

# Build CPU compiler
bazel build //xla/service/cpu:cpu_compiler

# Enable HLO dumping to see compilation stages
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"

# Run a JAX program and examine dumps
python3 your_jax_script.py
ls /tmp/xla_dump/  # See HLO at each compilation stage!
```

---

## Phase 4: Optimization Passes (3-4 weeks)

### Week 10: Understanding Passes

**XLA Directory:** `xla/hlo/pass/`, `xla/hlo/transforms/`

**Concepts to Learn:**
- Pass infrastructure (HloPassInterface, HloModulePass)
- Pass pipelines
- Fixed-point iteration
- Hardware-independent optimizations

**Files to Study:**
```
xla/hlo/pass/hlo_pass_interface.h
xla/hlo/pass/hlo_pass_pipeline.h
xla/hlo/transforms/algebraic_simplifier.h
xla/hlo/transforms/constant_folding.h
```

**MLC Course Connection:**
Read MLC chapters on **Automatic Optimization** to understand compiler optimization principles that apply to both XLA and other ML compilers.

**Hands-on Exercise:**
1. Read a simple pass implementation: `xla/hlo/transforms/constant_folding.cc`
2. Understand how it traverses HLO and makes changes
3. Study test file: `xla/hlo/transforms/constant_folding_test.cc`

### Week 11: Common Optimization Passes

**Passes to Study (in order of complexity):**

**Easy:**
- `constant_folding` - Evaluate constant expressions at compile time
- `dead_code_elimination` - Remove unused computations

**Medium:**
- `algebraic_simplifier` - Simplify algebraic expressions (x * 0 → 0)
- `reshape_mover` - Move reshapes to reduce memory traffic
- `hlo_constant_folding` - More advanced constant folding

**Advanced:**
- `fusion` - Combine operations to reduce memory bandwidth
- `layout_assignment` - Decide memory layouts for tensors
- `buffer_assignment` - Assign physical memory to tensors

**Study Pattern for Each Pass:**
1. Read header file (*.h) for high-level description
2. Read implementation (*.cc) to understand the algorithm
3. Read test file (*_test.cc) to see concrete examples
4. Enable pass dumping and see it in action on real HLO

**Hands-on Exercise:**
```bash
# Dump HLO before and after a specific pass
export XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_pass_re=algebraic-simplifier"

# Run your JAX/TF program
python3 test.py

# Compare before/after
diff /tmp/xla/module_0000.before_algebraic-simplifier.txt \
     /tmp/xla/module_0000.after_algebraic-simplifier.txt
```

### Week 12-13: Backend-Specific Passes

**XLA Directory:** `xla/service/gpu/transforms/` (100+ passes!)

**MLC Course Connection:**
Study MLC chapters on **GPU Hardware Acceleration** to understand GPU programming models, memory hierarchies, and optimization strategies that XLA implements.

**GPU-Specific Concepts:**
- Kernel fusion strategies
- Memory coalescing
- Thread block sizing
- Triton code generation

**Key GPU Passes to Study:**
```
xla/service/gpu/transforms/fusion_merger.h
xla/service/gpu/transforms/gemm_rewriter.h
xla/service/gpu/transforms/gpu_conv_rewriter.h
```

**CPU-Specific Directory:** `xla/service/cpu/`

---

## Phase 5: Advanced Topics (4+ weeks)

### Week 14-15: Memory Management

**MLC Course Connection:**
Review MLC chapters on **Memory Optimization** - understand general principles of memory management in ML compilers before diving into XLA's specific implementation.

**Concepts to Learn:**
- Buffer assignment algorithm
- Heap simulation
- Memory space assignment
- Aliasing and liveness analysis

**Files to Study:**
```
xla/service/buffer_assignment.h
xla/service/heap_simulator.h
xla/service/memory_space_assignment/
```

**Documentation:** Read papers on XLA's memory optimization strategies

### Week 16: Code Generation

**MLC Course Connection:**
Study MLC chapters on **Code Generation** and **Hardware Integration** to see how different ML compilers approach the code generation problem.

**Concepts to Learn:**
- LLVM IR emission from HLO
- GPU kernel generation
- Triton integration

**Files to Study:**
```
xla/service/llvm_ir/llvm_util.h
xla/service/gpu/ir_emitter.h
xla/service/gpu/triton_emitter.h
```

**Hands-on Exercise:**
```bash
# Dump LLVM IR
export XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text=true"

# Look for *.ll files in dump directory
ls /tmp/xla/*.ll
```

### Week 17: PJRT and Runtime

**Concepts to Learn:**
- PJRT (Pretty Just-in-Time Runtime) architecture
- Plugin system
- Multi-device execution
- Distributed runtime

**Files to Study:**
```
xla/pjrt/c/README.md               # Start here!
xla/pjrt/pjrt_client.h
xla/pjrt/plugin/xla_cpu/           # Example plugin
```

### Week 18: Testing and Debugging

**Concepts to Learn:**
- Writing HLO tests
- Using HloTestBase
- Debugging failed optimizations
- Filecheck syntax for test expectations

**Files to Study:**
```
xla/tests/hlo_test_base.h
xla/tests/client_library_test_base.h
xla/tests/literal_test_util.h
```

**Hands-on Exercise:**
1. Write your first XLA test
2. Use test utilities to verify correctness
3. Debug a failing test

**Example Test Structure:**
```cpp
#include "xla/tests/hlo_test_base.h"

class MyOptimizationTest : public HloTestBase {};

TEST_F(MyOptimizationTest, SimplifiesAddZero) {
  const char* hlo = R"(
    HloModule test
    ENTRY main {
      x = f32[10] parameter(0)
      zero = f32[10] constant({0,0,0,0,0,0,0,0,0,0})
      ROOT result = f32[10] add(x, zero)
    }
  )";

  // Test that x + 0 simplifies to x
  RunAndFilecheckHloRewrite(hlo, AlgebraicSimplifier(), R"(
    CHECK: ROOT {{.*}} parameter(0)
  )");
}
```

---

## Phase 6: Contribution and Specialization (Ongoing)

### Week 19+: Choose Your Focus Area

**Option A: GPU Performance**
- Deep dive into `xla/service/gpu/`
- Study CUDA programming
- Learn about GPU memory hierarchies
- Contribute GPU optimizations

**Option B: New Hardware Backend**
- Study PJRT plugin system
- Implement backend for custom hardware
- Learn `xla/pjrt/plugin/example_plugin/`

**Option C: HLO Optimizations**
- Design new optimization passes
- Study compiler optimization literature
- Implement and benchmark improvements

**Option D: ML Framework Integration**
- JAX integration (already uses XLA)
- PyTorch XLA improvements
- TensorFlow XLA bridge

### First Contribution Ideas

**Beginner-Friendly:**
1. Add test coverage for existing passes
2. Improve documentation or fix typos
3. Add small HLO canonicalization rules

**Intermediate:**
1. Implement a simple optimization pass
2. Fix bugs in existing passes
3. Improve error messages

**Advanced:**
1. Design new fusion strategies
2. Improve memory planning algorithms
3. Add support for new operations

---

## Recommended Tools and Environment

### Essential Tools
```bash
# Install Bazelisk (Bazel version manager)
npm install -g @bazel/bazelisk

# Or use brew on macOS
brew install bazelisk

# Install compiler tools
sudo apt-get install clang llvm

# Python environment
pip install jax jaxlib numpy
```

### Helpful Debugging Flags
```bash
# Dump HLO at all stages
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"

# Dump only specific passes
export XLA_FLAGS="--xla_dump_hlo_pass_re=fusion"

# Disable optimizations for debugging
export XLA_FLAGS="--xla_disable_all_hlo_passes=true"

# Enable detailed logging
export XLA_FLAGS="--xla_log_hlo_text=true"
export TF_CPP_VMODULE="gpu_compiler=2,llvm_compiler=2"
```

### IDE Setup
- VS Code with C++ extensions
- clangd for code navigation
- Bazel plugin for build integration

---

## Learning Resources by Topic

### Compilers (General)
- **Book:** "Engineering a Compiler" (Cooper & Torczon)
- **Course:** Stanford CS143 (free online)
- **Online:** "Crafting Interpreters" (craftinginterpreters.com)

### LLVM
- **Tutorial:** LLVM Kaleidoscope Tutorial
- **Book:** "Getting Started with LLVM Core Libraries"
- **Docs:** llvm.org/docs

### ML Compilers
- **MLC Course (PRIMARY RESOURCE):** https://mlc.ai - Complete online course on ML compilation
  - **What it covers:** Tensor program abstraction, optimization, GPU acceleration, computational graphs, end-to-end compilation
  - **Why it's essential:** Provides foundational concepts that apply across all ML compilers including XLA
  - **How to use it:** Read chapters alongside XLA learning to understand the "why" behind XLA's design
  - **Key chapters:**
    - Introduction - ML compilation overview
    - Tensor Program Abstraction - How operations are represented (relates to HLO)
    - Automatic Optimization - Compiler passes and transformations
    - GPU Acceleration - Hardware-specific optimizations
    - Computational Graphs - High-level representations
    - Memory Optimization - Buffer management and scheduling
  - **Connection to XLA:** MLC covers general ML compiler principles; XLA is a specific implementation. Understanding both gives you deeper insight.
- **Survey Paper:** "The Deep Learning Compiler: A Comprehensive Survey"
- **Blog:** "Making Deep Learning Go Brrrr" (Horace He)
- **Docs:** TVM documentation (Apache TVM is the compiler used in MLC course examples)

### XLA Specific
- **XLA Docs:** `docs/` directory in this repo
- **Architecture:** `docs/architecture.md`
- **HLO Passes:** `docs/hlo_passes.md`
- **GPU Architecture:** `docs/gpu_architecture.md`
- **PJRT:** `xla/pjrt/c/README.md`

### Papers
- "XLA: Compiling Machine Learning for Peak Performance"
- "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems"
- "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"

---

## Weekly Time Commitment

**Minimum:** 10-15 hours/week
- 5-7 hours: Reading and studying
- 3-5 hours: Hands-on exercises
- 2-3 hours: Building and experimenting with XLA

**Ideal:** 20-25 hours/week
- Accelerates learning
- More time for deep dives
- Extra time for contribution attempts

---

## Milestones and Self-Assessment

### After Phase 1 (Compiler Fundamentals)
✓ I can explain what a compiler does in 3-5 sentences
✓ I understand what an IR is and why it's useful
✓ I can read simple LLVM IR
✓ I know what SSA form means

### After Phase 2 (ML Fundamentals)
✓ I understand what a tensor is
✓ I can explain matrix multiplication and convolution
✓ I know why ML frameworks use compilers
✓ I can write simple JAX code

### After Phase 3 (XLA Foundations)
✓ I can read HLO text format
✓ I understand HloModule/Computation/Instruction hierarchy
✓ I can generate HLO from JAX and examine it
✓ I know the main stages of XLA compilation

### After Phase 4 (Optimization Passes)
✓ I can explain how XLA passes work
✓ I've read and understood 5+ optimization passes
✓ I can trace HLO transformations through dump files
✓ I understand common optimizations (fusion, constant folding, etc.)

### After Phase 5 (Advanced Topics)
✓ I understand XLA's memory management
✓ I know how code generation works
✓ I can write XLA tests
✓ I'm ready to contribute

### After Phase 6 (Contribution)
✓ I've made my first contribution to XLA
✓ I've specialized in a focus area
✓ I can review others' code changes
✓ I'm a productive XLA contributor

---

## Study Tips

1. **Use MLC Course as a companion:** When learning XLA concepts, read the corresponding MLC chapters to understand the general principles. This dual approach (theory from MLC + practice with XLA) accelerates learning.

2. **Compare XLA with TVM:** MLC course uses Apache TVM for examples. Comparing TVM and XLA helps you understand different design choices in ML compilers.

3. **Don't rush:** Compilers are complex. It's okay to spend extra time on concepts.

4. **Learn by doing:** Reading code is important, but writing and running code cements understanding.

5. **Use HLO dumps extensively:** Seeing transformations in action is invaluable.

6. **Start small:** Don't try to understand the entire codebase at once.

7. **Ask questions:** Use the XLA discussion forums and GitHub issues.

8. **Read tests:** Test files (`*_test.cc`) often have the best examples.

9. **Track your progress:** Keep notes on what you've learned each week.

10. **Build often:** Regular builds help you catch issues early.

11. **Experiment:** Modify HLO and see what happens. Break things on purpose to learn.

12. **Connect with community:** Join OpenXLA discussions, read commit messages.

---

## Next Steps After This Syllabus

1. **Join the community:** openxla-discuss mailing list
2. **Follow development:** Watch the GitHub repository
3. **Read commit history:** Learn from recent changes
4. **Find a mentor:** Reach out to XLA maintainers
5. **Contribute regularly:** Consistent small contributions > sporadic large ones

---

**Remember:** Learning compilers is a marathon, not a sprint. Be patient with yourself, celebrate small wins, and enjoy the journey of understanding one of the most sophisticated ML compilers in existence!

**Good luck on your XLA learning journey! 🚀**
