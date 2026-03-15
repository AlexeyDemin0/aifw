# aifw

A learning project — a lightweight AI inference framework written in modern C++20.

## Goals

- **Flexible**: clean abstractions that cover most common AI workloads
- **Minimal dependencies**: the core has none; external libs are opt-in (e.g. BLAS, CUDA)
- **Cross-platform**: no OS or architecture assumptions in the core
- **Readable**: clean architecture, SOLID/DRY/KISS principles throughout

## Design

The framework is built around three layers:

- **Backend** — memory allocation and device abstraction (`IBackend`)
- **Kernel registry** — compute kernels per device (`IKernelRegistry`)
- **Ops API** — user-facing operations (`ops::add`, `ops::matmul`, ...)

Swapping CPU for CUDA means passing a different backend. Nothing else changes.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Options:

| Flag | Default | Description |
|---|---|---|
| `AIFW_BUILD_TESTS` | `ON` | Build test binaries |
| `AIFW_BUILD_BENCHMARKS` | `OFF` | Build benchmarks binaries |
| `AIFW_ENABLE_CUDA` | `OFF` | Enable CUDA backend |

## Requirements

- C++20 compiler (GCC 12+, Clang 14+, MSVC 19.34+)
- CMake 3.20+
- CUDA Toolkit (optional)

## Status

Early development. CPU backend and core tensor ops are functional.
CUDA backend is in progress.

## License

MIT
