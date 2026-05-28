cmake_minimum_required(VERSION 4.0)
list(APPEND CMAKE_PREFIX_PATH "/opt/miniconda3")

project(BLAEQ_CUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

link_directories(BEFORE /opt/miniconda3/lib)
add_link_options("-Wl,--disable-new-dtags")
add_link_options("-L/opt/miniconda3/lib")

find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED)

find_package(cuvs REQUIRED)

# --- Ariel benchmark integration (M005) ---------------------------------
# The header-only benchmark timer lives at <repo-root>/benchmarks/cuda/.
# upstream/blaeq is two directories below the ariel-ASA repo root.
set(ARIEL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH
        "Path to the ariel-ASA repository root (holds benchmarks/).")

# Opt-in NVTX ranges inside ariel_bench::BenchScope (BUG-S5 conditional path).
# OFF by default so the standard build needs no nvToolsExt dependency.
option(ARIEL_USE_NVTX "Enable NVTX ranges in Ariel benchmark scopes" OFF)
# ------------------------------------------------------------------------

set(CUDA_SOURCES
        src/Data_Structures/Data_Structures.cu
        src/Data_Structures/File.cu
        src/Kernel/SpTSpMKernel.cu
        src/Kernel/SpMSpV.cu
        src/MergeSort/MergeSort.cu
        src/Query/GridCompact.cu
        src/Query/KNN.cu
        src/Query/Refactor.cu
        src/Query/RangePruning.cu
        src/Setup/Setup.cu
        src/Query/Query.cu
        src/Kmeans/CUDAKmeans.cu
        src/utils/NVTXProfiler.cu
        src/Test/QueryTest.cu
        src/main.cu
)

set(CPP_SOURCES
        src/Kmeans/Kmeans.cpp
)

add_executable(T-BLAEQ ${CUDA_SOURCES} ${CPP_SOURCES})

target_include_directories(T-BLAEQ PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ARIEL_REPO_ROOT}
)

target_link_libraries(T-BLAEQ PRIVATE
        CUDA::cudart_static
        CUDA::cusparse
        CUDA::cublas
        OpenMP::OpenMP_CXX
        cuvs::cuvs
)

target_compile_options(T-BLAEQ PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-Wall
        -Xcompiler=-fopenmp
        >
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Ariel benchmark target (M005) --------------------------------------
# Same sources as T-BLAEQ but compiled with the benchmark hook active. The
# QueryTimingAccumulator integration in Query.cu is dormant unless a caller
# invokes setBenchAccumulator(), so this target is behaviourally identical to
# T-BLAEQ until the benchmark driver wires up an accumulator. Defining the
# target separately lets us toggle NVTX and (later) a bench-only main without
# perturbing the production binary.
add_executable(T-BLAEQ-bench ${CUDA_SOURCES} ${CPP_SOURCES})

target_include_directories(T-BLAEQ-bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ARIEL_REPO_ROOT}
)

target_link_libraries(T-BLAEQ-bench PRIVATE
        CUDA::cudart_static
        CUDA::cusparse
        CUDA::cublas
        OpenMP::OpenMP_CXX
        cuvs::cuvs
)

target_compile_options(T-BLAEQ-bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-Wall
        -Xcompiler=-fopenmp
        >
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

target_compile_definitions(T-BLAEQ-bench PRIVATE ARIEL_BENCH_ENABLED=1)

# NVTX ranges (ariel_bench::BenchScope) are gated behind -DARIEL_USE_NVTX and
# require linking CUDA's nvToolsExt. Only pulled in when the option is ON so
# the default build keeps the existing dependency set.
if(ARIEL_USE_NVTX)
    target_compile_definitions(T-BLAEQ-bench PRIVATE ARIEL_USE_NVTX=1)
    target_link_libraries(T-BLAEQ-bench PRIVATE CUDA::nvToolsExt)
endif()
# ------------------------------------------------------------------------