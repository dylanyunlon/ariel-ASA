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