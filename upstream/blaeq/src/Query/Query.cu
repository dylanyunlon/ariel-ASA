//
// Created by cuda01 on 2026/1/7.
//

#include "Query.cuh"
#include "src/Setup/Setup.cuh"
#include "KNN.cuh"
#include "RangePruning.cuh"
#include "GridCompact.cuh"
#include "Refactor.cuh"
#include "src/Kernel/SpMSpV.cuh"
#include "src/func.hpp"
#include "src/utils/NVTXProfiler.cuh"

namespace debug {
    void save_cluster_data(const double* centers,      // CPU端聚类中心 (n*d)
                       const double* h_radius,       // 半径 (n)
                       size_t n,                     // 聚类数量
                       size_t d,                     // 维度
                       const std::string& filepath)  // 文件路径
{
        // 1. 创建目录（如果不存在）
        std::filesystem::path file_path(filepath);
        std::filesystem::path dir_path = file_path.parent_path();

        if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
            try {
                std::filesystem::create_directories(dir_path);
            } catch (const std::filesystem::filesystem_error& e) {
                throw std::runtime_error("Failed to create directory: " +
                                         std::string(e.what()));
            }
        }
        // 2. 打开文件
        std::ofstream outfile(filepath);
        if (!outfile.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        // 设置输出精度
        outfile << std::fixed << std::setprecision(10);

        // 3. 写入头部：长度 维度数
        outfile << n << " " << d << "\n";

        // 拷贝数据
        double* h_centers;
        cudaPointerAttributes attr{};
        CUDA_CHECK(cudaPointerGetAttributes(&attr, centers));
        if (attr.type == cudaMemoryTypeDevice) {
            h_centers = new double[n * d];
            CUDA_CHECK(cudaMemcpy(h_centers, centers, n * d * sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            h_centers = const_cast<double*>(centers);
        }

        // 4. 写入n行聚类中心数据（每行d个数据点）
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                outfile << h_centers[i * d + j];
                if (j < d - 1) {
                    outfile << " ";
                }
            }
            outfile << "\n";
        }

        if (attr.type == cudaMemoryTypeDevice) {
            delete[] h_centers;
        }

        // 5. 写入最后一行：n个半径数据
        for (size_t i = 0; i < n; ++i) {
            outfile << h_radius[i];
            if (i < n - 1) {
                outfile << " ";
            }
        }
        outfile << "\n";

        outfile.close();

        // 验证写入成功
        if (!outfile.good()) {
            throw std::runtime_error("Error occurred while writing file: " + filepath);
        }
}
}

QueryResult::~QueryResult() {
    for (auto mesh : fineMesh) {
        deleteGrid(mesh);
    }
}

QueryHandler::QueryHandler(const std::string& datasetPath) {
    std::cout << "T-BLAEQ Building Index!" << std::endl;
    std::cout << std::filesystem::current_path().string() << std::endl;

    // Initialize parameters
    ratios = new size_t[3]{100, 50, 20};
    Is_AOS_Arch = true;

    // Load dataset
    dataset = loadFromFile(datasetPath);
    D = dataset.D;
    N = dataset.N;

    // Build index
    auto t1 = std::chrono::steady_clock::now();
    buildIndex();
    auto t2 = std::chrono::steady_clock::now();

    std::cout << "Finished" << std::endl;
    Chrono::printElapsed("Consumes Time", t1, t2);
}

void QueryHandler::buildIndex() {
    // 1.1 compute index height
    GPU_Index_Height = Compute_Layer_nums(N);
    assert(GPU_Index_Height >= 2);
    GPU_Index_Intervals = GPU_Index_Height - 1;
    std::cout << "GPU Index Height are " << GPU_Index_Height << std::endl;

    // 1.2 allocate arrays
    P_Tensors = new SparseTensorCscFormat*[GPU_Index_Height - 1];
    Meshs_Max_Radius = new double*[GPU_Index_Height - 1];

    maps = new size_t*[GPU_Index_Height];
    d_maps = new size_t*[GPU_Index_Height];
    for (size_t i = 0; i < GPU_Index_Height; ++i) {
        maps[i] = nullptr;
        d_maps[i] = nullptr;
    }

    Coreast_Mesh = nullptr;
    auto* KMeans_Ptr = new CUDAKmeans(dataset.data, N, D, Is_AOS_Arch);

    size_t datas_nums = N;
    size_t centroids_nums = 0;

    std::cout << "Begin to Generate Prolongation Tensors" << std::endl;
    std::cout << "0 represents the Coarsest Mesh, "
        << GPU_Index_Height - 1
        << " represents the Finest Mesh, and so on" << std::endl;
    std::cout << std::endl;

    constexpr size_t KMeans_Max_Itr_Param = 7;

    // 1.2 for-loop cal P_Tensors and Coreast_Mesh
    for (size_t i = 0; i < GPU_Index_Height - 1; i++) {
        // 1.2.1 calc KMeans params
        datas_nums = KMeans_Ptr->getdatas().size();
        meshs_nums.push_back(datas_nums);
        size_t ratio = ratios[i];
        centroids_nums = Compute_Centroid_nums(datas_nums, ratio);

        printf("Begin to Do Mesh_%ld -> Mesh_%ld Kmeans, (%ld -> %ld)\n", GPU_Index_Intervals - i,
               GPU_Index_Intervals - i - 1, datas_nums, centroids_nums);

        auto Kmeans_t1 = std::chrono::steady_clock::now();
        KMeans_Ptr->run(centroids_nums, KMeans_Max_Itr_Param);
        auto Kmeans_t2 = std::chrono::steady_clock::now();

        std::string Kmeans_Info = "Mesh_" + std::to_string(GPU_Index_Intervals - i) + " -> " + "Mesh_" +
            std::to_string(GPU_Index_Intervals - i - 1) + " KMeans Consumes Time";
        Chrono::printElapsed(Kmeans_Info, Kmeans_t1, Kmeans_t2);

        // 1.2.2 new csc
        SparseTensorCscFormat* SparseTensorCsc_Descr_t = nullptr;
        double* max_radius = nullptr;

        // 1.2.3 build one P_Tensor and calc max radius
        size_t curr_loop_coreast_mesh_idx = GPU_Index_Height - 2 - i;

        auto Generate_One_t1 = std::chrono::steady_clock::now();
        SparseTensorCsc_Descr_t = Genenate_One_P_Tensor(D, datas_nums, centroids_nums, KMeans_Ptr,
                                                        maps[curr_loop_coreast_mesh_idx + 1]);
        assert(maps[curr_loop_coreast_mesh_idx + 1] != nullptr);
        auto Generate_One_t2 = std::chrono::steady_clock::now();

        printf("P_Tensor idx = %ld\n", (GPU_Index_Height - 2 - i));
        std::string Generate_One_Info = "Mesh_" + std::to_string(GPU_Index_Intervals - i - 1) + " -> " + "Mesh_" +
            std::to_string(GPU_Index_Intervals - i) + " P_Tensor Constructs Time";
        Chrono::printElapsed(Generate_One_Info, Generate_One_t1, Generate_One_t2);

        auto Compute_Max_Radius_t1 = std::chrono::steady_clock::now();
        max_radius = Compute_Max_Radius(D, SparseTensorCsc_Descr_t->get_col_res(), maps[curr_loop_coreast_mesh_idx + 1],
                                        KMeans_Ptr);
        std::string Compute_Info = "Mesh_" + std::to_string(GPU_Index_Intervals - i - 1) + " Computes Time";
        auto Compute_Max_Radius_t2 = std::chrono::steady_clock::now();

        printf("Corease Mesh idx = %ld\n", (GPU_Index_Height - 2 - i));
        Chrono::printElapsed(Compute_Info, Compute_Max_Radius_t1, Compute_Max_Radius_t2);

        // 1.2.4 assign
        size_t P_Tensor_Write_Idx = curr_loop_coreast_mesh_idx;
        size_t Max_Radius_Write_Idx = curr_loop_coreast_mesh_idx;
        assert(SparseTensorCsc_Descr_t != nullptr);
        P_Tensors[P_Tensor_Write_Idx] = SparseTensorCsc_Descr_t;
        Meshs_Max_Radius[Max_Radius_Write_Idx] = max_radius;

        if (i + 2 == GPU_Index_Height) {
            auto& Centroids = KMeans_Ptr->getCentroids();
            meshs_nums.push_back(Centroids.size());
            Coreast_Mesh = new GridAsSparseMatrix(Centroids, 0, Centroids.size());
        }

        // 1.2.5 prepare for next-loop
        KMeans_Ptr->reset();
        std::cout << std::endl;
    }

    /* display mesh constructions*/
    std::cout << "From Finest Mesh To Coreast Mesh, val nums are : ";
    assert(meshs_nums.size() >= 2);
    {
        std::cout << meshs_nums[0];
        for (size_t i = 1; i < meshs_nums.size(); i++) {
            std::cout << " -> " << meshs_nums[i];
        }
        std::cout << std::endl;
    }

    std::cout << "We have finished establishing GPU Index in CPU !!!" << std::endl;

    // 1.3 load to dram
    loadMapsToDevice();

    // 1.4 clean work
    delete KMeans_Ptr;

    std::cout << "finish Tensors To File work" << std::endl;
}

void QueryHandler::loadMapsToDevice() {
    std::cout << "Load maps to DRAM !!!" << std::endl;
    auto load_t1 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < GPU_Index_Height; ++i) {
        assert(maps[0] == nullptr);
        assert(d_maps[0] == nullptr);
        if (maps[i] == nullptr) {
            assert(d_maps[i] == nullptr);
            continue;
        }

        // 1. device malloc
        size_t map_kv_size = meshs_nums[GPU_Index_Intervals - i];
        size_t* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, map_kv_size * sizeof(size_t));
        if (err != cudaSuccess) {
            std::cout << "fail to do cuda malloc" << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }

        // 2. Host -> Device copy
        err = cudaMemcpy(d_ptr, maps[i], map_kv_size * sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_ptr);
            std::cout << "fail to do cuda memcpy" << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
        d_maps[i] = d_ptr;
    }

    auto load_t2 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Load maps Consumes Time", load_t1, load_t2);
}

int QueryHandler::performQuery(const std::string& queryPath, QueryType qType,
                              long& totalTimeOut, size_t& queryCountOut,
                              int maxQueryCount, size_t K) {
    // 2. query stage
    std::cout << "Then we will do query" << std::endl;

    // 2.1 load query data
    Query queryData = qType == QueryType::POINT ? loadQueryPointFromFile(queryPath) : loadQueryRangeFromFile(queryPath);
    const auto queryCount = std::min(maxQueryCount, queryData.length);

    if (qType != queryData.type) {
        std::cerr << "Query type mismatch! Expected" << getQueryTypeString(qType) << " but got " <<
            getQueryTypeString(queryData.type) << std::endl;
        return 1;
    }

    int QueryType;
    switch (qType) {
    case QueryType::RANGE:
        QueryType = 0;
        break;
    case QueryType::POINT:
        QueryType = 1;
        break;
    default:
        std::cerr << "Unsupported Query Type!" << std::endl;
        return 1;
    }

    // 2.2 Q_count times queries
    std::cout << "begin to execute " << getQueryTypeString(qType) << " task " << std::endl;
    if (qType == QueryType::POINT) {
        std::cout << "KNN K is " << K << std::endl;
    }

    const auto nDim = queryData.dim;

    auto* lowBounds = new double[nDim];
    auto* upBounds = new double[nDim];
    auto* query_point = new double[nDim];

    long totalTime = 0;

    for (auto i = 0; i < queryCount; ++i) {
        std::cout << "begin to execute query " << i << std::endl;

        // prepare query data
        if (qType == QueryType::RANGE) {
            auto range = queryData.getQueryRange(i);
            std::copy(range.first.begin(), range.first.end(), lowBounds);
            std::copy(range.second.begin(), range.second.end(), upBounds);
        } else if (qType == QueryType::POINT) {
            auto point = queryData.getQueryPoint(i);
            std::copy(point.begin(), point.end(), query_point);
        } else {
            std::cerr << "Unsupported Query Type!" << std::endl;
            return 1;
        }

        auto t1 = std::chrono::steady_clock::now();
        NvtxProfiler profiler("Query", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Red);
        GridAsSparseMatrix* original_coreast_grid = Coreast_Mesh;
        GridAsSparseMatrix* sort_fine_grid = nullptr;
        GridAsSparseMatrix* res = nullptr;

        for (auto l = 0; l < GPU_Index_Height - 1; ++l) {
            auto innerProfilerName = "Level_" + std::to_string(l) + "_Query";
            NvtxProfiler innerProfiler(innerProfilerName.c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Green);
            bool* mask;
            size_t select_count;
            SparseTensorCscFormat* curr_P_Tensor = P_Tensors[l];

            if (QueryType == 0) {
                mask = rangePruning(lowBounds, upBounds, D, original_coreast_grid->get_vals_(), Meshs_Max_Radius[l],
                                    original_coreast_grid->get_nnz_nums(), original_coreast_grid->get_ids_(),
                                    original_coreast_grid->get_num_rows(), select_count);
            }
            else if (QueryType == 1) {
                mask = knnPruning(K, original_coreast_grid->get_nnz_nums(), D, original_coreast_grid->get_num_rows(),
                                  query_point, original_coreast_grid->get_vals_(), Meshs_Max_Radius[l],
                                  curr_P_Tensor->get_nnz_per_col(), original_coreast_grid->get_ids_(), select_count);
            }
            else {
                throw std::invalid_argument("Invalid QueryType");
            }

            auto pruned_original_coreast_grid = compactGrid(*original_coreast_grid, mask, select_count);
            sort_fine_grid = SpTSpMMultiplication_v2(curr_P_Tensor, pruned_original_coreast_grid);
            refactor(*sort_fine_grid, d_maps[l + 1]);

            // xak note :: clean
            if (l > 0) {
                deleteGrid(original_coreast_grid);
            }
            deleteGrid(pruned_original_coreast_grid);

            // xak note :: prepare for next_loop
            original_coreast_grid = sort_fine_grid;
        }

        profiler.release();
        auto t2 = std::chrono::steady_clock::now();

        auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        totalTime += time;
        std::cout << "Query " << i << " takes " << (static_cast<double>(time) / 1000.0) << " ms." <<std::endl;
    }

    // cleanup query buffers
    delete[] lowBounds;
    delete[] upBounds;
    delete[] query_point;

    // 2.4 finish query stage
    std::cout << "finish query task !!!" << std::endl;
    std::cout << "Average query time: " << (static_cast<double>(totalTime) / static_cast<double>(queryCount) / 1000.0) << " ms." << std::endl;
    std::cout << "Total query time: " << (static_cast<double>(totalTime) / 1000.0) << " ms." << std::endl;

    totalTimeOut = totalTime;
    queryCountOut = queryCount;
    return 0;
}

QueryResult QueryHandler::performQueryWithPreLoadPvals(const std::string& queryPath, QueryType qType, bool saveFineMesh, int maxQueryCount, size_t K) {
    QueryResult result;
    result.type = qType;

    if (!isTensorInDevice) {
        loadTensorValsToDevice();
    }
        // 2. query stage
    std::cout << "Then we will do query" << std::endl;

    // 2.1 load query data
    Query queryData = qType == QueryType::POINT ? loadQueryPointFromFile(queryPath) : loadQueryRangeFromFile(queryPath);
    const auto queryCount = std::min(maxQueryCount, queryData.length);

    if (qType != queryData.type) {
        std::cerr << "Query type mismatch! Expected" << getQueryTypeString(qType) << " but got " <<
            getQueryTypeString(queryData.type) << std::endl;
        result.errorCode = 1;
        return result;
    }

    int QueryType;
    switch (qType) {
    case QueryType::RANGE:
        QueryType = 0;
        break;
    case QueryType::POINT:
        QueryType = 1;
        break;
    default:
        std::cerr << "Unsupported Query Type!" << std::endl;
        result.errorCode = 1;
        return result;
    }

    // 2.2 Q_count times queries
    std::cout << "begin to execute " << getQueryTypeString(qType) << " task " << std::endl;
    if (qType == QueryType::POINT) {
        std::cout << "KNN K is " << K << std::endl;
    }

    const auto nDim = queryData.dim;

    auto* lowBounds = new double[nDim];
    auto* upBounds = new double[nDim];
    auto* query_point = new double[nDim];

    long totalTime = 0;

    if (saveFineMesh) {
        result.fineMesh.reserve(queryCount);
    }
    result.queryRangeVolume.reserve(queryCount);
    result.fineMeshSize.reserve(queryCount);

    for (auto i = 0; i < queryCount; ++i) {
        std::cout << "begin to execute query " << i << std::endl;

        // prepare query data
        if (qType == QueryType::RANGE) {
            auto range = queryData.getQueryRange(i);
            std::copy(range.first.begin(), range.first.end(), lowBounds);
            std::copy(range.second.begin(), range.second.end(), upBounds);
        } else if (qType == QueryType::POINT) {
            auto point = queryData.getQueryPoint(i);
            std::copy(point.begin(), point.end(), query_point);
        } else {
            std::cerr << "Unsupported Query Type!" << std::endl;
            result.errorCode = 1;
            return result;
        }

        auto profilerName = getQueryProfilerName(qType, N, D);
        auto t1 = std::chrono::steady_clock::now();
        NvtxProfiler profiler(profilerName.c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Red);
        GridAsSparseMatrix* original_coreast_grid = Coreast_Mesh;
        GridAsSparseMatrix* sort_fine_grid = nullptr;
        GridAsSparseMatrix* res = nullptr;

        for (auto l = 0; l < GPU_Index_Height - 1; ++l) {
            auto innerProfilerName = "QueryLevel" + std::to_string(l);
            NvtxProfiler innerProfiler(innerProfilerName.c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Green);
            bool* mask;
            size_t select_count;
            SparseTensorCscFormat* curr_P_Tensor = P_Tensors[l];
            auto d_P_Tensor_val = d_P_Tensor_vals[l];

            // debug::save_cluster_data(original_coreast_grid->get_vals_(), Meshs_Max_Radius[l], original_coreast_grid->get_nnz_nums(), original_coreast_grid->get_dimensions(), "debug/level_" + std::to_string(l) + "_data.txt");

            if (QueryType == 0) {
                mask = rangePruning(lowBounds, upBounds, D, original_coreast_grid->get_vals_(), Meshs_Max_Radius[l],
                                    original_coreast_grid->get_nnz_nums(), original_coreast_grid->get_ids_(),
                                    original_coreast_grid->get_num_rows(), select_count);
            }
            else if (QueryType == 1) {
                mask = knnPruning(K, original_coreast_grid->get_nnz_nums(), D, original_coreast_grid->get_num_rows(),
                                  query_point, original_coreast_grid->get_vals_(), Meshs_Max_Radius[l],
                                  curr_P_Tensor->get_nnz_per_col(), original_coreast_grid->get_ids_(), select_count);
            }
            else {
                throw std::invalid_argument("Invalid QueryType");
            }

            auto pruned_original_coreast_grid = compactGrid(*original_coreast_grid, mask, select_count);
            /*
            // 对于范围查询，检查倒数第二层的剪枝正确性
            if (l == GPU_Index_Height -2 && QueryType == 0) {
                const auto checkResult = checkRangeQuery(pruned_original_coreast_grid->get_vals_(), lowBounds, upBounds, Meshs_Max_Radius[l], pruned_original_coreast_grid->get_ids_(), pruned_original_coreast_grid->get_num_rows(), pruned_original_coreast_grid->get_nnz_nums(), D);
                if (!checkResult) {
                    std::cerr << "Range Pruning check failed" << std::endl;
                }
            }
            */
            sort_fine_grid = SpTSpMMultiplication_v3(curr_P_Tensor, pruned_original_coreast_grid, d_P_Tensor_val);
            refactor(*sort_fine_grid, d_maps[l + 1]);

            // xak note :: clean
            if (l > 0) {
                deleteGrid(original_coreast_grid);
            }
            deleteGrid(pruned_original_coreast_grid);

            // xak note :: prepare for next_loop
            original_coreast_grid = sort_fine_grid;

            std::cout << "Level: " << l << " Count： " << sort_fine_grid->get_nnz_nums() << std::endl;
        }

        profiler.release();
        auto t2 = std::chrono::steady_clock::now();

        auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        totalTime += time;
        std::cout << "Query " << i << " takes " << (static_cast<double>(time) / 1000.0) << " ms." <<std::endl;
        result.fineMeshSize.push_back(sort_fine_grid->get_nnz_nums());
        // 计算查询范围体积
        if (qType == QueryType::RANGE) {
            result.queryRangeVolume.push_back(calcRangeQueryVolume(lowBounds, upBounds, nDim));
        } else if (qType == QueryType::POINT) {
            result.queryRangeVolume.push_back(calcKNNQueryVolume(query_point, sort_fine_grid));
        }
        if (saveFineMesh) {
            result.fineMesh.push_back(sort_fine_grid);
        } else {
            deleteGrid(sort_fine_grid);
        }
    }

    // cleanup query buffers
    delete[] lowBounds;
    delete[] upBounds;
    delete[] query_point;

    // 2.4 finish query stage
    std::cout << "finish query task !!!" << std::endl;
    std::cout << "Average query time: " << (static_cast<double>(totalTime) / static_cast<double>(queryCount) / 1000.0) << " ms." << std::endl;
    std::cout << "Total query time: " << (static_cast<double>(totalTime) / 1000.0) << " ms." << std::endl;

    result.totalTime = totalTime;
    result.queryCount = queryCount;
    return result;
}

void QueryHandler::cleanup() {
    // 2.3 clean work
    for (size_t i = 0; i < GPU_Index_Height - 1; i++) {
        SparseTensorCscFormat* curr_P_Tensor = P_Tensors[i];
        delete curr_P_Tensor;
        delete[] Meshs_Max_Radius[i];
    }
    for (size_t i = 0; i < GPU_Index_Height; i++) {
        delete[] maps[i];
        if (d_maps[i] != nullptr) {
            cudaFree(d_maps[i]);
        }
    }
    if (isTensorInDevice) {
        for (size_t i = 0; i < GPU_Index_Height - 1; i++) {
            CUDA_CHECK(cudaFree(d_P_Tensor_vals[i]));
        }
        delete[] d_P_Tensor_vals;
    }
    delete Coreast_Mesh;
    delete[] ratios;
    delete[] P_Tensors;
    delete[] Meshs_Max_Radius;
    delete[] maps;
    delete[] d_maps;
}

QueryHandler::QueryHandler(const std::string& indexPath, bool loadFromIndex) {
    if (!loadFromIndex) {
        throw std::invalid_argument("Use QueryHandler(datasetPath) for building new index");
    }

    std::cout << "Loading index from: " << indexPath << std::endl;
    auto t1 = std::chrono::steady_clock::now();

    loadIndexFromFile(indexPath);

    auto t2 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index Loading Time", t1, t2);

    // 加载 maps 到设备
    loadMapsToDevice();
}

// 保存索引到文件
void QueryHandler::saveIndex(const std::string& indexPath) {
    std::cout << "Saving index to: " << indexPath << std::endl;
    auto t1 = std::chrono::steady_clock::now();

    // 创建目录
    std::filesystem::create_directories(indexPath);

    // 1. 保存元数据
    saveMetadata(indexPath + "/metadata.bin");

    // 2. 保存 P_Tensors
    for (size_t i = 0; i < GPU_Index_Intervals; ++i) {
        std::string filename = indexPath + "/P_Tensor_" + std::to_string(i) + ".bin";
        saveSparseTensorCsc(filename, P_Tensors[i]);
    }

    // 3. 保存 Meshs_Max_Radius
    for (size_t i = 0; i < GPU_Index_Intervals; ++i) {
        std::string filename = indexPath + "/MaxRadius_" + std::to_string(i) + ".bin";
        size_t count = meshs_nums[GPU_Index_Intervals - i];
        saveMaxRadius(filename, Meshs_Max_Radius[i], count);
    }

    // 4. 保存 maps
    for (size_t i = 1; i < GPU_Index_Height; ++i) {
        if (maps[i] != nullptr) {
            std::string filename = indexPath + "/Map_" + std::to_string(i) + ".bin";
            size_t size = meshs_nums[GPU_Index_Intervals - i];
            saveMap(filename, maps[i], size);
        }
    }

    // 5. 保存 Coreast_Mesh
    saveCoreastMesh(indexPath + "/Coreast_Mesh.bin", Coreast_Mesh);

    auto t2 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index Saving Time", t1, t2);
    std::cout << "Index saved successfully!" << std::endl;
}

// 从文件加载索引
void QueryHandler::loadIndexFromFile(const std::string& indexPath) {
    // 1. 加载元数据
    loadMetadata(indexPath + "/metadata.bin");

    // 2. 分配数组
    P_Tensors = new SparseTensorCscFormat*[GPU_Index_Intervals];
    Meshs_Max_Radius = new double*[GPU_Index_Intervals];
    maps = new size_t*[GPU_Index_Height];
    d_maps = new size_t*[GPU_Index_Height];

    for (size_t i = 0; i < GPU_Index_Height; ++i) {
        maps[i] = nullptr;
        d_maps[i] = nullptr;
    }

    // 3. 加载 P_Tensors
    for (size_t i = 0; i < GPU_Index_Intervals; ++i) {
        std::string filename = indexPath + "/P_Tensor_" + std::to_string(i) + ".bin";
        P_Tensors[i] = loadSparseTensorCsc(filename);
    }

    // 4. 加载 Meshs_Max_Radius
    for (size_t i = 0; i < GPU_Index_Intervals; ++i) {
        std::string filename = indexPath + "/MaxRadius_" + std::to_string(i) + ".bin";
        size_t count;
        Meshs_Max_Radius[i] = loadMaxRadius(filename, count);
        assert(count == meshs_nums[GPU_Index_Intervals - i]);
    }

    // 5. 加载 maps
    for (size_t i = 1; i < GPU_Index_Height; ++i) {
        std::string filename = indexPath + "/Map_" + std::to_string(i) + ".bin";
        if (std::filesystem::exists(filename)) {
            size_t size;
            maps[i] = loadMap(filename, size);
            assert(size == meshs_nums[GPU_Index_Intervals - i]);
        }
    }

    // 6. 加载 Coreast_Mesh
    Coreast_Mesh = loadCoreastMesh(indexPath + "/Coreast_Mesh.bin");

    std::cout << "Index loaded successfully!" << std::endl;
}

// 保存元数据
void QueryHandler::saveMetadata(const std::string& filepath) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    ofs.write(reinterpret_cast<const char*>(&D), sizeof(D));
    ofs.write(reinterpret_cast<const char*>(&N), sizeof(N));
    ofs.write(reinterpret_cast<const char*>(&GPU_Index_Height), sizeof(GPU_Index_Height));
    ofs.write(reinterpret_cast<const char*>(&GPU_Index_Intervals), sizeof(GPU_Index_Intervals));
    ofs.write(reinterpret_cast<const char*>(&Is_AOS_Arch), sizeof(Is_AOS_Arch));

    // 保存 ratios
    for (size_t i = 0; i < GPU_Index_Intervals; ++i) {
        ofs.write(reinterpret_cast<const char*>(&ratios[i]), sizeof(size_t));
    }

    // 保存 meshs_nums
    size_t meshs_size = meshs_nums.size();
    ofs.write(reinterpret_cast<const char*>(&meshs_size), sizeof(meshs_size));
    ofs.write(reinterpret_cast<const char*>(meshs_nums.data()), meshs_size * sizeof(size_t));

    ofs.close();
}

// 加载元数据
void QueryHandler::loadMetadata(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    ifs.read(reinterpret_cast<char*>(&D), sizeof(D));
    ifs.read(reinterpret_cast<char*>(&N), sizeof(N));
    ifs.read(reinterpret_cast<char*>(&GPU_Index_Height), sizeof(GPU_Index_Height));
    ifs.read(reinterpret_cast<char*>(&GPU_Index_Intervals), sizeof(GPU_Index_Intervals));
    ifs.read(reinterpret_cast<char*>(&Is_AOS_Arch), sizeof(Is_AOS_Arch));

    std::cout << "Metadata Loaded: D=" << D << ", N=" << N
              << ", GPU_Index_Height=" << GPU_Index_Height
              << ", GPU_Index_Intervals=" << GPU_Index_Intervals
              << ", Is_AOS_Arch=" << Is_AOS_Arch << std::endl;

    // 加载 ratios
    ratios = new size_t[GPU_Index_Intervals];
    for (size_t i = 0; i < GPU_Index_Intervals; ++i) {
        ifs.read(reinterpret_cast<char*>(&ratios[i]), sizeof(size_t));
    }

    // 加载 meshs_nums
    size_t meshs_size;
    ifs.read(reinterpret_cast<char*>(&meshs_size), sizeof(meshs_size));
    meshs_nums.resize(meshs_size);
    ifs.read(reinterpret_cast<char*>(meshs_nums.data()), meshs_size * sizeof(size_t));

    ifs.close();
}

// 保存 SparseTensorCscFormat
void QueryHandler::saveSparseTensorCsc(const std::string& filepath, SparseTensorCscFormat* tensor) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    size_t D = tensor->get_dim();
    size_t row_nums = tensor->get_row_nums();
    size_t col_nums = tensor->get_col_nums();
    size_t nnz_nums = tensor->get_nnz_nums();
    bool is_aos = tensor->get_memory_arch();

    ofs.write(reinterpret_cast<const char*>(&D), sizeof(D));
    ofs.write(reinterpret_cast<const char*>(&row_nums), sizeof(row_nums));
    ofs.write(reinterpret_cast<const char*>(&col_nums), sizeof(col_nums));
    ofs.write(reinterpret_cast<const char*>(&nnz_nums), sizeof(nnz_nums));
    ofs.write(reinterpret_cast<const char*>(&is_aos), sizeof(is_aos));

    ofs.write(reinterpret_cast<const char*>(tensor->get_row_ids()), nnz_nums * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(tensor->get_nnz_per_col()), col_nums * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(tensor->get_col_res()), (col_nums + 1) * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(tensor->get_vals()), nnz_nums * D * sizeof(double));

    ofs.close();
}

// 加载 SparseTensorCscFormat
SparseTensorCscFormat* QueryHandler::loadSparseTensorCsc(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    size_t D, row_nums, col_nums, nnz_nums;
    bool is_aos;

    ifs.read(reinterpret_cast<char*>(&D), sizeof(D));
    ifs.read(reinterpret_cast<char*>(&row_nums), sizeof(row_nums));
    ifs.read(reinterpret_cast<char*>(&col_nums), sizeof(col_nums));
    ifs.read(reinterpret_cast<char*>(&nnz_nums), sizeof(nnz_nums));
    ifs.read(reinterpret_cast<char*>(&is_aos), sizeof(is_aos));

    // 创建临时向量用于构造
    std::vector<size_t> nnz_per_col_vec(col_nums);
    ifs.seekg(nnz_nums * sizeof(size_t), std::ios::cur); // 跳过 row_ids
    ifs.read(reinterpret_cast<char*>(nnz_per_col_vec.data()), col_nums * sizeof(size_t));
    ifs.seekg(-(nnz_nums * sizeof(size_t) + col_nums * sizeof(size_t)), std::ios::cur); // 回退

    auto* tensor = new SparseTensorCscFormat(D, row_nums, col_nums, nnz_per_col_vec);

    ifs.read(reinterpret_cast<char*>(const_cast<size_t*>(tensor->get_row_ids())), nnz_nums * sizeof(size_t));
    ifs.seekg(col_nums * sizeof(size_t), std::ios::cur); // 跳过已读的 nnz_per_col
    ifs.read(reinterpret_cast<char*>(const_cast<size_t*>(tensor->get_col_res())), (col_nums + 1) * sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(const_cast<double*>(tensor->get_vals())), nnz_nums * D * sizeof(double));

    ifs.close();
    return tensor;
}

// 保存 MaxRadius 数组
void QueryHandler::saveMaxRadius(const std::string& filepath, double* radius, size_t count) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(radius), count * sizeof(double));

    ofs.close();
}

// 加载 MaxRadius 数组
double* QueryHandler::loadMaxRadius(const std::string& filepath, size_t& count) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    auto* radius = new double[count];
    ifs.read(reinterpret_cast<char*>(radius), count * sizeof(double));

    ifs.close();
    return radius;
}

// 保存 Map 数组
void QueryHandler::saveMap(const std::string& filepath, size_t* map, size_t size) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char*>(map), size * sizeof(size_t));

    ofs.close();
}

// 加载 Map 数组
size_t* QueryHandler::loadMap(const std::string& filepath, size_t& size) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
    auto* map = new size_t[size];
    ifs.read(reinterpret_cast<char*>(map), size * sizeof(size_t));

    ifs.close();
    return map;
}

// 保存 Coreast_Mesh
void QueryHandler::saveCoreastMesh(const std::string& filepath, GridAsSparseMatrix* mesh) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }

    size_t M_row = mesh->get_num_rows();
    size_t M_dim = mesh->get_dimensions();
    size_t M_nnz_nums = mesh->get_nnz_nums();
    bool is_aos = mesh->get_memory_arch();

    ofs.write(reinterpret_cast<const char*>(&M_row), sizeof(M_row));
    ofs.write(reinterpret_cast<const char*>(&M_dim), sizeof(M_dim));
    ofs.write(reinterpret_cast<const char*>(&M_nnz_nums), sizeof(M_nnz_nums));
    ofs.write(reinterpret_cast<const char*>(&is_aos), sizeof(is_aos));

    ofs.write(reinterpret_cast<const char*>(mesh->get_ids_()), M_nnz_nums * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(mesh->get_vals_()), M_nnz_nums * M_dim * sizeof(double));

    ofs.close();
}

// 加载 Coreast_Mesh
GridAsSparseMatrix* QueryHandler::loadCoreastMesh(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }

    size_t M_row, M_dim, M_nnz_nums;
    bool is_aos;

    ifs.read(reinterpret_cast<char*>(&M_row), sizeof(M_row));
    ifs.read(reinterpret_cast<char*>(&M_dim), sizeof(M_dim));
    ifs.read(reinterpret_cast<char*>(&M_nnz_nums), sizeof(M_nnz_nums));
    ifs.read(reinterpret_cast<char*>(&is_aos), sizeof(is_aos));

    auto* ids = new size_t[M_nnz_nums];
    auto* vals = new double[M_nnz_nums * M_dim];

    ifs.read(reinterpret_cast<char*>(ids), M_nnz_nums * sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(vals), M_nnz_nums * M_dim * sizeof(double));

    auto* mesh = new GridAsSparseMatrix(M_row, M_dim, M_nnz_nums, ids, vals);
    mesh->set_memory_arch(is_aos);

    ifs.close();
    return mesh;
}

QueryHandler::~QueryHandler() {
    cleanup();
}

void QueryHandler::loadTensorValsToDevice() {
    if (!isTensorInDevice) {
        d_P_Tensor_vals = new double*[GPU_Index_Intervals];
        for (auto i = 0; i < GPU_Index_Intervals; ++i) {
            auto curr_P_Tensor = P_Tensors[i];
            auto nzz_num = curr_P_Tensor->get_nnz_nums();
            auto dim = curr_P_Tensor->get_dim();
            double* d_vals = nullptr;
            CUDA_CHECK(cudaMalloc(&d_vals, nzz_num * dim * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_vals, curr_P_Tensor->get_vals(), nzz_num * dim * sizeof(double), cudaMemcpyHostToDevice));
            d_P_Tensor_vals[i] = d_vals;
        }
    }
}

size_t QueryHandler::getSize() const {
    return N;
}

size_t QueryHandler::getDim() const {
    return D;
}

double calcRangeQueryVolume(const double* lowBounds, const double* upBounds, const size_t dim) {
    double logVolume = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const auto side = upBounds[i] - lowBounds[i];
        if (side <= 0) return -INFINITY;
        logVolume += std::log(side);
    }
    return logVolume;
}

double calcDistance(const double* pointA, const double* pointB, const size_t dim) {
    double dist = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double diff = pointA[i] - pointB[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

double calcSphereVolume(const double radius, const size_t dim) {
    const auto fdim = static_cast<double>(dim);
    const auto l_volume = (fdim / 2.0) * std::log(std::acos(-1.0)) + fdim * std::log(radius) - std::lgamma(fdim / 2.0 + 1.0);
    return l_volume;
}

double calcKNNQueryVolume(const double* point, const GridAsSparseMatrix* fineMesh) {
    /*
    // 计算以查询点为圆心，查询点到网格中最远点的距离为半径的超球的体积
    const auto numNNZ = fineMesh->get_nnz_nums();
    const auto dim = fineMesh->get_dimensions();
    const auto vals = fineMesh->get_vals_();

    // 将网格数据拷贝至主机端
    auto* h_vals = new double[numNNZ * dim];
    CUDA_CHECK(cudaMemcpy(h_vals, vals, numNNZ * dim * sizeof(double), cudaMemcpyDeviceToHost));

    // 寻找最远点
    auto maxDist = 0.0;
    for (size_t i = 0; i < numNNZ; ++i) {
        if (const double dist = calcDistance(point, &h_vals[i * dim], dim); dist > maxDist) {
            maxDist = dist;
        }
    }

    // 清理数据
    delete [] h_vals;

    // 计算超球体积
    return calcSphereVolume(maxDist, dim);
    */
    return 1.0;
}

std::string formatLogVolume(double l_volume) {
    if (std::isinf(l_volume)) {
        return (l_volume < 0) ? "0" : "inf";
    }
    if (std::isnan(l_volume)) {
        return "nan";
    }

    double log10_val = l_volume / std::log(10.0);

    double exponent = std::floor(log10_val);
    double mantissa = std::pow(10.0, log10_val - exponent);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << mantissa
        << "E" << (exponent >= 0 ? "+" : "") << (long long)exponent;

    return oss.str();
}

std::string getQueryProfilerName(const QueryType type, const size_t N, const size_t D) {
    std::string profilerName;
    if (type == QueryType::POINT) {
        profilerName += "KNN";
    } else if (type == QueryType::RANGE) {
        profilerName += "Range";
    } else {
        throw std::invalid_argument("Invalid QueryType");
    }
    profilerName += "Query";
    profilerName += "N" + std::to_string(N);
    profilerName += "D" + std::to_string(D);
    return profilerName;
}

bool checkRangeQuery(const double* data, const double* lowBounds, const double* upBounds, const double* radius, const size_t* indexes, const size_t length, const size_t numNNZ, const size_t dim) {
    std::cout << "Checking Range Pruning correctness..." << std::endl;
    std::cout << "Data points: " << numNNZ << ", Dimensions: " << dim << ", Radius length: " << length << std::endl;

    // 准备数据
    double* h_data;
    double* h_lowBounds;
    double* h_upBounds;
    double* h_radius;
    size_t* h_indexes;

    cudaPointerAttributes dataAttr{}, lowBoundsAttr{}, upBoundsAttr{}, radiusAttr{}, indexesAttr{};
    CUDA_CHECK(cudaPointerGetAttributes(&dataAttr, data));
    CUDA_CHECK(cudaPointerGetAttributes(&lowBoundsAttr, lowBounds));
    CUDA_CHECK(cudaPointerGetAttributes(&upBoundsAttr, upBounds));
    CUDA_CHECK(cudaPointerGetAttributes(&radiusAttr, radius));
    CUDA_CHECK(cudaPointerGetAttributes(&indexesAttr, indexes));
    if (dataAttr.type == cudaMemoryTypeDevice) {
        h_data = new double[numNNZ * dim];
        CUDA_CHECK(cudaMemcpy(h_data, data, numNNZ * dim * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        h_data = const_cast<double*>(data);
    }
    if (lowBoundsAttr.type == cudaMemoryTypeDevice) {
        h_lowBounds = new double[dim];
        CUDA_CHECK(cudaMemcpy(h_lowBounds, lowBounds, dim * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        h_lowBounds = const_cast<double*>(lowBounds);
    }
    if (upBoundsAttr.type == cudaMemoryTypeDevice) {
        h_upBounds = new double[dim];
        CUDA_CHECK(cudaMemcpy(h_upBounds, upBounds, dim * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        h_upBounds = const_cast<double*>(upBounds);
    }
    if (radiusAttr.type == cudaMemoryTypeDevice) {
        h_radius = new double[length];
        CUDA_CHECK(cudaMemcpy(h_radius, radius, length * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        h_radius = const_cast<double*>(radius);
    }
    if (indexesAttr.type == cudaMemoryTypeDevice) {
        h_indexes = new size_t[numNNZ];
        CUDA_CHECK(cudaMemcpy(h_indexes, indexes, numNNZ * sizeof(size_t), cudaMemcpyDeviceToHost));
    } else {
        h_indexes = const_cast<size_t*>(indexes);
    }

    // 检查
    bool result = true;
    for (auto i = 0; i < numNNZ; ++i) {
        const auto index = h_indexes[i];
        const auto r = h_radius[index];
        bool inRange = true;
        for (size_t d = 0; d < dim; ++d) {
            const auto val = h_data[i * dim + d];
            if (val < h_lowBounds[d] - r || val > h_upBounds[d] + r) {
                inRange = false;
                std::cout << "Data point " << i << " with radius " << r << " is out of range in dimension " << d << ": value = " << val << ", lowBound = " << h_lowBounds[d] << ", upBound = " << h_upBounds[d] << std::endl;
                break;
            }
        }
        if (!inRange) {
            result = false;
            auto dist = 0.0;
            for (auto d = 0; d < dim; ++d) {
                auto diff = 0.0;
                if (h_data[i * dim + d] < h_lowBounds[d]) {
                    diff += h_lowBounds[d] - h_data[i * dim + d];
                } else if (h_data[i * dim + d] > h_upBounds[d]) {
                    diff += h_data[i * dim + d] - h_upBounds[d];
                }
                dist += diff * diff;
            }
            dist = sqrt(dist);
            std::cout << "  -> Distance to range: " << dist << ", Radius: " << r << std::endl;
            break;
        }
    }
    // 清理
    if (dataAttr.type == cudaMemoryTypeDevice) {
        delete [] h_data;
    }
    if (lowBoundsAttr.type == cudaMemoryTypeDevice) {
        delete [] h_lowBounds;
    }
    if (upBoundsAttr.type == cudaMemoryTypeDevice) {
        delete [] h_upBounds;
    }
    if (radiusAttr.type == cudaMemoryTypeDevice) {
        delete [] h_radius;
    }
    if (indexesAttr.type == cudaMemoryTypeDevice) {
        delete [] h_indexes;
    }
    return result;
}