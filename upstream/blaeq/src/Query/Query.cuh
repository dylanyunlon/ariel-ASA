//
// Created by cuda01 on 2026/1/7.
//

#ifndef BLAEQ_CUDA_QUERY_CUH
#define BLAEQ_CUDA_QUERY_CUH

#include <filesystem>
#include "src/Data_Structures/File.cuh"

struct QueryResult {
    ~QueryResult();

    QueryType type; // 查询类型
    int errorCode = 0; // 0表示成功，非0表示失败
    long totalTime = 0; // 总查询时间，单位为微秒
    size_t queryCount = 0; // 成功处理的查询数量
    std::vector<double> queryRangeVolume; // 每次查询的查询范围体积，以对数值存储s
    std::vector<size_t> fineMeshSize; // 每次查询得到的最细层网格大小（非零元数量）
    std::vector<GridAsSparseMatrix*> fineMesh; // 最终得到的最细层网格
};

class QueryHandler {
public:
    QueryHandler(const std::string& datasetPath);

    QueryHandler(const std::string& indexPath, bool loadFromIndex);

    void saveIndex(const std::string& indexPath);

    ~QueryHandler();

    int performQuery(const std::string& queryPath, QueryType qType,
                     long& totalTimeOut, size_t& queryCountOut,
                     int maxQueryCount = std::numeric_limits<int>::max(), size_t K = 0);

    QueryResult performQueryWithPreLoadPvals(const std::string& queryPath, QueryType qType, bool saveFineMesh = false,
                 int maxQueryCount = std::numeric_limits<int>::max(), size_t K = 0);

    void loadTensorValsToDevice();

    [[nodiscard]] size_t getSize() const;

    [[nodiscard]] size_t getDim() const;

private:
    void buildIndex();
    void loadMapsToDevice();
    void cleanup();

    void loadIndexFromFile(const std::string& indexPath);

    void saveSparseTensorCsc(const std::string& filepath, SparseTensorCscFormat* tensor);

    SparseTensorCscFormat* loadSparseTensorCsc(const std::string& filepath);

    void saveMaxRadius(const std::string& filepath, double* radius, size_t count);

    double* loadMaxRadius(const std::string& filepath, size_t& count);

    void saveMap(const std::string& filepath, size_t* map, size_t size);

    size_t* loadMap(const std::string& filepath, size_t& size);

    void saveCoreastMesh(const std::string& filepath, GridAsSparseMatrix* mesh);

    GridAsSparseMatrix* loadCoreastMesh(const std::string& filepath);

    void saveMetadata(const std::string& filepath);

    void loadMetadata(const std::string& filepath);

    // Dataset info
    Multidimensional_Arr dataset;
    size_t D;
    size_t N;

    // Index structure
    size_t GPU_Index_Height;
    size_t GPU_Index_Intervals;
    SparseTensorCscFormat** P_Tensors;
    double** Meshs_Max_Radius;
    size_t** maps;
    size_t** d_maps;
    GridAsSparseMatrix* Coreast_Mesh;
    std::vector<size_t> meshs_nums;
    double** d_P_Tensor_vals;

    // Parameters
    size_t* ratios;
    bool Is_AOS_Arch;
    bool isTensorInDevice = false;
};

double calcRangeQueryVolume(const double* lowBounds, const double* upBounds, size_t dim);

double calcKNNQueryVolume(const double* point, const GridAsSparseMatrix* fineMesh);

std::string formatLogVolume(double l_volume);

std::string getQueryProfilerName(QueryType type, size_t N, size_t D);

bool checkRangeQuery(const double* data, const double* lowBounds, const double* upBounds, const double* radius, const size_t* indexes, const size_t length, const size_t numNNZ, const size_t dim);

#endif //BLAEQ_CUDA_QUERY_CUH