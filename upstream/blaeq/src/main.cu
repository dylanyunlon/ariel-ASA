#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <thrust/device_vector.h>
#include "src/Test/QueryTest.cuh"
#include "CLI11.hpp"


#define BOOST_DISABLE_CURRENT_LOCATION


std::unordered_map<int, std::string> qtype = {
    {0, "Range_Query"},
    {1, "KNN_Query"}
};


int main(int argc, char** argv) {

    CLI::App app{"Index Builder and Query Tester"};

    bool buildIndexFlag = false;
    bool testQueryFlag = false;
    auto* group = app.add_option_group("operation");
    group->add_flag("--build-index", buildIndexFlag, "Build and save index for the specified dataset");
    group->add_flag("--test-query", testQueryFlag, "Test queries on the specified dataset");
    group->require_option(1, 1);

    std::string datasetPath;
    app.add_option("-d,--dataset", datasetPath, "Path to the dataset file");

    std::string indexPath = "indexes/";
    app.add_option("-i,--index-path", indexPath, "Path to save/load index (default : indexes/)");

    std::string queryFilePath;
    app.add_option("-f,--query-file", queryFilePath, "Path to the query file");

    int maxQueryCount = 10;
    app.add_option("-q,--max-queries", maxQueryCount, "Maximum number of queries to process (default: 10)");

    int queryType = 0;
    app.add_option("-t,--query-type", queryType, "Type of query to test: 0 for Range Query, 1 for KNN Query (default: 0)");

    int k = 10;
    app.add_option("-k,--knn-k", k, "Number of neighbors for KNN query (default: 10)");

    CLI11_PARSE(app, argc, argv);

    if (buildIndexFlag) {
        QueryHandler handler(datasetPath);
        handler.saveIndex(indexPath);
    }
    if (testQueryFlag) {
        QueryHandler handler(indexPath, true);
        QueryResult result;
        if (queryType == 0) {
            result = handler.performQueryWithPreLoadPvals(queryFilePath, QueryType::RANGE, false, maxQueryCount);
        } else if (queryType == 1) {
            result = handler.performQueryWithPreLoadPvals(queryFilePath, QueryType::POINT, false, maxQueryCount, k);
        } else {
            throw std::runtime_error("Unsupported query type: " + std::to_string(queryType));
        }
        std::cout << "Completed: " << result.queryCount << " queries." << std::endl;
        std::cout << "Total: " << result.totalTime << " ms." << std::endl;
        std::cout << "Average: " << (result.totalTime / result.queryCount) << " ms per query." << std::endl;
    }
    return 0;
}


