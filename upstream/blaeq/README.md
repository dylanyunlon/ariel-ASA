# T-BLAEQ
The code to our submitted paper to VLDB 2026 titled T-BLAEQ: A Tensor-Based Multigrid Index for GPU-Accelerated Multi-dimensional Query Processing

The code is fully implemented in CUDA C/C++.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support

### Software
- **CUDA Toolkit**: 12.0 or higher
- **GCC/G++**: 12.0 or higher
- **CMake**: 4.0 or higher
- **cuVS**: CUDA Vector Search library ([https://github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs))

## Building

To build the project, use CMake:
```bash
mkdir build
cd build
cmake ..
make
```

## Usage

After building, run the program from the build directory:
```bash
./T-BLAEQ [OPTIONS]
```

### Command-Line Options

#### Required Options (choose one)
- `--build-index` - Build and save index for the specified dataset
- `--test-query` - Test queries on the specified dataset

#### General Options
- `-h, --help` - Print help message and exit
- `-d, --dataset TEXT` - Path to the dataset file
- `-i, --index-path TEXT` - Path to save/load index (default: `indexes/`)

#### Query Options
- `-f, --query-file TEXT` - Path to the query file
- `-q, --max-queries INT` - Maximum number of queries to process (default: 10)
- `-t, --query-type INT` - Type of query to test:
    - `0` - Range Query (default)
    - `1` - KNN Query
- `-k, --knn-k INT` - Number of neighbors for KNN query (default: 10)

### Examples

**Build an index:**
```bash
./T-BLAEQ --build-index -d dataset.txt -i indexes/
```

**Run range queries:**
```bash
./T-BLAEQ --test-query -i indexes/dataset_index -f queries.txt -t 0 -q 100
```

**Run KNN queries:**
```bash
./T-BLAEQ --test-query -i indexes/dataset_index -f queries.txt -t 1 -k 20 -q 100
```

## License

See [LICENSE](LICENSE) file for details.

## Third-Party Licenses

This project uses third-party libraries. See [NOTICE](NOTICE) and the [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/) directory for details.