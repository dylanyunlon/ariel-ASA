#include "Data_Structures.cuh"
#include "src/func.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <assert.h>
#include <iterator>
#include <iomanip>
#include <numeric>
#include <cstring>


Multidimensional_Arr::Multidimensional_Arr(int input_N, int input_D) {
	D = input_D;
	N = input_N;
	data = new double[D * N];
}

Multidimensional_Arr::~Multidimensional_Arr() {
    delete[] data;
}

void Multidimensional_Arr::getRecord(int n, double* result, int* size) {
    return;
	// size = &D;
	// double* result_arr_ptr = (double*)malloc(D * sizeof(double));
	// for (int i = 0; i < D; i++) {
	// 	result_arr_ptr[i] = data[i * N + n];
	// }
	// result = result_arr_ptr;
}


GridAsSparseMatrix::GridAsSparseMatrix() : M_row_(0), M_dim_(0), M_nnz_nums_(0) {}


GridAsSparseMatrix::GridAsSparseMatrix(size_t M_row, size_t M_dim, size_t M_nnz_nums) 
    : M_row_(M_row), M_dim_(M_dim), M_nnz_nums_(M_nnz_nums) {}

GridAsSparseMatrix::GridAsSparseMatrix(const std::vector<Mul_Dim_Point>& data, const std::vector<size_t>& row_ids, size_t logic_row_nums, size_t M_dim){
    // 1. check whether data is empty 
    if (data.empty()) {
        M_dim_ = M_dim; M_nnz_nums_ = 0; M_row_ = logic_row_nums; ids_ = nullptr; vals_ = nullptr; return;
    }

    // do some assertions
    assert(logic_row_nums >= data.size());
    assert(data.size() == row_ids.size());

    // 2. set row_nums_ and dims_    
    M_row_ = logic_row_nums;
    M_dim_ = (data.size() > 0)? data[0].size() : 0;
        
    // 3. set M_nnz_nums_, ids_, vals_
    M_nnz_nums_ = row_ids.size();
    ids_ = new size_t[M_nnz_nums_];
    vals_ = new double[M_nnz_nums_ * M_dim_];

    // construct ids_
    std::copy(row_ids.begin(), row_ids.end(), ids_);

    // construct vals_
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i].empty()){
            assert(false);
        }
        assert(data[i].size() == M_dim_);
        for(size_t j = 0; j < M_dim_; j++){
            vals_[i * M_dim_ + j] = data[i * M_dim_][j];
        }
    }
}


GridAsSparseMatrix::GridAsSparseMatrix(const std::vector<Mul_Dim_Point>& data, size_t begin_pos, size_t last_pos){
    // 1. check whether data is empty 
    if (data.empty()) {
        M_row_ = 0; M_dim_ = 0; M_nnz_nums_ = 0; return;
    }

    assert(begin_pos <= last_pos);

    assert(last_pos <= data.size());

    // 2. set row_nums_ and dims_    
    M_row_ = data.size();
    M_dim_ = (data.size() > 0)? data[0].size() : 0;
    M_nnz_nums_ = last_pos - begin_pos;
    std::cout << "Coreast Mesh nnz_nums are " << M_nnz_nums_ << std::endl;
    // 3. set M_nnz_nums_, ids_, vals_
    ids_ = new size_t[M_nnz_nums_]();
    vals_ = new double[M_nnz_nums_ * M_dim_]();
    size_t write_idx = 0;
    for (size_t i = begin_pos; i < last_pos; i++) {
        if (data[i].empty()){
            assert(false);
        }
        // add nnz
        ids_[write_idx] = i;
        for(size_t j = 0; j < M_dim_; j++){
            vals_[write_idx * M_dim_ + j] = data[i][j];
        }
        write_idx++;
    }
}
    

// 拷贝构造函数
GridAsSparseMatrix::GridAsSparseMatrix(const GridAsSparseMatrix& other)
    : M_row_(other.M_row_), M_dim_(other.M_dim_), M_nnz_nums_(other.M_nnz_nums_),
      ids_(other.ids_), vals_(other.vals_) {}


GridAsSparseMatrix& GridAsSparseMatrix::operator=(const GridAsSparseMatrix& other) {
    if (this != &other) {
        M_row_ = other.M_row_;
        M_dim_ = other.M_dim_;
        M_nnz_nums_ = other.M_nnz_nums_;
        ids_ = other.ids_;
        vals_ = other.vals_;
    }
    return *this;
}


GridAsSparseMatrix::GridAsSparseMatrix(GridAsSparseMatrix&& other) noexcept
    : M_row_(other.M_row_), M_dim_(other.M_dim_), M_nnz_nums_(other.M_nnz_nums_),
      ids_(std::move(other.ids_)), vals_(std::move(other.vals_)) {
    other.M_row_ = 0;
    other.M_dim_ = 0;
    other.M_nnz_nums_ = 0;
}


GridAsSparseMatrix& GridAsSparseMatrix::operator=(GridAsSparseMatrix&& other) noexcept {
    if (this != &other) {
        M_row_ = other.M_row_;
        M_dim_ = other.M_dim_;
        M_nnz_nums_ = other.M_nnz_nums_;
        ids_ = std::move(other.ids_);
        vals_ = std::move(other.vals_);
        other.M_row_ = 0;
        other.M_dim_ = 0;
        other.M_nnz_nums_ = 0;
    }
    return *this;
}

// destructor !!!
GridAsSparseMatrix::~GridAsSparseMatrix(){
    delete[] ids_;
    delete[] vals_;
}

void GridAsSparseMatrix::Load_Coreast_Mesh_to_DRAM(){
    // assert(d_M_coreast_layer_whole_vals_ == nullptr);
    // size_t D = M_dim_;
    // size_t nnz_nums = M_nnz_nums_;
    // size_t total    = D * nnz_nums;          // 元素个数
    // // 1.allocate memory for d_vals_ptr and error check
    // cudaError_t err = cudaMalloc(&d_M_coreast_layer_whole_vals_, total * sizeof(double));
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "cudaMalloc failed (%s): size=%zu doubles\n",
    //             cudaGetErrorString(err), total);
    //     throw std::bad_alloc();             
    // }

    // // 2. get corresponding h_vals_ptr
    // double* h_M_coreast_layer_whole_vals = vals_;


    // // 3. copy and error check
    // err = cudaMemcpy(d_M_coreast_layer_whole_vals_, h_M_coreast_layer_whole_vals, total * sizeof(double), cudaMemcpyHostToDevice);
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "cudaMemcpy H2D failed (%s)\n", cudaGetErrorString(err));
    //     cudaFree(d_M_coreast_layer_whole_vals_);
    //     d_M_coreast_layer_whole_vals_ = nullptr;
    //     delete[] h_M_coreast_layer_whole_vals;
    //     throw std::runtime_error("HostToDevice copy failed");
    // }

}

void GridAsSparseMatrix::set_ids_using_vec(std::vector<size_t>& ids){
    assert(ids_ == nullptr);
    // 2. 按新长度重新分配
    size_t M_nnz_nums = ids.size();
    ids_ = new size_t[M_nnz_nums];

    // 3. 深拷贝 vector 内容到 ids_
    std::copy(ids.begin(), ids.end(), ids_);
}


SparseTensorCooFormat::SparseTensorCooFormat(size_t D, size_t row_nums, size_t col_nums){
    D_ = D;
    row_nums_ = row_nums;
    col_nums_ = col_nums;
    nnz_nums_ = row_nums_;
    row_ids_ = new size_t[nnz_nums_]();
    col_ids_ = new size_t[nnz_nums_]();
    vals_ = new double[nnz_nums_ * D_]();
}

SparseTensorCooFormat::~SparseTensorCooFormat(){
    delete[] row_ids_;
    delete[] col_ids_;
    delete[] vals_;
}

// for SparseTensorCooFormat debug use !!!
void SparseTensorCooFormat::display(const std::string& filename){
    std::ofstream outFile(filename);
    // 1. check ofstream and set double-type precision
    if (!outFile.is_open()) {
        throw std::runtime_error("can't open the specific file : " + filename);
    }
    outFile << std::fixed << std::setprecision(3);

    // 2. init sparse_int_map
    std::vector<std::vector<int>> sparse_int_map(row_nums_, std::vector<int>(col_nums_, -1));
    for(size_t i = 0; i < nnz_nums_; i++){
        size_t nnz_row_id = row_ids_[i];
        size_t nnz_col_id = col_ids_[i];
        sparse_int_map[nnz_row_id][nnz_col_id] = static_cast<int>(i);
    }

    // 3. for each cell, specially handle nnz elements !!!s
    for(size_t i = 0; i < row_nums_; i++){
        outFile << "{";
        for(size_t j = 0; j < col_nums_; j++){
            if(sparse_int_map[i][j] >= 0){
                size_t nnz_idx = static_cast<size_t>(sparse_int_map[i][j]);
                outFile << "[";
                for(size_t k = 0; k < D_; k++){
                    outFile << vals_[nnz_idx * D_ + k];
                    if(k + 1 == D_) break;
                    outFile << ", ";
                }
                outFile << "]";
            }
            else{
                outFile << "[0.0, ..., 0.0]"; 
            }
            if(j + 1 == col_nums_) break;
            outFile << ", ";
        }
        outFile << "}\n";
    }
    
    // 4. close the file 
    outFile.close();
}


void SparseTensorCooFormat::insert_one_nnz(size_t nnz_row_coordinate, size_t nnz_col_coordinate, double* val_vector){
    row_ids_[curr_write_idx_] = nnz_row_coordinate;
    col_ids_[curr_write_idx_] = nnz_col_coordinate;
    for(size_t i = 0; i < D_; i++){
        vals_[curr_write_idx_ * D_ + i] = val_vector[i];
    }
    curr_write_idx_++;
}

SparseTensorCscFormat::SparseTensorCscFormat(size_t D, size_t row_nums, size_t col_nums, std::vector<size_t>& nnz_per_col_vec){
    // 1. specific data members and new memory
    D_ = D;
    row_nums_ = row_nums;
    col_nums_ = col_nums;
    nnz_nums_ = row_nums;
    row_ids_ = new size_t[nnz_nums_]();
    nnz_per_col_ = new size_t[col_nums_]();
    col_res_ = new size_t[col_nums_ + 1]();
    vals_ = new double[nnz_nums_ * D_]();
    assert(col_nums_ == nnz_per_col_vec.size());

    // 2. get nnz_per_col_array
    std::copy(nnz_per_col_vec.begin(), nnz_per_col_vec.end(), nnz_per_col_);

    // 3. get col_res_array
    for(size_t i = 1; i < col_nums_ + 1; i++){
        col_res_[i] = col_res_[i-1] + nnz_per_col_[i-1];
    }

    // 4. get row_ids_array
    std::iota(row_ids_, row_ids_ + nnz_nums_, 0);

}

 // xak note :: csc class func declaration !!!
 // xak review :: unfinish !!!
SparseTensorCscFormat::SparseTensorCscFormat(SparseTensorCooFormat* coo){
    D_ = coo->D_;
    row_nums_ = coo->row_nums_;
    col_nums_ = coo->col_nums_;
    nnz_nums_ = coo->nnz_nums_;
    row_ids_ = new size_t[nnz_nums_]();
    nnz_per_col_ = new size_t[col_nums_]();
    col_res_ = new size_t[col_nums_ + 1]();
    vals_ = new double[nnz_nums_ * D_]();
    // 1. get nnz_per_col_
    for(size_t i = 0; i < nnz_nums_; i++){
        size_t nnz_col_id = coo->col_ids_[i];
        nnz_per_col_[nnz_col_id]++;
    }
    // 2. nnz_per_col_ 2 col_res_
    for(size_t i = 1; i < col_nums_ + 1; i++){
        col_res_[i] = col_res_[i-1] + nnz_per_col_[i-1];
    }

    // 3. handle row_ids_ and vals_
    // xak note :: row_ids_ is ascending !!!
    size_t* col_ptr = new size_t[col_nums_ + 1];   // allocate tmp col_ptr
    std::copy(col_res_, col_res_ + col_nums_ + 1, col_ptr);  // then copy

   
    for (size_t i = 0; i < nnz_nums_; ++i) {
        size_t col = coo->col_ids_[i];
        // xak note :: get one nnz is csc idx !!!
        size_t idx = col_ptr[col]++;
        row_ids_[idx] = coo->row_ids_[i];
        for(size_t j = 0; j < D_; j++){
            vals_[idx * D_ + j] = coo->vals_[i * D_ + j];
        }
    }

    delete[] col_ptr;   // release col_ptr
}

SparseTensorCscFormat::~SparseTensorCscFormat(){
    delete[] row_ids_;                   // nnz row id
    delete[] nnz_per_col_;               // nnz per col
    delete[] col_res_;                   // col res
    delete[] vals_;                      // corresponding nnzs      
}


void SparseTensorCscFormat::Insert_One_Batch(double* P_vals_tmp, size_t begin_pos, size_t end_pos){
    assert(begin_pos <= end_pos);
    // 1. get len
    size_t copy_len = end_pos - begin_pos;

    // 2. memcpy
    double* dst = vals_ + begin_pos * D_;          // 目标起始地址
    const double* src = P_vals_tmp;                       // 源起始地址
    std::memcpy(dst, src, copy_len * D_ * sizeof(double));
}

void SparseTensorCscFormat::Load_To_File(std::string& base_dir, size_t idx){
    // 1. handle base_dir, to prevent corner case.
    if (!base_dir.empty() && base_dir.back() != '/') base_dir += '/';

    // 2 construct file name
    std::ostringstream oss;
    oss << base_dir << idx << '_' << (idx + 1);

    /* 3. 打开文件并写入 */
    std::ofstream fout(oss.str());
    if (!fout.is_open())
        throw std::runtime_error("Load_To_File: cannot create " + oss.str());

    fout << std::fixed << std::setprecision(6);

    /* ---------- 小矩阵：nnz < 100，按行打印 ---------- */
    if (nnz_nums_ < 100)
    {
        std::vector<int> SparseTensorAsMatrix_Bitmap(row_nums_ * col_nums_, -1);

        // get bit_map
        for (size_t i = 0; i < col_nums_; i++) {
            size_t start_pos = col_res_[i];
            size_t end_pos   = col_res_[i + 1];
            for (size_t j = start_pos; j < end_pos; j++) {
                size_t row_id = row_ids_[j];
                SparseTensorAsMatrix_Bitmap[row_id * col_nums_ + i] = static_cast<int>(j);
            }
        }

        // loop Bitmap
        for(size_t i = 0; i < row_nums_; i++){
            fout << "{";
            for(size_t j = 0; j < col_nums_; j++){
                if(SparseTensorAsMatrix_Bitmap[i * col_nums_ + j] == -1){
                    fout << '[';
                    if (D_ > 0) fout << '0';
                    for (size_t k = 1; k < D_; k++) fout << ",0";
                    fout << ']';
                }
                else{
                    size_t nnz_in_vals_ptr = SparseTensorAsMatrix_Bitmap[i * col_nums_ + j];
                    fout << "[";
                    if(D_ > 0) fout << vals_[nnz_in_vals_ptr * D_ + 0];
                    for(size_t k = 1; k < D_; k++) fout << "," << vals_[nnz_in_vals_ptr * D_ + k];
                    fout << "]";
                }
                fout << ", ";
            }
            fout << "}\n";
        }

   
    }
    /* ---------- 大矩阵：保持原来的三元组格式 ---------- */
    else
    {
        for (size_t c = 0; c < col_nums_; ++c) {
            size_t start = col_res_[c];
            size_t end = col_res_[c + 1];
            for (size_t idx = start; idx < end; ++idx) {
                size_t r = row_ids_[idx];
                const double* v = vals_ + idx * D_;
                fout << c << ' ' << r;
                for (size_t d = 0; d < D_; ++d) fout << ' ' << v[d];
                fout << '\n';
            }
            fout << "\n";
        }
    }
    fout.close();
}


// for SparseTensorCooFormat debug use !!!
void SparseTensorCscFormat::display(const std::string& filename){
    std::ofstream outFile(filename);
    // 1. check ofstream and set double-type precision
    if (!outFile.is_open()) {
        throw std::runtime_error("can't open the specific file : " + filename);
    }
    outFile << std::fixed << std::setprecision(3);

    // 2. init sparse_int_map
    std::vector<std::vector<int>> sparse_int_map(row_nums_, std::vector<int>(col_nums_, -1));
    // col_res_[i]  0~i-1 total nnz
    for(size_t i = 1; i < col_nums_ + 1; i++){
        size_t begin_pos = col_res_[i-1];
        size_t last_pos = col_res_[i];
        assert(begin_pos <= last_pos);
        for(size_t j = begin_pos; j < last_pos; j++){
            size_t row_id = row_ids_[j];
            size_t col_id = i-1;
            sparse_int_map[row_id][col_id] = static_cast<int>(j);
        }
    }

    // 3. for each cell, specially handle nnz elements !!!s
    for(size_t i = 0; i < row_nums_; i++){
        outFile << "{";
        for(size_t j = 0; j < col_nums_; j++){
            if(sparse_int_map[i][j] >= 0){
                size_t nnz_idx = static_cast<size_t>(sparse_int_map[i][j]);
                assert(nnz_idx < nnz_nums_);
                outFile << "[";
                for(size_t k = 0; k < D_; k++){
                    outFile << vals_[nnz_idx * D_ + k];
                    if(k + 1 == D_) break;
                    outFile << ", ";
                }
                outFile << "]";
            }
            else{
                outFile << "[0.0, ..., 0.0]"; 
            }
            if(j + 1 == col_nums_) break;
            outFile << ", ";
        }
        outFile << "}\n";
    }
    
    // 4. close the file 
    outFile.close();
}


bool SparseTensorConverter::Verify_Coo_Equal_Csc(SparseTensorCooFormat *coo, SparseTensorCscFormat *csc){
    if(coo->D_ != csc->D_) return false;
    if(coo->nnz_nums_ != csc->nnz_nums_) return false;
    if(coo->row_nums_ != csc->row_nums_) return false;
    if(coo->col_nums_ != csc->col_nums_) return false;
    size_t row_nums = coo->row_nums_;
    size_t col_nums = csc->col_nums_;
    size_t D = csc->D_;

    // 1. coo csc nnz bitmap init, save the corresponding point format idx
    std::vector<std::vector<int>> coo_sparse_int_map(row_nums, std::vector<int>(col_nums, -1));
    std::vector<std::vector<int>> csc_sparse_int_map(row_nums, std::vector<int>(col_nums, -1));

    // std::ofstream of("csc_col_res_array");
    // debug_tool::print_vector(csc->col_res_, "Debug Csc Format Col Res", ", ", of);
    // of.close();

    // 2. fulfill the bitmap
    // 2.1. fulfill the coo_map
    for(size_t i = 0; i < coo->nnz_nums_; i++){
        size_t coo_row_id = coo->row_ids_[i];
        size_t coo_col_id = coo->col_ids_[i];
        coo_sparse_int_map[coo_row_id][coo_col_id] = static_cast<int>(i);
    }
    // 2.2. fulfill the csc_map
    for(size_t i = 1; i < csc->col_nums_ + 1; i++){
        size_t begin_pos = csc->col_res_[i-1];
        size_t last_pos = csc->col_res_[i];
        assert(last_pos >= begin_pos);
        for(size_t j = begin_pos; j < last_pos; j++){
            size_t csc_row_id = csc->row_ids_[j];
            size_t csc_col_id = i-1;
            csc_sparse_int_map[csc_row_id][csc_col_id] = static_cast<int>(j);
        }
    }
    // 3. loop*2 for sparse map
    for(size_t i = 0; i < row_nums; i++){
        for(size_t j = 0; j < col_nums; j++){
            if(coo_sparse_int_map[i][j] == -1 || csc_sparse_int_map[i][j] == -1){
                if(coo_sparse_int_map[i][j] != csc_sparse_int_map[i][j]){
                    std::cout << coo_sparse_int_map[i][j] << " " << csc_sparse_int_map[i][j] << std::endl;
                }
                assert(coo_sparse_int_map[i][j] == csc_sparse_int_map[i][j]);
                continue;
            }
            else{
                size_t coo_val_idx = static_cast<size_t>(coo_sparse_int_map[i][j]);
                size_t csc_val_idx = static_cast<size_t>(csc_sparse_int_map[i][j]);
                for(size_t k = 0; k < D; k++){
                    double coo_val = coo->vals_[coo_val_idx * D + k];
                    double csc_val = csc->vals_[csc_val_idx * D + k];
                    if(!Comp::isZero(coo_val - csc_val)) return false;
                }

            }
            
        }
    }

    // 4. all is equal, so return equal
    return true;
}