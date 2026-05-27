#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
// in this file, we will have four data structures
// 1.Multidimensional_Arr
// 2.GridAsSparseMatrix
// 3.SparseTensorCooFormat
// 4.SparseTensorCscFormat

class Multidimensional_Arr {
public:
	int D;
	int N;
	double* data = nullptr;
    Multidimensional_Arr() = default;
	Multidimensional_Arr(int N, int D);
	~Multidimensional_Arr();
    Multidimensional_Arr& operator=(const Multidimensional_Arr& other) {
        if (this != &other) {
            // 1. release old resource
            delete[] data;

            // 2. allocate new and copy
            N = other.N;
            D = other.D;
            data = new double[N * D];
            std::copy(other.data, other.data + N * D, data);
        }
        return *this;
    }

	void getRecord(int n, double* result, int* size);
};

class GridAsSparseMatrix {
public:
    using Mul_Dim_Point = std::vector<double>;
    // constructor !!!
    GridAsSparseMatrix();
    GridAsSparseMatrix(size_t rows, size_t dim, size_t nnz);
    GridAsSparseMatrix(const std::vector<Mul_Dim_Point>& data, const std::vector<size_t>& row_ids, size_t logic_row_nums, size_t M_dim);
    GridAsSparseMatrix(const std::vector<Mul_Dim_Point>& data, size_t begin_pos, size_t last_pos);
    GridAsSparseMatrix(size_t length, size_t dim, size_t nnz, size_t* indexes, double* data) 
    {
        this->M_row_ = length;
        this->M_dim_ = dim;
        this->M_nnz_nums_ = nnz;
        this->ids_ = indexes;
        this->vals_ = data;
    }

    // copy constrctor !!!
    GridAsSparseMatrix(const GridAsSparseMatrix& other);
    GridAsSparseMatrix& operator=(const GridAsSparseMatrix& other);
    GridAsSparseMatrix(GridAsSparseMatrix&& other) noexcept;
    GridAsSparseMatrix& operator=(GridAsSparseMatrix&& other) noexcept;
    
    // destructor !!!
    ~GridAsSparseMatrix();

    void Load_Coreast_Mesh_to_DRAM();
    
    // get-method
    bool get_memory_arch() const { return is_aos_; }
    size_t get_num_rows() const { return M_row_; }
    size_t get_dimensions() const { return M_dim_; }
    size_t get_nnz_nums() const { return M_nnz_nums_; }
    
    size_t* get_ids_() const { return ids_; }
    double* get_vals_() const { return vals_; }

    // set-method
    void set_memory_arch(bool is_aos) { is_aos_ = is_aos; }
    void set_ids(size_t* ids) { ids_ = ids; }
    void set_vals(double* vals) { vals_ = vals; }
    void set_ids_using_vec(std::vector<size_t>& ids);
    void set_nnz_vals_P_col_ids_using_vec(std::vector<size_t>& nnz_vals_P_col_ids);

    void free_DRAM();

    void pre_allocate_d_vals();

    void pre_allocate_h_vals();
    
// xak note :: when we handle the GridAsSparseMatrix, we regard matrix as vector, whose element is vector
// so we regard it as sparse vector !!!
private:
    bool is_aos_ = true;
    size_t M_row_;                          // logic vec length
    size_t M_dim_;                          // dim
    size_t M_nnz_nums_;                     // nnz nums
    size_t* ids_ = nullptr;                 // nnz vec_id
    double* vals_ = nullptr;                // nnz itself
};




// to support sparse vector,  fulfill the data structure - SparseMatrix
// matrix to csc

class SparseTensorCooFormat
{
public:
    friend class SparseTensorCscFormat;
    friend class SparseTensorConverter;
    using Mul_Dim_Point = std::vector<double>;  //
    using Mul_Dim_Point_Coordinate = std::pair<size_t, size_t>;
    // constructor
    // xak review : column first
    explicit SparseTensorCooFormat(size_t D, size_t row_nums, size_t col_nums);
    ~SparseTensorCooFormat();
    void display(const std::string& filename);
    void insert_one_nnz(size_t nnz_row_coordinate, size_t nnz_col_coordinate, double* val_vector);

    


private:
    size_t D_;                               // depth
    size_t curr_write_idx_ = 0;              // simulation vector push_back
    size_t row_nums_;                        // logic row nums
    size_t col_nums_;                        // logic column nums
    size_t nnz_nums_;                        // nnz nums
    size_t* row_ids_;                        // nnz row id
    size_t* col_ids_;                        // nnz col id
    double* vals_;                           // corresponding nnzs
};



class SparseTensorCscFormat
{
public:
    friend class SparseTensorConverter;
    // constructor
    // xak review : column first
    explicit SparseTensorCscFormat(size_t D, size_t row_nums, size_t col_nums) : D_(D), row_nums_(row_nums), col_nums_(col_nums)  {}
    explicit SparseTensorCscFormat(size_t D, size_t row_nums, size_t col_nums, std::vector<size_t>& nnz_per_col_vec);
    explicit SparseTensorCscFormat(SparseTensorCooFormat* coo);
    ~SparseTensorCscFormat();
    void display(const std::string& filename);

    // Get methods for private members
    bool get_memory_arch() const { return is_aos_; }
    size_t get_dim() const { return D_; }
    size_t get_row_nums() const { return row_nums_; }
    size_t get_col_nums() const { return col_nums_; }
    size_t get_nnz_nums() const { return nnz_nums_; }
    
    const size_t* get_row_ids() const { return row_ids_; }
    const size_t* get_nnz_per_col() const { return nnz_per_col_; }
    const size_t* get_col_res() const { return col_res_; }
    const double* get_vals() const { return vals_; }

    void Insert_One_Batch(double* P_vals_tmp, size_t begin_pos, size_t end_pos);

    void Load_To_File(std::string& base_dir, size_t idx);
    


private:
    bool is_aos_ = true;                   // memory arch
    size_t D_;                             // depth
    size_t row_nums_;                      // logic row nums
    size_t col_nums_;                      // logic column nums
    size_t nnz_nums_;                      // nnz nums  
    size_t* row_ids_ = nullptr;            // nnz row id
    size_t* nnz_per_col_ = nullptr;        // nnz per col
    size_t* col_res_ = nullptr;            // col res
    double* vals_ = nullptr;         // corresponding nnzs
};



class SparseTensorConverter
{
public:
    static SparseTensorCscFormat* Convert_Coo2Csc(SparseTensorCooFormat *coo){
        assert(coo != nullptr);
        SparseTensorCscFormat* csc = new SparseTensorCscFormat(coo);
        bool res = Verify_Coo_Equal_Csc(coo, csc);
        assert(res);
        for(size_t i = 0; i < csc->nnz_nums_; i++){
            assert(csc->row_ids_[i] == i);
        }
        delete coo;
        return csc;
    }

    static bool Verify_Coo_Equal_Csc(SparseTensorCooFormat *coo, SparseTensorCscFormat *csc);

private:
};
#endif