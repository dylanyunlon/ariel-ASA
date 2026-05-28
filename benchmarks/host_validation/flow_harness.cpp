// flow_harness.cpp
// ----------------------------------------------------------------------------
// 在无 GPU 环境下，用 host 串行逻辑复刻 BLAEQ multigrid 范围查询的完整数据通路，
// 并实跑 AOS 与 SOA(M006 新增)两条 SpMSpV 路径，逐元素对拍。
//
// 复刻的真实 kernel(语义一一对应源码):
//   rangePruningKernel        (RangePruning.cu)
//   gridCompactPrefixKernel   (GridCompact.cu, warp-shuffle 前缀和的 host 等价)
//   SpMSpVKernelAOS_v2        (SpMSpV.cuh:281)
//   SpMSpVKernelSOA_v2        (SpMSpV.cuh, M006 新增)
//   refactorKernel            (Refactor.cu)
//
// 注意: 这验证算法逻辑与数据流自洽性，不验证 GPU 性能 / CUDA API 行为。
// ----------------------------------------------------------------------------
#include <vector>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cstring>

using std::size_t;

// ===== 一个最细层网格之上的 CSC P-Tensor(把第 l 层簇映射到 l+1 层) =====
// 与 SparseTensorCscFormat 对齐: col_res(列偏移), row_ids, vals(nnz x D, AOS or SOA)
struct PTensor {
    size_t D;
    size_t row_nums;          // 下一层逻辑行数
    size_t col_nums;          // 当前层逻辑列数
    size_t nnz;               // 非零元数
    std::vector<size_t> col_res;   // size col_nums+1
    std::vector<size_t> row_ids;   // size nnz
    std::vector<double> vals_aos;  // nnz*D, [j*D+d]
    std::vector<double> vals_soa;  // nnz*D, [d*nnz+j]
};

// ===== Grid(GridAsSparseMatrix): nnz 个非零元, 每个 D 维, 带原始 id =====
struct Grid {
    size_t row;               // 逻辑行数
    size_t D;
    size_t nnz;
    bool   is_aos;
    std::vector<size_t> ids;       // nnz
    std::vector<double> vals_aos;  // nnz*D
    std::vector<double> vals_soa;  // nnz*D
    std::vector<double> radius;    // 每个逻辑行一个半径(按 id 索引)
};

static void aos_to_soa(const std::vector<double>& aos, std::vector<double>& soa, size_t nnz, size_t D){
    soa.assign(nnz*D, 0.0);
    for(size_t j=0;j<nnz;j++) for(size_t d=0;d<D;d++) soa[d*nnz+j]=aos[j*D+d];
}

// ---- kernel 1: rangePruning (RangePruning.cu) ----
// 点到查询矩形的距离平方 <= r^2 则保留
static std::vector<char> rangePruning(const Grid& g, const std::vector<double>& lo,
                                      const std::vector<double>& hi, size_t& selected){
    std::vector<char> mask(g.nnz,0);
    for(size_t idx=0; idx<g.nnz; ++idx){
        double dist=0.0;
        for(size_t d=0; d<g.D; ++d){
            double c=g.vals_aos[idx*g.D+d], diff=0.0;
            if(c<lo[d]) diff=lo[d]-c; else if(c>hi[d]) diff=c-hi[d];
            dist+=diff*diff;
        }
        double r=g.radius[g.ids[idx]];
        mask[idx]= (dist<=r*r)?1:0;
    }
    selected=std::count(mask.begin(),mask.end(),(char)1);
    return mask;
}

// ---- kernel 2: compactGrid (GridCompact.cu) ----
// 源码用 warp-shuffle 前缀和做 stream compaction; host 等价: 稳定保序压缩
static Grid compactGrid(const Grid& g, const std::vector<char>& mask, size_t validCount){
    Grid out; out.row=g.row; out.D=g.D; out.nnz=validCount; out.is_aos=g.is_aos;
    out.radius=g.radius;
    out.ids.resize(validCount); out.vals_aos.resize(validCount*g.D);
    size_t w=0;
    for(size_t i=0;i<g.nnz;i++) if(mask[i]){
        out.ids[w]=g.ids[i];
        for(size_t d=0;d<g.D;d++) out.vals_aos[w*g.D+d]=g.vals_aos[i*g.D+d];
        w++;
    }
    aos_to_soa(out.vals_aos,out.vals_soa,out.nnz,out.D);
    return out;
}

// ---- kernel 3a: SpMSpVKernelAOS_v2 (SpMSpV.cuh:281) ----
// 真实 v3 路径: 先在 host 展开 (element=每个匹配的 P 列项), 再 element-major 计算
struct Expanded { size_t numProcessed; std::vector<size_t> colInd, rowInd, matPos; };

static Expanded expand(const PTensor& P, const Grid& g){
    Expanded e; e.numProcessed=0;
    for(size_t i=0;i<g.nnz;i++){ size_t col=g.ids[i]; e.numProcessed += P.col_res[col+1]-P.col_res[col]; }
    e.colInd.reserve(e.numProcessed); e.rowInd.reserve(e.numProcessed); e.matPos.reserve(e.numProcessed);
    for(size_t i=0;i<g.nnz;i++){
        size_t col=g.ids[i];
        for(size_t j=P.col_res[col]; j<P.col_res[col+1]; ++j){
            e.colInd.push_back(i);            // 输入向量(grid)中的位置
            e.rowInd.push_back(P.row_ids[j]); // 输出行 id
            e.matPos.push_back(j);            // P 中原始位置
        }
    }
    return e;
}

static Grid sptm_aos(const PTensor& P, const Grid& g, const Expanded& e){
    Grid out; out.row=P.row_nums; out.D=g.D; out.nnz=e.numProcessed; out.is_aos=true;
    out.radius=g.radius;
    out.ids=e.rowInd;
    out.vals_aos.assign(e.numProcessed*g.D,0.0);
    size_t total=e.numProcessed*g.D;
    for(size_t index=0; index<total; ++index){          // 复刻 AOS_v2 kernel
        size_t elem=index/g.D, dim=index%g.D;
        size_t col=e.colInd[elem], mp=e.matPos[elem];
        out.vals_aos[index] = P.vals_aos[mp*g.D+dim] * g.vals_aos[col*g.D+dim];
    }
    aos_to_soa(out.vals_aos,out.vals_soa,out.nnz,out.D);
    return out;
}

// ---- kernel 3b: SpMSpVKernelSOA_v2 (M006 新增) ----
static Grid sptm_soa(const PTensor& P, const Grid& g, const Expanded& e){
    Grid out; out.row=P.row_nums; out.D=g.D; out.nnz=e.numProcessed; out.is_aos=false;
    out.radius=g.radius;
    out.ids=e.rowInd;
    std::vector<double> y(e.numProcessed*g.D,0.0);       // d-major
    size_t N=e.numProcessed, nnzM=P.nnz, rowsX=g.nnz; // FIX: stride=物理nnz,非逻辑row
    for(size_t elem=0; elem<N; ++elem){                  // 复刻 SOA_v2 kernel: 一线程一 element
        size_t col=e.colInd[elem], mp=e.matPos[elem];
        for(size_t d=0; d<g.D; ++d){
            size_t out_i=d*N+elem;
            y[out_i] = P.vals_soa[d*nnzM+mp] * g.vals_soa[d*rowsX+col];
        }
    }
    out.vals_soa=y;
    // 回填 AOS 视图以便对拍
    out.vals_aos.assign(N*g.D,0.0);
    for(size_t e2=0;e2<N;e2++) for(size_t d=0;d<g.D;d++) out.vals_aos[e2*g.D+d]=y[d*N+e2];
    return out;
}

// ---- kernel 4: refactorKernel (Refactor.cu) ---- id 经 map 重映射到下一层
static void refactor(Grid& g, const std::vector<size_t>& map){
    for(auto& v: g.ids) v=map[v];
}

int main(){
    printf("=== BLAEQ multigrid range-query 流程实跑 (host 复刻, AOS vs SOA) ===\n\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> du(0.0,10.0);

    const size_t D=3;
    // ---- 构造一个 2 层索引: Coreast(粗) --P0--> 细层 ----
    // 粗层网格: 4 个簇
    Grid coreast; coreast.row=4; coreast.D=D; coreast.nnz=4; coreast.is_aos=true;
    coreast.ids={0,1,2,3};
    coreast.vals_aos.resize(4*D);
    for(size_t i=0;i<4;i++) for(size_t d=0;d<D;d++) coreast.vals_aos[i*D+d]=du(rng);
    aos_to_soa(coreast.vals_aos,coreast.vals_soa,4,D);
    coreast.radius={3.0,3.0,3.0,3.0};   // 每个粗簇半径

    // P0: 4 列(粗簇) -> 共 6 个细行, 列展开
    PTensor P0; P0.D=D; P0.col_nums=4; P0.row_nums=6; P0.nnz=6;
    P0.col_res={0,2,3,5,6};              // 簇0->{0,1}, 簇1->{2}, 簇2->{3,4}, 簇3->{5}
    P0.row_ids={0,1,2,3,4,5};
    P0.vals_aos.resize(6*D);
    for(size_t j=0;j<6;j++) for(size_t d=0;d<D;d++) P0.vals_aos[j*D+d]=du(rng)*0.1+1.0; // 接近 1 的缩放
    aos_to_soa(P0.vals_aos,P0.vals_soa,6,D);
    std::vector<size_t> map_to_fine={0,1,2,3,4,5}; // 细层恒等 map

    // ---- 查询矩形 ----
    std::vector<double> lo={0,0,0}, hi={6,6,6};
    printf("查询矩形 lo=[0,0,0] hi=[6,6,6], 粗层 4 簇, D=%zu\n", D);

    auto run=[&](bool use_soa)->Grid{
        const char* tag = use_soa?"SOA":"AOS";
        printf("\n----- 路径: %s -----\n", tag);
        // level 0: 粗层
        Grid g=coreast; g.is_aos=!use_soa;
        size_t sel=0;
        auto mask=rangePruning(g,lo,hi,sel);
        printf("[L0 rangePruning] nnz %zu -> 保留 %zu\n", g.nnz, sel);
        Grid pruned=compactGrid(g,mask,sel);
        printf("[L0 compactGrid ] 压缩后 ids = ");
        for(auto id:pruned.ids) printf("%zu ",id); printf("\n");
        pruned.is_aos=!use_soa;
        auto e=expand(P0,pruned);
        printf("[L0 expand      ] 展开 element 数 = %zu\n", e.numProcessed);
        Grid fine = use_soa? sptm_soa(P0,pruned,e) : sptm_aos(P0,pruned,e);
        printf("[L0 SpMSpM_%s  ] 输出细层 nnz = %zu\n", tag, fine.nnz);
        refactor(fine,map_to_fine);
        printf("[L0 refactor    ] 重映射后 ids = ");
        for(auto id:fine.ids) printf("%zu ",id); printf("\n");
        return fine;
    };

    Grid aos_res=run(false);
    Grid soa_res=run(true);

    // ---- 对拍 ----
    printf("\n===== 对拍 AOS vs SOA =====\n");
    bool ok = (aos_res.nnz==soa_res.nnz) && (aos_res.ids==soa_res.ids);
    double maxerr=0;
    if(aos_res.nnz==soa_res.nnz){
        for(size_t i=0;i<aos_res.vals_aos.size();i++)
            maxerr=std::max(maxerr,std::fabs(aos_res.vals_aos[i]-soa_res.vals_aos[i]));
    } else ok=false;
    printf("nnz 一致: %s | ids 一致: %s | vals 最大误差: %.3e\n",
           (aos_res.nnz==soa_res.nnz)?"是":"否",
           (aos_res.ids==soa_res.ids)?"是":"否", maxerr);
    printf("\n结果: %s\n", (ok && maxerr<1e-12)?"✓ 全流程两条路径数值等价，数据流自洽":"✗ 不一致");
    return (ok && maxerr<1e-12)?0:1;
}
