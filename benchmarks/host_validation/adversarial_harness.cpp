// adversarial_harness.cpp
// ----------------------------------------------------------------------------
// 不再像上次那样在内部同时维护 AOS+SOA 两份"恰好一致"的数据。
// 真实系统里 grid->get_vals_() 只有【一份】，且整条 pipeline 是 AOS 布局。
// 本 harness 模拟真实情形：底层 buffer 是 AOS，SOA kernel 仍按 d-major 解读它，
// 看会不会算错；并模拟 2 层 multigrid 串联，看 SOA 输出喂回下一层会怎样。
// ----------------------------------------------------------------------------
#include <vector>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
using std::size_t;

int main(){
    printf("=== 对抗性验证：真实单一 AOS buffer 下 SOA kernel 的行为 ===\n\n");
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> du(1.0, 5.0);

    // ---- 真实约束：D != nnz，且只有一份 AOS buffer ----
    const size_t D = 3;
    const size_t gridNnz = 4;     // grid 物理 nnz
    const size_t gridRow = 4;     // 逻辑行数（这里恰好=nnz，先排除上次 bug 的干扰）
    const size_t Pnnz = 5;
    const size_t Pcol = 4;

    // grid 的唯一 buffer：AOS 布局 [i*D + d]
    std::vector<double> grid_buf_AOS(gridNnz * D);
    for (size_t i=0;i<gridNnz;i++) for (size_t d=0;d<D;d++) grid_buf_AOS[i*D+d]=du(rng);
    std::vector<size_t> grid_ids = {0,1,2,3};

    // P 的唯一 buffer：AOS 布局 [j*D + d]
    std::vector<double> P_buf_AOS(Pnnz * D);
    for (size_t j=0;j<Pnnz;j++) for (size_t d=0;d<D;d++) P_buf_AOS[j*D+d]=du(rng);
    std::vector<size_t> P_col_res = {0,1,2,4,5};   // 列偏移
    std::vector<size_t> P_row_ids = {0,1,2,3,4};

    // ---- expand（与源码一致）----
    std::vector<size_t> colInd, matPos, rowInd;
    for (size_t i=0;i<gridNnz;i++){
        size_t col=grid_ids[i];
        for (size_t j=P_col_res[col]; j<P_col_res[col+1]; ++j){
            colInd.push_back(i); matPos.push_back(j); rowInd.push_back(P_row_ids[j]);
        }
    }
    size_t N = colInd.size();
    printf("展开 element 数 N=%zu, D=%zu, gridNnz=%zu, Pnnz=%zu\n\n", N, D, gridNnz, Pnnz);

    // ---- AOS kernel：直接读 AOS buffer（这是正确基准）----
    std::vector<double> yAOS(N*D);
    for (size_t index=0; index<N*D; ++index){
        size_t e=index/D, dim=index%D;
        yAOS[index] = P_buf_AOS[matPos[e]*D+dim] * grid_buf_AOS[colInd[e]*D+dim];
    }

    // ---- SOA kernel：按 d-major 解读【同一份 AOS buffer】（真实系统的实际情形）----
    // 源码 SOA: yValue[d*N+e] = matrixData[d*nnzMatrix+matPos] * xValue[d*gridNnz+col]
    // 但 matrixData / xValue 实际是 AOS buffer！
    std::vector<double> ySOA_onAOSbuf(N*D);
    for (size_t e=0;e<N;e++) for (size_t d=0;d<D;d++){
        size_t out=d*N+e;
        ySOA_onAOSbuf[out] = P_buf_AOS[d*Pnnz+matPos[e]] * grid_buf_AOS[d*gridNnz+colInd[e]];
    }
    // 对拍：ySOA_onAOSbuf[d*N+e] vs yAOS[e*D+d]
    double err1=0; for(size_t e=0;e<N;e++)for(size_t d=0;d<D;d++)
        err1=std::max(err1,std::fabs(ySOA_onAOSbuf[d*N+e]-yAOS[e*D+d]));
    printf("[场景1] SOA kernel 直接跑在真实 AOS buffer 上:\n");
    printf("        max |SOA_onAOS - AOS| = %.3e  -> %s\n\n", err1,
           err1<1e-12 ? "巧合一致" : "✗ 算错！SOA 把 AOS buffer 误读成 d-major");

    // ---- 只有当 buffer 真的预转成 SOA 时，SOA kernel 才对 ----
    std::vector<double> grid_buf_SOA(gridNnz*D), P_buf_SOA(Pnnz*D);
    for(size_t i=0;i<gridNnz;i++)for(size_t d=0;d<D;d++) grid_buf_SOA[d*gridNnz+i]=grid_buf_AOS[i*D+d];
    for(size_t j=0;j<Pnnz;j++)for(size_t d=0;d<D;d++) P_buf_SOA[d*Pnnz+j]=P_buf_AOS[j*D+d];
    std::vector<double> ySOA_correct(N*D);
    for (size_t e=0;e<N;e++) for (size_t d=0;d<D;d++){
        size_t out=d*N+e;
        ySOA_correct[out]=P_buf_SOA[d*Pnnz+matPos[e]]*grid_buf_SOA[d*gridNnz+colInd[e]];
    }
    double err2=0; for(size_t e=0;e<N;e++)for(size_t d=0;d<D;d++)
        err2=std::max(err2,std::fabs(ySOA_correct[d*N+e]-yAOS[e*D+d]));
    printf("[场景2] 仅当 buffer 预转 SOA 后再跑 SOA kernel:\n");
    printf("        max diff = %.3e -> %s\n", err2, err2<1e-12?"✓ 一致(但需要预转换层，当前代码没有)":"✗");

    // ---- 场景3：多层串联的布局崩塌 ----
    // SOA kernel 输出 yValue 是 d-major。下一层 rangePruning 读 grid->get_vals_()[idx*D+d]（AOS!）
    // 把 d-major 的 yValue 当 AOS 解读会读到什么？
    printf("\n[场景3] SOA 输出(d-major) 喂回下一层 rangePruning(按 AOS [idx*D+d] 读):\n");
    // 取下一层第 0 个元素的"第 0 维"，AOS 解读 = yValue[0*D+0]=yValue[0]
    // 但 d-major 里 yValue[0] = (d=0,e=0)，碰巧对；yValue[1] 在 AOS 解读是(e=0,d=1)，
    // d-major 里却是 (d=0,e=1) —— 错位。
    bool layout_consistent = (N==1); // 只有 N==1 时两种布局才重合
    printf("        下一层按 AOS 读 SOA 输出: %s\n",
           layout_consistent ? "重合" : "✗ 布局错位，rangePruning 读到错误的维度值");
    printf("        (除非 N==1，d-major 与 AOS 在 N>1 时必然错位)\n");

    printf("\n========== 结论 ==========\n");
    printf("当前 M006 SOA 路径在生产中【不可用】，因为：\n");
    printf("1. 它假定输入 buffer 已是 SOA，但 pipeline 全程是 AOS，无预转换层。\n");
    printf("2. 它的输出是 d-major，但下一层所有 kernel 按 AOS 读，多层串联必崩。\n");
    printf("3. GridAsSparseMatrix 无 stride/layout 元数据，is_aos_ 仅是标志位，\n");
    printf("   set 了也不会真的改变 buffer 物理布局。\n");
    return 0;
}
