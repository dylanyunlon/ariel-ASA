# 代码审查报告：M004–M006（第 2 位 Claude 自审）

> 标准：以《计算机程序设计艺术》作者的严苛度审查。每条结论都由
> `benchmarks/host_validation/` 下的可复跑 harness 实证，而非主观判断。
> 审查对象是 commit `f0a96eb`、`c1daa11`、`17ea9fe` 三次提交。

---

## 零、最重要的发现：上一轮的"数值等价 ✓"是无效证明

`flow_harness.cpp`（上一轮）在 harness 内部**同时维护 `vals_aos` 和 `vals_soa`
两份预先对齐的数据**，再断言两条路径输出一致。这是循环论证：它喂给 SOA kernel
的输入本就是正确转置的，自然算得对。

真实系统里 `GridAsSparseMatrix::get_vals_()` 只返回**一份** buffer，且整条
pipeline（建索引、KMeans、SpTSpM、compact）全程是 **AOS** 布局。`adversarial_harness.cpp`
喂真实情形（单一 AOS buffer，SOA kernel 按 d-major 解读它）：

```
[场景1] SOA kernel 直接跑在真实 AOS buffer 上:
        max |SOA_onAOS - AOS| = 1.391e+01  -> ✗ 算错
```

**结论：M006 的 SOA 路径在生产中根本不可用。** 详见 §1。

---

## 一、用户视角批判：会引入哪些新 bug

### BUG-1【致命】SOA 路径读错数据，且静默返回错误结果
- **现象**：`SpTSpMMultiplication_v3_SOA` 按 `xValue[d*gridNnz+col]` 解读
  buffer，但 `grid->get_vals_()` 是 AOS（`[i*D+d]`）。当 `D != gridNnz`（几乎总是）
  时读到完全错误的元素。harness 实测误差 13.9，且**不报错、不崩溃**——返回一个
  看似合理的错误网格。对查询系统，静默错误比崩溃危险得多。
- **触发条件**：任何走 SOA 分支的查询。当前 `is_aos_` 默认 true 所以分支走不到，
  但这意味着**我加的整个 SOA 代码路径从未被真实执行过**——一旦后人把某个 grid
  标记成 SOA（M007 做对比 panel 时极可能这么干），立刻触发。
- **根因**：`set_memory_arch(false)` 只翻转一个 bool，**不会真的转置 buffer**。
  布局是物理事实，标志位是元数据，二者脱节。

### BUG-2【高】多层 multigrid 串联时布局崩塌
- SOA kernel 输出 `yValue` 是 d-major，但下一层 `rangePruning` / `compactGrid`
  全部按 AOS `[idx*D+d]` 读。harness 场景3 证明：除非 `N==1`，d-major 与 AOS
  必然错位。即便 BUG-1 被修，**多层串联仍崩**，因为没有"输出转回 AOS"或"全程 SOA"
  的一致性保证。

### BUG-3【中】`numProcessedNonZero` 用 `unsigned int`，大数据集溢出
- harness 实测：`4.0e9 + 0.5e9`（unsigned int）= 2.05e8，回绕。GIST/SIFT 百万级
  点 + 高命中率范围查询，单次展开的非零元可能超 2^32。溢出后 `cudaMalloc` 分配过小
  buffer → kernel 越界写 → 显存损坏。**我照抄了原 v3 的 `unsigned int`，没修也没标注。**

### BUG-4【中】`new BenchScope` + `delete` 在异常路径下泄漏
- `Query.cu:483` 用裸 `new` 建 `BenchScope`，`:548` `delete`。但二者之间的循环里
  有 `throw std::invalid_argument("Invalid QueryType")`（:509）和多个可能抛出的
  CUDA 包装。一旦抛异常，`delete bench_scope` 被跳过 → **CUDA event 泄漏**
  （cudaEventCreate 的 handle 不回收）。RAII 的全部意义就是防这个，我却用裸指针
  破坏了它。讽刺：BenchScope 本身是 RAII，我却 heap-allocate 它。

### BUG-5【低】CUDA event 计时语义与"per-query"标注不符
- 注释说"records per-query CUDA-event latency"。但 BenchScope 的 start/stop event
  只 enqueue 在 query 循环的首尾，中间每个 kernel 后都有 `cudaDeviceSynchronize()`。
  所以测的是"含多次隐式同步的墙钟"，不是纯 GPU 时间。对 benchmark 数据，这个偏差
  会把 host 端开销算进 GPU 延迟。语义应改为 host steady_clock，或移除中间同步。

---

## 二、系统视角批判

### SYS-1 抽象层缺失：布局应是类型不变量，不是运行期 if
- 当前用 `if (grid->get_memory_arch())` 在 host 端二选一 kernel。这是把"内存布局"
  这个**编译期可知的不变量**降级成运行期分支。大厂做法（CUTLASS / CUB）：布局是
  **模板参数**（`cutlass::layout::RowMajor` vs `ColumnMajor`），编译期特化，零运行期
  开销，且类型系统强制 buffer 与 kernel 布局匹配——BUG-1 在那种设计下**编译不过**，
  而不是运行期静默出错。
- 参考：`cutlass/include/cutlass/layout/matrix.h` 的 `RowMajor`/`ColumnMajor` 标签类型。

### SYS-2 代码重复：v3 与 v3_SOA 有 ~60 行逐字重复的索引提取
- Step 1–3（计数、分配、提取 colInd/rowInd/matPos）两个函数完全一样。这违反
  DRY，且是 bug 温床——改一处忘改另一处。应抽出 `expand_indices()` 共享函数，
  两个 driver 只在 launch 配置上分叉。

### SYS-3 每次查询重复 malloc/free，无 buffer 复用
- v3 和 v3_SOA 每次调用都 `cudaMalloc` 5 个 buffer 再 `cudaFree`。在 2000-query
  benchmark 里这是 10000 次 malloc/free。大厂（vLLM PagedAttention、FairScale）用
  **预分配 + 内存池**。这不是我引入的，但我复制 v3 时把这个反模式也复制了，且没在
  PLAN 里标注为待优化项。

### SYS-4 错误处理：`CUDA_CHECK` 在库代码里直接 abort
- `CUDA_CHECK` 失败即 `exit`。在 benchmark/库语境下，单次查询失败不该杀死整个进程
  （后面还有 1999 个查询要跑）。应返回 error code（`QueryResult::errorCode` 已存在
  但 SOA driver 没用它）。

### SYS-5 缺单元测试 / CI，"验证"靠手写一次性 harness
- 我的 harness 是 `.cpp` 散落在仓库，没接入 CMake/CTest，没有断言框架。生产级应有
  `tests/` + GoogleTest + `ctest`，且 SOA 正确性测试**必须用真实单一 buffer**
  （否则重蹈零号覆辙）。

### SYS-6 CMake 的 T-BLAEQ-bench target 几乎全量复制 T-BLAEQ
- 我加的 bench target 把 source list、include、link、compile options 全抄一遍。
  改一处编译选项要改两处。应抽 `add_library` 公共对象库或用 `function()` 封装。

---

## 三、处置决定

| 项 | 决定 |
|----|------|
| BUG-1, BUG-2（SOA 不可用）| **撤销 SOA 运行期分派**（回退 Query.cu 的 if），把 SOA kernel 降级为"实验性、未接线"并在头注释明确警告：需要布局转换层 + 全程 SOA 才能用。不假装它能工作。 |
| BUG-3（溢出）| 在 v3_SOA 加 `size_t` 累加 + 溢出断言；原 v3 的同款缺陷在 PLAN 标注为 M00x 待修（不擅自改他人 baseline 代码的行为，但记录）。 |
| BUG-4（泄漏）| 改回栈上 RAII：用 `std::optional<BenchScope>` 或作用域块，杜绝裸 new/delete。 |
| BUG-5（计时语义）| 注释改为诚实描述；真正的 per-kernel 计时留作 M00x，标注。 |
| SYS-1 | 在 PLAN 记录："布局应模板化（CUTLASS 风格）"作为 M006 的正确重做方向，移交后续。 |
| SYS-2..6 | 在 PLAN 的"技术债"小节逐条登记，不在本次擅自大改 baseline。 |

**核心原则**：宁可交付一个**诚实标注为"实验性/未接线"的 SOA kernel** + 一条
**正确工作的 AOS 路径**，也不交付一个"看起来支持 SOA 实则静默算错"的产品。
后者在生产里会让用户拿到错误查询结果而不自知，这是查询系统最严重的失败模式。
