# MUSA (摩尔线程)

`BACKEND=MUSA` / `FLAGSPARSE_BACKEND=mthreads`。**除 CUDA 外唯一有 adaptor 实现的后端**。

## 环境检查
> 这一层离不开可导入的 `flagsparse` 包：内核是 **re-export** 而不是拷贝，所以
> `FLAGSPARSE_PYTHON_SRC` 指错、或机器上有旧副本，症状分别是 `EXECUTION_FAILED`
> 和 `CompilationError`。依赖关系、版本配套和部署清单见
> [README.md#这个库离不开-flagsparse](README.md)。


```bash
mthreads-gmi                                    # 或 musa-smi，视驱动版本
ls $MUSA_HOME/lib64/libmusart.so $MUSA_HOME/include/musa.h
python3 -c "import torch, torch_musa; print(torch.__version__, torch.musa.is_available())"
export FLAGSPARSE_BACKEND=mthreads
```

CMake 要三样，缺任一项会**指名报错**而不是在链接期报缺符号：
`$MUSA_HOME/lib64/libmusart.so`、`libmusa.so`、`$MUSA_HOME/include/musa.h`
（`MUSA_HOME` 默认 `/usr/local/musa`）。

## 构建

```bash
cmake -S . -B build -G Ninja -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa \
      -DFLAGSPARSE_PYTHON_SRC=/path/to/FlagSparse/src -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`src/adaptor/backend/musa/adaptor.cpp` 是 CUDA 那份**加一个符号前缀**——摩尔线程的
driver API 与 CUDA 逐符号镜像，所以两者共用 `backend/_template.inc`，25 行。这也是
填其他预留槽位的模板。

## 跑测试

同 [README.md](README.md) 的通用流程。**先跑 `ctest -R pytest`**：MUSA 上 Triton 是
健康的（见下），所以内核层失败大概率是环境，不是代码。

## 这个平台上量过的能力边界

来自 S5000 上的逐层实测，**不是文档抄来的**：

| 层 | 状态 |
|---|---|
| Triton | **完全健康**，含 `associative_scan` |
| muDNN | 缺口在 **gemv**，不是 fp64 —— 这一点容易搞反 |
| `torch.sparse` | **没有任何可用的 matmul** |

最后一条直接决定了测试设计：**精度参考解恒在 host 上以 fp64 计算**。在这块卡上
"拿加速器算参考解"根本不成立，而这个选择让同一套 ctest 在每个后端都成立。

## 跑测试

```bash
export FLAGSPARSE_PYTHON_SRC=/path/to/FlagSparse/src   # 注意是 src/
export FLAGSPARSE_BACKEND=mthreads

cmake -S . -B build -G Ninja -DBACKEND=MUSA \
      -DFLAGSPARSE_PYTHON_SRC=$FLAGSPARSE_PYTHON_SRC \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 精度
ctest --test-dir build -R accuracy --output-on-failure

# 性能：真实矩阵语料。不设 FLAGSPARSE_MATRIX_DIR 会退回合成形状，
# 并把每行标成 corpus=synthetic，不会被误读成真实数据
FLAGSPARSE_MATRIX_DIR=/path/to/mtx FLAGSPARSE_BENCH_OUT=./bench \
    ctest --test-dir build -R benchmark

# 汇总：40+ 变体一张表 / 与 FlagSparse Python 侧同 schema 的 summary.json
python3 tools/report.py --bench-dir ./bench
python3 tools/write_summary.py --bench-dir ./bench --out ./bench

# 算子清单与实现是否一致（CI 可用 --strict）
python3 tools/check_manifest.py --bench-dir ./bench
```

算子清单来自 `conf/operators.yaml`，构建时由 `tools/gen_variants.py` 生成成扫描用的
变体表——**加算子是改 YAML 加重新构建，不动测试代码**。

### 这个后端上预期会看到什么

* **`speedup` 列大概率是空的**（`musparse not found under /usr/local/musa`），
  理由同 MACA：库名按摩尔线程的 CUDA 镜像惯例填，未经验证；
* **精度的参考值必须是 host fp64**，不能用 torch.sparse —— MUSA 上 `torch.sparse`
  没有可用的 matmul。ctest 本来就用 host 参考，这一条是提醒别在 Python 侧照搬。

### 跑不起来时按这个顺序查

1. **配置就停下** → adaptor 没写。MUSA 的 driver API 镜像 CUDA，所以 adaptor 是
   `backend/_template.inc` 加一个符号前缀表（参考 `backend/musa/adaptor.cpp`，25 行）；
2. **`ctest baseline: MUSA -> none`** → 正常，见上面三 token 表；
3. **某个算子 `not_supported`** → 先看是不是内核没 lower。MUSA 上 Triton 是健康的
   （associative_scan 都能用），所以更可能是 dispatch 没接；
4. **gemv 相关的性能异常** → muDNN 的缺口是 gemv 不是 fp64，别往精度方向查。

## 基线：muSPARSE，按命名惯例填好，等探测

`ctest/baseline/musa/baseline.cpp` 是 35 行前缀表加共享的
`ctest/baseline/_template.inc`。摩尔线程 Moore Threads 的工具链整体镜像 CUDA 并加 `mu` 前缀——本仓库
在 runtime 这一层已经依赖这个惯例（musart / musa.h），所以稀疏库按同族推为
`musparse` / `<musparse.h>`，其 generic（描述符）API 与 cuSPARSE 逐调用对应。

**这些名字未经验证**，但在这里是可接受的：`ctest/baseline/CMakeLists.txt` 会在
`$MUSA_HOME`（默认 `/usr/local/musa`）下用 `find_path`/`find_library` 探测，任一缺失就回落到
`baseline/none` 并打印"在 X 下没找到 musparse"。`src/adaptor/CMakeLists.txt` 里那条
**不猜库名**的规矩针对的是链接期报一个既不指明厂商也不指明错误的缺失符号——探测过后
才选用的槽位不是那种情况。

拿到 SDK 后**最先核对这三个 token**，过了就能链接：

| | 值 | 备注 |
|---|---|---|
| 头文件 | `<musparse.h>` | 可能带命名空间目录，hipSPARSE 就是 |
| 数据类型枚举 | `MU_R_32F` | 来自 **runtime** 而非稀疏库（`muDataType_t`） |
| 库名 | `libmusparse.so` | |

## 注意

* 复数走实部/虚部交错数组（Triton 没有复数类型），和 Python 侧 `view_as_real` 一致；
* 原子路线（CSC `non`、BSR 两个方向）在任何后端上 fp32 都不是逐位可复现的 ——
  那是原子累加的性质，不是缺陷；
* Python 层的测试方法见隔壁 checkout 的 `docs/MUSA_TESTING.md`，以及排障记录
  `docs/MUSA_DEBUG_HANDOFF.md`。本文只讲 C API 这一层。
