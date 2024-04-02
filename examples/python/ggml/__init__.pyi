# auto-generated file
import ggml.ffi as ffi
import numpy as np
class lib:
  @property
  def GGML_BACKEND_TYPE_CPU(self) -> int: ...
  @property
  def GGML_BACKEND_TYPE_GPU(self) -> int: ...
  @property
  def GGML_BACKEND_TYPE_GPU_SPLIT(self) -> int: ...
  @property
  def GGML_CGRAPH_EVAL_ORDER_COUNT(self) -> int: ...
  @property
  def GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT(self) -> int: ...
  @property
  def GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT(self) -> int: ...
  @property
  def GGML_FTYPE_ALL_F32(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_F16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ1_M(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ1_S(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ2_S(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ2_XS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ2_XXS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ3_S(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ3_XXS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ4_NL(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_IQ4_XS(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q2_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q3_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_0(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_1(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_1_SOME_F16(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q4_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_0(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_1(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q5_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q6_K(self) -> int: ...
  @property
  def GGML_FTYPE_MOSTLY_Q8_0(self) -> int: ...
  @property
  def GGML_FTYPE_UNKNOWN(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_ARMIJO(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE(self) -> int: ...
  @property
  def GGML_LINESEARCH_BACKTRACKING_WOLFE(self) -> int: ...
  @property
  def GGML_LINESEARCH_DEFAULT(self) -> int: ...
  @property
  def GGML_LINESEARCH_FAIL(self) -> int: ...
  @property
  def GGML_LINESEARCH_INVALID_PARAMETERS(self) -> int: ...
  @property
  def GGML_LINESEARCH_MAXIMUM_ITERATIONS(self) -> int: ...
  @property
  def GGML_LINESEARCH_MAXIMUM_STEP(self) -> int: ...
  @property
  def GGML_LINESEARCH_MINIMUM_STEP(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_DEBUG(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_ERROR(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_INFO(self) -> int: ...
  @property
  def GGML_LOG_LEVEL_WARN(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_COUNT(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_DISABLED(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_DISTRIBUTE(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_ISOLATE(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_MIRROR(self) -> int: ...
  @property
  def GGML_NUMA_STRATEGY_NUMACTL(self) -> int: ...
  @property
  def GGML_OBJECT_TYPE_GRAPH(self) -> int: ...
  @property
  def GGML_OBJECT_TYPE_TENSOR(self) -> int: ...
  @property
  def GGML_OBJECT_TYPE_WORK_BUFFER(self) -> int: ...
  @property
  def GGML_OPT_RESULT_CANCEL(self) -> int: ...
  @property
  def GGML_OPT_RESULT_DID_NOT_CONVERGE(self) -> int: ...
  @property
  def GGML_OPT_RESULT_FAIL(self) -> int: ...
  @property
  def GGML_OPT_RESULT_INVALID_WOLFE(self) -> int: ...
  @property
  def GGML_OPT_RESULT_NO_CONTEXT(self) -> int: ...
  @property
  def GGML_OPT_RESULT_OK(self) -> int: ...
  @property
  def GGML_OPT_TYPE_ADAM(self) -> int: ...
  @property
  def GGML_OPT_TYPE_LBFGS(self) -> int: ...
  @property
  def GGML_OP_ACC(self) -> int: ...
  @property
  def GGML_OP_ADD(self) -> int: ...
  @property
  def GGML_OP_ADD1(self) -> int: ...
  @property
  def GGML_OP_ADD_REL_POS(self) -> int: ...
  @property
  def GGML_OP_ALIBI(self) -> int: ...
  @property
  def GGML_OP_ARANGE(self) -> int: ...
  @property
  def GGML_OP_ARGMAX(self) -> int: ...
  @property
  def GGML_OP_ARGSORT(self) -> int: ...
  @property
  def GGML_OP_CLAMP(self) -> int: ...
  @property
  def GGML_OP_CONCAT(self) -> int: ...
  @property
  def GGML_OP_CONT(self) -> int: ...
  @property
  def GGML_OP_CONV_TRANSPOSE_1D(self) -> int: ...
  @property
  def GGML_OP_CONV_TRANSPOSE_2D(self) -> int: ...
  @property
  def GGML_OP_COUNT(self) -> int: ...
  @property
  def GGML_OP_CPY(self) -> int: ...
  @property
  def GGML_OP_CROSS_ENTROPY_LOSS(self) -> int: ...
  @property
  def GGML_OP_CROSS_ENTROPY_LOSS_BACK(self) -> int: ...
  @property
  def GGML_OP_DIAG(self) -> int: ...
  @property
  def GGML_OP_DIAG_MASK_INF(self) -> int: ...
  @property
  def GGML_OP_DIAG_MASK_ZERO(self) -> int: ...
  @property
  def GGML_OP_DIV(self) -> int: ...
  @property
  def GGML_OP_DUP(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN(self) -> int: ...
  @property
  def GGML_OP_FLASH_ATTN_BACK(self) -> int: ...
  @property
  def GGML_OP_FLASH_FF(self) -> int: ...
  @property
  def GGML_OP_GET_REL_POS(self) -> int: ...
  @property
  def GGML_OP_GET_ROWS(self) -> int: ...
  @property
  def GGML_OP_GET_ROWS_BACK(self) -> int: ...
  @property
  def GGML_OP_GROUP_NORM(self) -> int: ...
  @property
  def GGML_OP_IM2COL(self) -> int: ...
  @property
  def GGML_OP_LEAKY_RELU(self) -> int: ...
  @property
  def GGML_OP_LOG(self) -> int: ...
  @property
  def GGML_OP_MAP_BINARY(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM1(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM1_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM2(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM2_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM3(self) -> int: ...
  @property
  def GGML_OP_MAP_CUSTOM3_F32(self) -> int: ...
  @property
  def GGML_OP_MAP_UNARY(self) -> int: ...
  @property
  def GGML_OP_MEAN(self) -> int: ...
  @property
  def GGML_OP_MUL(self) -> int: ...
  @property
  def GGML_OP_MUL_MAT(self) -> int: ...
  @property
  def GGML_OP_MUL_MAT_ID(self) -> int: ...
  @property
  def GGML_OP_NONE(self) -> int: ...
  @property
  def GGML_OP_NORM(self) -> int: ...
  @property
  def GGML_OP_OUT_PROD(self) -> int: ...
  @property
  def GGML_OP_PAD(self) -> int: ...
  @property
  def GGML_OP_PERMUTE(self) -> int: ...
  @property
  def GGML_OP_POOL_1D(self) -> int: ...
  @property
  def GGML_OP_POOL_2D(self) -> int: ...
  @property
  def GGML_OP_POOL_AVG(self) -> int: ...
  @property
  def GGML_OP_POOL_COUNT(self) -> int: ...
  @property
  def GGML_OP_POOL_MAX(self) -> int: ...
  @property
  def GGML_OP_REPEAT(self) -> int: ...
  @property
  def GGML_OP_REPEAT_BACK(self) -> int: ...
  @property
  def GGML_OP_RESHAPE(self) -> int: ...
  @property
  def GGML_OP_RMS_NORM(self) -> int: ...
  @property
  def GGML_OP_RMS_NORM_BACK(self) -> int: ...
  @property
  def GGML_OP_ROPE(self) -> int: ...
  @property
  def GGML_OP_ROPE_BACK(self) -> int: ...
  @property
  def GGML_OP_SCALE(self) -> int: ...
  @property
  def GGML_OP_SET(self) -> int: ...
  @property
  def GGML_OP_SILU_BACK(self) -> int: ...
  @property
  def GGML_OP_SOFT_MAX(self) -> int: ...
  @property
  def GGML_OP_SOFT_MAX_BACK(self) -> int: ...
  @property
  def GGML_OP_SQR(self) -> int: ...
  @property
  def GGML_OP_SQRT(self) -> int: ...
  @property
  def GGML_OP_SSM_CONV(self) -> int: ...
  @property
  def GGML_OP_SSM_SCAN(self) -> int: ...
  @property
  def GGML_OP_SUB(self) -> int: ...
  @property
  def GGML_OP_SUM(self) -> int: ...
  @property
  def GGML_OP_SUM_ROWS(self) -> int: ...
  @property
  def GGML_OP_TIMESTEP_EMBEDDING(self) -> int: ...
  @property
  def GGML_OP_TRANSPOSE(self) -> int: ...
  @property
  def GGML_OP_UNARY(self) -> int: ...
  @property
  def GGML_OP_UPSCALE(self) -> int: ...
  @property
  def GGML_OP_VIEW(self) -> int: ...
  @property
  def GGML_OP_WIN_PART(self) -> int: ...
  @property
  def GGML_OP_WIN_UNPART(self) -> int: ...
  @property
  def GGML_PREC_DEFAULT(self) -> int: ...
  @property
  def GGML_PREC_F32(self) -> int: ...
  @property
  def GGML_SORT_ORDER_ASC(self) -> int: ...
  @property
  def GGML_SORT_ORDER_DESC(self) -> int: ...
  @property
  def GGML_STATUS_ABORTED(self) -> int: ...
  @property
  def GGML_STATUS_ALLOC_FAILED(self) -> int: ...
  @property
  def GGML_STATUS_FAILED(self) -> int: ...
  @property
  def GGML_STATUS_SUCCESS(self) -> int: ...
  @property
  def GGML_TASK_TYPE_COMPUTE(self) -> int: ...
  @property
  def GGML_TASK_TYPE_FINALIZE(self) -> int: ...
  @property
  def GGML_TASK_TYPE_INIT(self) -> int: ...
  @property
  def GGML_TENSOR_FLAG_INPUT(self) -> int: ...
  @property
  def GGML_TENSOR_FLAG_OUTPUT(self) -> int: ...
  @property
  def GGML_TENSOR_FLAG_PARAM(self) -> int: ...
  @property
  def GGML_TYPE_COUNT(self) -> int: ...
  @property
  def GGML_TYPE_F16(self) -> int: ...
  @property
  def GGML_TYPE_F32(self) -> int: ...
  @property
  def GGML_TYPE_F64(self) -> int: ...
  @property
  def GGML_TYPE_I16(self) -> int: ...
  @property
  def GGML_TYPE_I32(self) -> int: ...
  @property
  def GGML_TYPE_I64(self) -> int: ...
  @property
  def GGML_TYPE_I8(self) -> int: ...
  @property
  def GGML_TYPE_IQ1_M(self) -> int: ...
  @property
  def GGML_TYPE_IQ1_S(self) -> int: ...
  @property
  def GGML_TYPE_IQ2_S(self) -> int: ...
  @property
  def GGML_TYPE_IQ2_XS(self) -> int: ...
  @property
  def GGML_TYPE_IQ2_XXS(self) -> int: ...
  @property
  def GGML_TYPE_IQ3_S(self) -> int: ...
  @property
  def GGML_TYPE_IQ3_XXS(self) -> int: ...
  @property
  def GGML_TYPE_IQ4_NL(self) -> int: ...
  @property
  def GGML_TYPE_IQ4_XS(self) -> int: ...
  @property
  def GGML_TYPE_Q2_K(self) -> int: ...
  @property
  def GGML_TYPE_Q3_K(self) -> int: ...
  @property
  def GGML_TYPE_Q4_0(self) -> int: ...
  @property
  def GGML_TYPE_Q4_1(self) -> int: ...
  @property
  def GGML_TYPE_Q4_K(self) -> int: ...
  @property
  def GGML_TYPE_Q5_0(self) -> int: ...
  @property
  def GGML_TYPE_Q5_1(self) -> int: ...
  @property
  def GGML_TYPE_Q5_K(self) -> int: ...
  @property
  def GGML_TYPE_Q6_K(self) -> int: ...
  @property
  def GGML_TYPE_Q8_0(self) -> int: ...
  @property
  def GGML_TYPE_Q8_1(self) -> int: ...
  @property
  def GGML_TYPE_Q8_K(self) -> int: ...
  @property
  def GGML_UNARY_OP_ABS(self) -> int: ...
  @property
  def GGML_UNARY_OP_COUNT(self) -> int: ...
  @property
  def GGML_UNARY_OP_ELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_GELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_GELU_QUICK(self) -> int: ...
  @property
  def GGML_UNARY_OP_HARDSIGMOID(self) -> int: ...
  @property
  def GGML_UNARY_OP_HARDSWISH(self) -> int: ...
  @property
  def GGML_UNARY_OP_NEG(self) -> int: ...
  @property
  def GGML_UNARY_OP_RELU(self) -> int: ...
  @property
  def GGML_UNARY_OP_SGN(self) -> int: ...
  @property
  def GGML_UNARY_OP_SILU(self) -> int: ...
  @property
  def GGML_UNARY_OP_STEP(self) -> int: ...
  @property
  def GGML_UNARY_OP_TANH(self) -> int: ...
  @property
  def GGUF_TYPE_ARRAY(self) -> int: ...
  @property
  def GGUF_TYPE_BOOL(self) -> int: ...
  @property
  def GGUF_TYPE_COUNT(self) -> int: ...
  @property
  def GGUF_TYPE_FLOAT32(self) -> int: ...
  @property
  def GGUF_TYPE_FLOAT64(self) -> int: ...
  @property
  def GGUF_TYPE_INT16(self) -> int: ...
  @property
  def GGUF_TYPE_INT32(self) -> int: ...
  @property
  def GGUF_TYPE_INT64(self) -> int: ...
  @property
  def GGUF_TYPE_INT8(self) -> int: ...
  @property
  def GGUF_TYPE_STRING(self) -> int: ...
  @property
  def GGUF_TYPE_UINT16(self) -> int: ...
  @property
  def GGUF_TYPE_UINT32(self) -> int: ...
  @property
  def GGUF_TYPE_UINT64(self) -> int: ...
  @property
  def GGUF_TYPE_UINT8(self) -> int: ...
  def dequantize_row_iq1_m(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq1_m (const block_iq1_m * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq1_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq1_s (const block_iq1_s * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq2_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq2_s (const block_iq2_s * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq2_xs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq2_xs (const block_iq2_xs * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq2_xxs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq2_xxs(const block_iq2_xxs * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq3_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq3_s (const block_iq3_s * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq3_xxs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq3_xxs(const block_iq3_xxs * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq4_nl(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq4_nl (const block_iq4_nl * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_iq4_xs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_iq4_xs (const block_iq4_xs * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q2_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q3_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q4_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q4_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q4_1(const block_q4_1 * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q4_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q5_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_0(const block_q5_0 * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q5_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_1(const block_q5_1 * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q5_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q6_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q8_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int k);"""
    ...
  def dequantize_row_q8_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int k);"""
    ...
  def ggml_abs(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_abs(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_abs_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_abs_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_acc(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_acc(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t nb1,
                size_t nb2,
                size_t nb3,
                size_t offset);
    """
    ...
  def ggml_acc_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_acc_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t nb1,
                size_t nb2,
                size_t nb3,
                size_t offset);
    """
    ...
  def ggml_add(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_add1(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add1(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_add1_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add1_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_add_cast(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, type: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add_cast(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                enum ggml_type type);
    """
    ...
  def ggml_add_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_add_rel_pos(ctx: ffi.CData, a: ffi.CData, pw: ffi.CData, ph: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add_rel_pos(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * pw,
                struct ggml_tensor * ph);
    """
    ...
  def ggml_add_rel_pos_inplace(ctx: ffi.CData, a: ffi.CData, pw: ffi.CData, ph: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_add_rel_pos_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * pw,
                struct ggml_tensor * ph);
    """
    ...
  def ggml_alibi(ctx: ffi.CData, a: ffi.CData, n_past: int, n_head: int, bias_max: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_alibi( struct ggml_context * ctx, struct ggml_tensor * a, int n_past, int n_head, float bias_max)
                                                                          ;
    """
    ...
  def ggml_arange(ctx: ffi.CData, start: float, stop: float, step: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_arange(
                struct ggml_context * ctx,
                float start,
                float stop,
                float step);
    """
    ...
  def ggml_are_same_shape(t0: ffi.CData, t1: ffi.CData) -> bool:
    """                 ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);"""
    ...
  def ggml_argmax(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_argmax(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_argsort(ctx: ffi.CData, a: ffi.CData, order: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_argsort(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                enum ggml_sort_order order);
    """
    ...
  def ggml_blck_size(type: int) -> int:
    """    int ggml_blck_size(enum ggml_type type);"""
    ...
  def ggml_build_backward_expand(ctx: ffi.CData, gf: ffi.CData, gb: ffi.CData, keep: bool) -> None:
    """
        void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb,
                                                                                                                             _Bool
                                                                                                                                  keep);
    """
    ...
  def ggml_build_backward_gradient_checkpointing(ctx: ffi.CData, gf: ffi.CData, gb: ffi.CData, gb_tmp: ffi.CData, checkpoints: ffi.CData, n_checkpoints: int) -> None:
    """
        void ggml_build_backward_gradient_checkpointing(
                struct ggml_context * ctx,
                struct ggml_cgraph * gf,
                struct ggml_cgraph * gb,
                struct ggml_cgraph * gb_tmp,
                struct ggml_tensor * * checkpoints,
                int n_checkpoints);
    """
    ...
  def ggml_build_forward_expand(cgraph: ffi.CData, tensor: ffi.CData) -> None:
    """    void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);"""
    ...
  def ggml_cast(ctx: ffi.CData, a: ffi.CData, type: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cast(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                enum ggml_type type);
    """
    ...
  def ggml_clamp(ctx: ffi.CData, a: ffi.CData, min: float, max: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_clamp(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float min,
                float max);
    """
    ...
  def ggml_concat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_concat(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_cont(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cont(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_cont_1d(ctx: ffi.CData, a: ffi.CData, ne0: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cont_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0);
    """
    ...
  def ggml_cont_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cont_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1);
    """
    ...
  def ggml_cont_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cont_3d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2);
    """
    ...
  def ggml_cont_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cont_4d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                int64_t ne3);
    """
    ...
  def ggml_conv_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, p0: int, d0: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int s0,
                int p0,
                int d0);
    """
    ...
  def ggml_conv_1d_ph(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s: int, d: int) -> ffi.CData:
    """
        struct ggml_tensor* ggml_conv_1d_ph(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int s,
                int d);
    """
    ...
  def ggml_conv_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int s0,
                int s1,
                int p0,
                int p1,
                int d0,
                int d1);
    """
    ...
  def ggml_conv_2d_s1_ph(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_2d_s1_ph(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_conv_2d_sk_p0(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_2d_sk_p0(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_conv_depthwise_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_depthwise_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int s0,
                int s1,
                int p0,
                int p1,
                int d0,
                int d1);
    """
    ...
  def ggml_conv_transpose_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, p0: int, d0: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_transpose_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int s0,
                int p0,
                int d0);
    """
    ...
  def ggml_conv_transpose_2d_p0(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, stride: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_conv_transpose_2d_p0(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int stride);
    """
    ...
  def ggml_cpu_has_arm_fma() -> int:
    """    int ggml_cpu_has_arm_fma (void);"""
    ...
  def ggml_cpu_has_avx() -> int:
    """    int ggml_cpu_has_avx (void);"""
    ...
  def ggml_cpu_has_avx2() -> int:
    """    int ggml_cpu_has_avx2 (void);"""
    ...
  def ggml_cpu_has_avx512() -> int:
    """    int ggml_cpu_has_avx512 (void);"""
    ...
  def ggml_cpu_has_avx512_vbmi() -> int:
    """    int ggml_cpu_has_avx512_vbmi(void);"""
    ...
  def ggml_cpu_has_avx512_vnni() -> int:
    """    int ggml_cpu_has_avx512_vnni(void);"""
    ...
  def ggml_cpu_has_avx_vnni() -> int:
    """    int ggml_cpu_has_avx_vnni (void);"""
    ...
  def ggml_cpu_has_blas() -> int:
    """    int ggml_cpu_has_blas (void);"""
    ...
  def ggml_cpu_has_clblast() -> int:
    """    int ggml_cpu_has_clblast (void);"""
    ...
  def ggml_cpu_has_cuda() -> int:
    """    int ggml_cpu_has_cuda (void);"""
    ...
  def ggml_cpu_has_f16c() -> int:
    """    int ggml_cpu_has_f16c (void);"""
    ...
  def ggml_cpu_has_fma() -> int:
    """    int ggml_cpu_has_fma (void);"""
    ...
  def ggml_cpu_has_fp16_va() -> int:
    """    int ggml_cpu_has_fp16_va (void);"""
    ...
  def ggml_cpu_has_gpublas() -> int:
    """    int ggml_cpu_has_gpublas (void);"""
    ...
  def ggml_cpu_has_kompute() -> int:
    """    int ggml_cpu_has_kompute (void);"""
    ...
  def ggml_cpu_has_matmul_int8() -> int:
    """    int ggml_cpu_has_matmul_int8(void);"""
    ...
  def ggml_cpu_has_metal() -> int:
    """    int ggml_cpu_has_metal (void);"""
    ...
  def ggml_cpu_has_neon() -> int:
    """    int ggml_cpu_has_neon (void);"""
    ...
  def ggml_cpu_has_sse3() -> int:
    """    int ggml_cpu_has_sse3 (void);"""
    ...
  def ggml_cpu_has_ssse3() -> int:
    """    int ggml_cpu_has_ssse3 (void);"""
    ...
  def ggml_cpu_has_sycl() -> int:
    """    int ggml_cpu_has_sycl (void);"""
    ...
  def ggml_cpu_has_vsx() -> int:
    """    int ggml_cpu_has_vsx (void);"""
    ...
  def ggml_cpu_has_vulkan() -> int:
    """    int ggml_cpu_has_vulkan (void);"""
    ...
  def ggml_cpu_has_wasm_simd() -> int:
    """    int ggml_cpu_has_wasm_simd (void);"""
    ...
  def ggml_cpy(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cpy(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_cross_entropy_loss(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cross_entropy_loss(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_cross_entropy_loss_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_cross_entropy_loss_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                struct ggml_tensor * c);
    """
    ...
  def ggml_cycles() -> int:
    """    int64_t ggml_cycles(void);"""
    ...
  def ggml_cycles_per_ms() -> int:
    """    int64_t ggml_cycles_per_ms(void);"""
    ...
  def ggml_diag(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_diag(
            struct ggml_context * ctx,
            struct ggml_tensor * a);
    """
    ...
  def ggml_diag_mask_inf(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_diag_mask_inf(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int n_past);
    """
    ...
  def ggml_diag_mask_inf_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_diag_mask_inf_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int n_past);
    """
    ...
  def ggml_diag_mask_zero(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_diag_mask_zero(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int n_past);
    """
    ...
  def ggml_diag_mask_zero_inplace(ctx: ffi.CData, a: ffi.CData, n_past: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_diag_mask_zero_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int n_past);
    """
    ...
  def ggml_div(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_div(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_div_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_div_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_dup(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_dup(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_dup_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_dup_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_dup_tensor(ctx: ffi.CData, src: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);"""
    ...
  def ggml_element_size(tensor: ffi.CData) -> int:
    """    size_t ggml_element_size(const struct ggml_tensor * tensor);"""
    ...
  def ggml_elu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_elu(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_elu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_elu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_flash_attn(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, masked: bool) -> ffi.CData:
    """
        struct ggml_tensor * ggml_flash_attn(
                struct ggml_context * ctx,
                struct ggml_tensor * q,
                struct ggml_tensor * k,
                struct ggml_tensor * v,
               _Bool
                                     masked);
    """
    ...
  def ggml_flash_attn_back(ctx: ffi.CData, q: ffi.CData, k: ffi.CData, v: ffi.CData, d: ffi.CData, masked: bool) -> ffi.CData:
    """
        struct ggml_tensor * ggml_flash_attn_back(
               struct ggml_context * ctx,
               struct ggml_tensor * q,
               struct ggml_tensor * k,
               struct ggml_tensor * v,
               struct ggml_tensor * d,
              _Bool
                                    masked);
    """
    ...
  def ggml_flash_ff(ctx: ffi.CData, a: ffi.CData, b0: ffi.CData, b1: ffi.CData, c0: ffi.CData, c1: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_flash_ff(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b0,
                struct ggml_tensor * b1,
                struct ggml_tensor * c0,
                struct ggml_tensor * c1);
    """
    ...
  def ggml_fopen(fname: ffi.CData, mode: ffi.CData) -> ffi.CData:
    """    FILE * ggml_fopen(const char * fname, const char * mode);"""
    ...
  def ggml_format_name(tensor: ffi.CData, fmt: ffi.CData, *args2) -> ffi.CData:
    """    struct ggml_tensor * ggml_format_name( struct ggml_tensor * tensor, const char * fmt, ...);"""
    ...
  def ggml_fp16_to_fp32(x: np.float16) -> float:
    """    float ggml_fp16_to_fp32(ggml_fp16_t x);"""
    ...
  def ggml_fp16_to_fp32_row(x: ffi.CData, y: ffi.CData, n: int) -> None:
    """    void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int n);"""
    ...
  def ggml_fp32_to_fp16(x: float) -> np.float16:
    """    ggml_fp16_t ggml_fp32_to_fp16(float x);"""
    ...
  def ggml_fp32_to_fp16_row(x: ffi.CData, y: ffi.CData, n: int) -> None:
    """    void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int n);"""
    ...
  def ggml_free(ctx: ffi.CData) -> None:
    """    void ggml_free(struct ggml_context * ctx);"""
    ...
  def ggml_ftype_to_ggml_type(ftype: int) -> int:
    """    enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);"""
    ...
  def ggml_gelu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_gelu(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_gelu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_gelu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_gelu_quick(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_gelu_quick(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_gelu_quick_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_gelu_quick_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_get_data(tensor: ffi.CData) -> ffi.CData:
    """    void * ggml_get_data (const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_data_f32(tensor: ffi.CData) -> ffi.CData:
    """    float * ggml_get_data_f32(const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_f32_1d(tensor: ffi.CData, i: int) -> float:
    """    float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);"""
    ...
  def ggml_get_f32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int) -> float:
    """    float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);"""
    ...
  def ggml_get_first_tensor(ctx: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);"""
    ...
  def ggml_get_i32_1d(tensor: ffi.CData, i: int) -> int:
    """    int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);"""
    ...
  def ggml_get_i32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int) -> int:
    """    int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);"""
    ...
  def ggml_get_max_tensor_size(ctx: ffi.CData) -> int:
    """    size_t ggml_get_max_tensor_size(const struct ggml_context * ctx);"""
    ...
  def ggml_get_mem_buffer(ctx: ffi.CData) -> ffi.CData:
    """    void * ggml_get_mem_buffer (const struct ggml_context * ctx);"""
    ...
  def ggml_get_mem_size(ctx: ffi.CData) -> int:
    """    size_t ggml_get_mem_size (const struct ggml_context * ctx);"""
    ...
  def ggml_get_name(tensor: ffi.CData) -> ffi.CData:
    """    const char * ggml_get_name (const struct ggml_tensor * tensor);"""
    ...
  def ggml_get_next_tensor(ctx: ffi.CData, tensor: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);"""
    ...
  def ggml_get_no_alloc(ctx: ffi.CData) -> bool:
    """                    ggml_get_no_alloc(struct ggml_context * ctx);"""
    ...
  def ggml_get_rel_pos(ctx: ffi.CData, a: ffi.CData, qh: int, kh: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_get_rel_pos(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int qh,
                int kh);
    """
    ...
  def ggml_get_rows(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_get_rows(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_get_rows_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_get_rows_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                struct ggml_tensor * c);
    """
    ...
  def ggml_get_tensor(ctx: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);"""
    ...
  def ggml_get_unary_op(tensor: ffi.CData) -> int:
    """    enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);"""
    ...
  def ggml_graph_clear(cgraph: ffi.CData) -> None:
    """    void ggml_graph_clear (struct ggml_cgraph * cgraph);"""
    ...
  def ggml_graph_compute(cgraph: ffi.CData, cplan: ffi.CData) -> int:
    """    enum ggml_status ggml_graph_compute ( struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);"""
    ...
  def ggml_graph_compute_with_ctx(ctx: ffi.CData, cgraph: ffi.CData, n_threads: int) -> int:
    """    enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);"""
    ...
  def ggml_graph_cpy(src: ffi.CData, dst: ffi.CData) -> None:
    """    void ggml_graph_cpy (struct ggml_cgraph * src, struct ggml_cgraph * dst);"""
    ...
  def ggml_graph_dump_dot(gb: ffi.CData, gf: ffi.CData, filename: ffi.CData) -> None:
    """    void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);"""
    ...
  def ggml_graph_dup(ctx: ffi.CData, cgraph: ffi.CData) -> ffi.CData:
    """    struct ggml_cgraph * ggml_graph_dup (struct ggml_context * ctx, struct ggml_cgraph * cgraph);"""
    ...
  def ggml_graph_export(cgraph: ffi.CData, fname: ffi.CData) -> None:
    """    void ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);"""
    ...
  def ggml_graph_get_tensor(cgraph: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);"""
    ...
  def ggml_graph_import(fname: ffi.CData, ctx_data: ffi.CData, ctx_eval: ffi.CData) -> ffi.CData:
    """    struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);"""
    ...
  def ggml_graph_overhead() -> int:
    """    size_t ggml_graph_overhead(void);"""
    ...
  def ggml_graph_overhead_custom(size: int, grads: bool) -> int:
    """
        size_t ggml_graph_overhead_custom(size_t size,
                                                               _Bool
                                                                    grads);
    """
    ...
  def ggml_graph_plan(cgraph: ffi.CData, n_threads: int) -> ffi.CData:
    """    struct ggml_cplan ggml_graph_plan (const struct ggml_cgraph * cgraph, int n_threads );"""
    ...
  def ggml_graph_print(cgraph: ffi.CData) -> None:
    """    void ggml_graph_print(const struct ggml_cgraph * cgraph);"""
    ...
  def ggml_graph_reset(cgraph: ffi.CData) -> None:
    """    void ggml_graph_reset (struct ggml_cgraph * cgraph);"""
    ...
  def ggml_graph_view(cgraph: ffi.CData, i0: int, i1: int) -> ffi.CData:
    """    struct ggml_cgraph ggml_graph_view (struct ggml_cgraph * cgraph, int i0, int i1);"""
    ...
  def ggml_group_norm(ctx: ffi.CData, a: ffi.CData, n_groups: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_group_norm(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int n_groups);
    """
    ...
  def ggml_group_norm_inplace(ctx: ffi.CData, a: ffi.CData, n_groups: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_group_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int n_groups);
    """
    ...
  def ggml_guid_matches(guid_a: ffi.CData, guid_b: ffi.CData) -> bool:
    """                 ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);"""
    ...
  def ggml_hardsigmoid(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_hardsigmoid(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_hardswish(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_hardswish(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_im2col(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int, is_2D: bool, dst_type: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_im2col(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int s0,
                int s1,
                int p0,
                int p1,
                int d0,
                int d1,
               _Bool
                                    is_2D,
                enum ggml_type dst_type);
    """
    ...
  def ggml_init(params: ffi.CData) -> ffi.CData:
    """    struct ggml_context * ggml_init(struct ggml_init_params params);"""
    ...
  def ggml_internal_get_type_traits(type: int) -> ffi.CData:
    """    ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);"""
    ...
  def ggml_is_3d(tensor: ffi.CData) -> bool:
    """                           ggml_is_3d (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_contiguous(tensor: ffi.CData) -> bool:
    """                           ggml_is_contiguous(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_empty(tensor: ffi.CData) -> bool:
    """                           ggml_is_empty (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_matrix(tensor: ffi.CData) -> bool:
    """                           ggml_is_matrix (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_numa() -> bool:
    """                    ggml_is_numa(void);"""
    ...
  def ggml_is_permuted(tensor: ffi.CData) -> bool:
    """                           ggml_is_permuted (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_quantized(type: int) -> bool:
    """                              ggml_is_quantized(enum ggml_type type);"""
    ...
  def ggml_is_scalar(tensor: ffi.CData) -> bool:
    """                           ggml_is_scalar (const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_transposed(tensor: ffi.CData) -> bool:
    """                           ggml_is_transposed(const struct ggml_tensor * tensor);"""
    ...
  def ggml_is_vector(tensor: ffi.CData) -> bool:
    """                           ggml_is_vector (const struct ggml_tensor * tensor);"""
    ...
  def ggml_leaky_relu(ctx: ffi.CData, a: ffi.CData, negative_slope: float, inplace: bool) -> ffi.CData:
    """
        struct ggml_tensor * ggml_leaky_relu(
                struct ggml_context * ctx,
                struct ggml_tensor * a, float negative_slope,
                                                              _Bool
                                                                   inplace);
    """
    ...
  def ggml_log(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_log(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_log_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_log_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_map_binary_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_binary_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun)
                                           ;
    """
    ...
  def ggml_map_binary_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_binary_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun)
                                                   ;
    """
    ...
  def ggml_map_custom1(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom1(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                ggml_custom1_op_t fun,
                int n_tasks,
                void * userdata);
    """
    ...
  def ggml_map_custom1_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom1_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun)
                                           ;
    """
    ...
  def ggml_map_custom1_inplace(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom1_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                ggml_custom1_op_t fun,
                int n_tasks,
                void * userdata);
    """
    ...
  def ggml_map_custom1_inplace_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom1_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun)
                                                   ;
    """
    ...
  def ggml_map_custom2(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom2(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                ggml_custom2_op_t fun,
                int n_tasks,
                void * userdata);
    """
    ...
  def ggml_map_custom2_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom2_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun)
                                           ;
    """
    ...
  def ggml_map_custom2_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom2_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                ggml_custom2_op_t fun,
                int n_tasks,
                void * userdata);
    """
    ...
  def ggml_map_custom2_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom2_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun)
                                                   ;
    """
    ...
  def ggml_map_custom3(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom3(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                struct ggml_tensor * c,
                ggml_custom3_op_t fun,
                int n_tasks,
                void * userdata);
    """
    ...
  def ggml_map_custom3_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom3_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun)
                                           ;
    """
    ...
  def ggml_map_custom3_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData, n_tasks: int, userdata: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom3_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                struct ggml_tensor * c,
                ggml_custom3_op_t fun,
                int n_tasks,
                void * userdata);
    """
    ...
  def ggml_map_custom3_inplace_f32(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, c: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_custom3_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun)
                                                   ;
    """
    ...
  def ggml_map_unary_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_unary_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun)
                                           ;
    """
    ...
  def ggml_map_unary_inplace_f32(ctx: ffi.CData, a: ffi.CData, fun: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_map_unary_inplace_f32( struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun)
                                                   ;
    """
    ...
  def ggml_mean(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_mean(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_mul(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_mul(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_mul_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_mul_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_mul_mat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_mul_mat(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_mul_mat_id(ctx: ffi.CData, as: ffi.CData, n_as: int, ids: ffi.CData, id: int, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_mul_mat_id(
                struct ggml_context * ctx,
                struct ggml_tensor * const as[],
                int n_as,
                struct ggml_tensor * ids,
                int id,
                struct ggml_tensor * b);
    """
    ...
  def ggml_mul_mat_set_prec(a: ffi.CData, prec: int) -> None:
    """
        void ggml_mul_mat_set_prec(
                struct ggml_tensor * a,
                enum ggml_prec prec);
    """
    ...
  def ggml_n_dims(tensor: ffi.CData) -> int:
    """    int ggml_n_dims (const struct ggml_tensor * tensor);"""
    ...
  def ggml_nbytes(tensor: ffi.CData) -> int:
    """    size_t ggml_nbytes (const struct ggml_tensor * tensor);"""
    ...
  def ggml_nbytes_pad(tensor: ffi.CData) -> int:
    """    size_t ggml_nbytes_pad (const struct ggml_tensor * tensor);"""
    ...
  def ggml_neg(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_neg(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_neg_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_neg_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_nelements(tensor: ffi.CData) -> int:
    """    int64_t ggml_nelements (const struct ggml_tensor * tensor);"""
    ...
  def ggml_new_f32(ctx: ffi.CData, value: float) -> ffi.CData:
    """    struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);"""
    ...
  def ggml_new_graph(ctx: ffi.CData) -> ffi.CData:
    """    struct ggml_cgraph * ggml_new_graph (struct ggml_context * ctx);"""
    ...
  def ggml_new_graph_custom(ctx: ffi.CData, size: int, grads: bool) -> ffi.CData:
    """
        struct ggml_cgraph * ggml_new_graph_custom (struct ggml_context * ctx, size_t size,
                                                                                                     _Bool
                                                                                                          grads);
    """
    ...
  def ggml_new_i32(ctx: ffi.CData, value: int) -> ffi.CData:
    """    struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);"""
    ...
  def ggml_new_tensor(ctx: ffi.CData, type: int, n_dims: int, ne: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_new_tensor(
                struct ggml_context * ctx,
                enum ggml_type type,
                int n_dims,
                const int64_t *ne);
    """
    ...
  def ggml_new_tensor_1d(ctx: ffi.CData, type: int, ne0: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_new_tensor_1d(
                struct ggml_context * ctx,
                enum ggml_type type,
                int64_t ne0);
    """
    ...
  def ggml_new_tensor_2d(ctx: ffi.CData, type: int, ne0: int, ne1: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_new_tensor_2d(
                struct ggml_context * ctx,
                enum ggml_type type,
                int64_t ne0,
                int64_t ne1);
    """
    ...
  def ggml_new_tensor_3d(ctx: ffi.CData, type: int, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_new_tensor_3d(
                struct ggml_context * ctx,
                enum ggml_type type,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2);
    """
    ...
  def ggml_new_tensor_4d(ctx: ffi.CData, type: int, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_new_tensor_4d(
                struct ggml_context * ctx,
                enum ggml_type type,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                int64_t ne3);
    """
    ...
  def ggml_norm(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_norm(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float eps);
    """
    ...
  def ggml_norm_inplace(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float eps);
    """
    ...
  def ggml_nrows(tensor: ffi.CData) -> int:
    """    int64_t ggml_nrows (const struct ggml_tensor * tensor);"""
    ...
  def ggml_numa_init(numa: int) -> None:
    """    void ggml_numa_init(enum ggml_numa_strategy numa);"""
    ...
  def ggml_op_desc(t: ffi.CData) -> ffi.CData:
    """    const char * ggml_op_desc(const struct ggml_tensor * t);"""
    ...
  def ggml_op_name(op: int) -> ffi.CData:
    """    const char * ggml_op_name (enum ggml_op op);"""
    ...
  def ggml_op_symbol(op: int) -> ffi.CData:
    """    const char * ggml_op_symbol(enum ggml_op op);"""
    ...
  def ggml_opt(ctx: ffi.CData, params: ffi.CData, f: ffi.CData) -> int:
    """
        enum ggml_opt_result ggml_opt(
                struct ggml_context * ctx,
                struct ggml_opt_params params,
                struct ggml_tensor * f);
    """
    ...
  def ggml_opt_default_params(type: int) -> ffi.CData:
    """    struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);"""
    ...
  def ggml_opt_init(ctx: ffi.CData, opt: ffi.CData, params: ffi.CData, nx: int) -> None:
    """
        void ggml_opt_init(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_opt_params params,
                int64_t nx);
    """
    ...
  def ggml_opt_resume(ctx: ffi.CData, opt: ffi.CData, f: ffi.CData) -> int:
    """
        enum ggml_opt_result ggml_opt_resume(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_tensor * f);
    """
    ...
  def ggml_opt_resume_g(ctx: ffi.CData, opt: ffi.CData, f: ffi.CData, gf: ffi.CData, gb: ffi.CData, callback: ffi.CData, callback_data: ffi.CData) -> int:
    """
        enum ggml_opt_result ggml_opt_resume_g(
                struct ggml_context * ctx,
                struct ggml_opt_context * opt,
                struct ggml_tensor * f,
                struct ggml_cgraph * gf,
                struct ggml_cgraph * gb,
                ggml_opt_callback callback,
                void * callback_data);
    """
    ...
  def ggml_out_prod(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_out_prod(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_pad(ctx: ffi.CData, a: ffi.CData, p0: int, p1: int, p2: int, p3: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_pad(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int p0,
                int p1,
                int p2,
                int p3);
    """
    ...
  def ggml_permute(ctx: ffi.CData, a: ffi.CData, axis0: int, axis1: int, axis2: int, axis3: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_permute(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int axis0,
                int axis1,
                int axis2,
                int axis3);
    """
    ...
  def ggml_pool_1d(ctx: ffi.CData, a: ffi.CData, op: int, k0: int, s0: int, p0: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_pool_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                enum ggml_op_pool op,
                int k0,
                int s0,
                int p0);
    """
    ...
  def ggml_pool_2d(ctx: ffi.CData, a: ffi.CData, op: int, k0: int, k1: int, s0: int, s1: int, p0: float, p1: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_pool_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                enum ggml_op_pool op,
                int k0,
                int k1,
                int s0,
                int s1,
                float p0,
                float p1);
    """
    ...
  def ggml_print_backtrace() -> None:
    """    void ggml_print_backtrace(void);"""
    ...
  def ggml_print_object(obj: ffi.CData) -> None:
    """    void ggml_print_object (const struct ggml_object * obj);"""
    ...
  def ggml_print_objects(ctx: ffi.CData) -> None:
    """    void ggml_print_objects(const struct ggml_context * ctx);"""
    ...
  def ggml_quantize_chunk(type: int, src: ffi.CData, dst: ffi.CData, start: int, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """
        size_t ggml_quantize_chunk(
                enum ggml_type type,
                   const float * src,
                          void * dst,
                           int start,
                           int nrows,
                           int n_per_row,
                   const float * imatrix);
    """
    ...
  def ggml_quantize_free() -> None:
    """    void ggml_quantize_free(void);"""
    ...
  def ggml_quantize_init(type: int) -> None:
    """    void ggml_quantize_init(enum ggml_type type);"""
    ...
  def ggml_quantize_requires_imatrix(type: int) -> bool:
    """                 ggml_quantize_requires_imatrix(enum ggml_type type);"""
    ...
  def ggml_relu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_relu(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_relu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_relu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_repeat(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_repeat(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_repeat_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_repeat_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_reshape(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_reshape(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_reshape_1d(ctx: ffi.CData, a: ffi.CData, ne0: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_reshape_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0);
    """
    ...
  def ggml_reshape_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_reshape_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1);
    """
    ...
  def ggml_reshape_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_reshape_3d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2);
    """
    ...
  def ggml_reshape_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_reshape_4d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                int64_t ne3);
    """
    ...
  def ggml_rms_norm(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rms_norm(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float eps);
    """
    ...
  def ggml_rms_norm_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, eps: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rms_norm_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                float eps);
    """
    ...
  def ggml_rms_norm_inplace(ctx: ffi.CData, a: ffi.CData, eps: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rms_norm_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float eps);
    """
    ...
  def ggml_rope(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rope(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int n_dims,
                int mode,
                int n_ctx);
    """
    ...
  def ggml_rope_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float, xpos_base: float, xpos_down: bool) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rope_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int n_dims,
                int mode,
                int n_ctx,
                int n_orig_ctx,
                float freq_base,
                float freq_scale,
                float ext_factor,
                float attn_factor,
                float beta_fast,
                float beta_slow,
                float xpos_base,
               _Bool
                                     xpos_down);
    """
    ...
  def ggml_rope_custom(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rope_custom(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int n_dims,
                int mode,
                int n_ctx,
                int n_orig_ctx,
                float freq_base,
                float freq_scale,
                float ext_factor,
                float attn_factor,
                float beta_fast,
                float beta_slow);
    """
    ...
  def ggml_rope_custom_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int, n_orig_ctx: int, freq_base: float, freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rope_custom_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int n_dims,
                int mode,
                int n_ctx,
                int n_orig_ctx,
                float freq_base,
                float freq_scale,
                float ext_factor,
                float attn_factor,
                float beta_fast,
                float beta_slow);
    """
    ...
  def ggml_rope_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, mode: int, n_ctx: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rope_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int n_dims,
                int mode,
                int n_ctx);
    """
    ...
  def ggml_rope_xpos_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, n_dims: int, base: float, down: bool) -> ffi.CData:
    """
        struct ggml_tensor * ggml_rope_xpos_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                int n_dims,
                float base,
               _Bool
                                     down);
    """
    ...
  def ggml_rope_yarn_corr_dims(n_dims: int, n_orig_ctx: int, freq_base: float, beta_fast: float, beta_slow: float, dims: float) -> None:
    """
        void ggml_rope_yarn_corr_dims(
            int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]);
    """
    ...
  def ggml_row_size(type: int, ne: int) -> int:
    """    size_t ggml_row_size (enum ggml_type type, int64_t ne);"""
    ...
  def ggml_scale(ctx: ffi.CData, a: ffi.CData, s: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_scale(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float s);
    """
    ...
  def ggml_scale_inplace(ctx: ffi.CData, a: ffi.CData, s: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_scale_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                float s);
    """
    ...
  def ggml_set(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_set(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t nb1,
                size_t nb2,
                size_t nb3,
                size_t offset);
    """
    ...
  def ggml_set_1d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_set_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t offset);
    """
    ...
  def ggml_set_1d_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_set_1d_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t offset);
    """
    ...
  def ggml_set_2d(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_set_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t nb1,
                size_t offset);
    """
    ...
  def ggml_set_2d_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_set_2d_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t nb1,
                size_t offset);
    """
    ...
  def ggml_set_f32(tensor: ffi.CData, value: float) -> ffi.CData:
    """    struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);"""
    ...
  def ggml_set_f32_1d(tensor: ffi.CData, i: int, value: float) -> None:
    """    void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);"""
    ...
  def ggml_set_f32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int, value: float) -> None:
    """    void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);"""
    ...
  def ggml_set_i32(tensor: ffi.CData, value: int) -> ffi.CData:
    """    struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);"""
    ...
  def ggml_set_i32_1d(tensor: ffi.CData, i: int, value: int) -> None:
    """    void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);"""
    ...
  def ggml_set_i32_nd(tensor: ffi.CData, i0: int, i1: int, i2: int, i3: int, value: int) -> None:
    """    void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);"""
    ...
  def ggml_set_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_set_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b,
                size_t nb1,
                size_t nb2,
                size_t nb3,
                size_t offset);
    """
    ...
  def ggml_set_input(tensor: ffi.CData) -> None:
    """    void ggml_set_input(struct ggml_tensor * tensor);"""
    ...
  def ggml_set_name(tensor: ffi.CData, name: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_set_name ( struct ggml_tensor * tensor, const char * name);"""
    ...
  def ggml_set_no_alloc(ctx: ffi.CData, no_alloc: bool) -> None:
    """
        void ggml_set_no_alloc(struct ggml_context * ctx,
                                                                     _Bool
                                                                          no_alloc);
    """
    ...
  def ggml_set_output(tensor: ffi.CData) -> None:
    """    void ggml_set_output(struct ggml_tensor * tensor);"""
    ...
  def ggml_set_param(ctx: ffi.CData, tensor: ffi.CData) -> None:
    """
        void ggml_set_param(
                struct ggml_context * ctx,
                struct ggml_tensor * tensor);
    """
    ...
  def ggml_set_scratch(ctx: ffi.CData, scratch: ffi.CData) -> int:
    """    size_t ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);"""
    ...
  def ggml_set_zero(tensor: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);"""
    ...
  def ggml_sgn(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sgn(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sgn_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sgn_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_silu(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_silu(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_silu_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_silu_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_silu_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_silu_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_soft_max(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_soft_max(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_soft_max_back(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_soft_max_back(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_soft_max_back_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_soft_max_back_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_soft_max_ext(ctx: ffi.CData, a: ffi.CData, mask: ffi.CData, pos: ffi.CData, scale: float, max_bias: float) -> ffi.CData:
    """
        struct ggml_tensor * ggml_soft_max_ext(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * mask,
                struct ggml_tensor * pos,
                float scale,
                float max_bias);
    """
    ...
  def ggml_soft_max_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_soft_max_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sqr(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sqr(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sqr_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sqr_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sqrt(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sqrt(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sqrt_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sqrt_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_ssm_conv(ctx: ffi.CData, s: ffi.CData, x: ffi.CData, c: ffi.CData, sq: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_ssm_conv(
                struct ggml_context * ctx,
                struct ggml_tensor * s,
                struct ggml_tensor * x,
                struct ggml_tensor * c,
                struct ggml_tensor * sq);
    """
    ...
  def ggml_ssm_scan(ctx: ffi.CData, s: ffi.CData, x: ffi.CData, dt: ffi.CData, A: ffi.CData, B: ffi.CData, C: ffi.CData, sq: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_ssm_scan(
                struct ggml_context * ctx,
                struct ggml_tensor * s,
                struct ggml_tensor * x,
                struct ggml_tensor * dt,
                struct ggml_tensor * A,
                struct ggml_tensor * B,
                struct ggml_tensor * C,
                struct ggml_tensor * sq);
    """
    ...
  def ggml_status_to_string(status: int) -> ffi.CData:
    """    const char * ggml_status_to_string(enum ggml_status status);"""
    ...
  def ggml_step(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_step(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_step_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_step_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sub(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sub(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_sub_inplace(ctx: ffi.CData, a: ffi.CData, b: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sub_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                struct ggml_tensor * b);
    """
    ...
  def ggml_sum(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sum(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_sum_rows(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_sum_rows(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_tanh(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_tanh(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_tanh_inplace(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_tanh_inplace(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_tensor_overhead() -> int:
    """    size_t ggml_tensor_overhead(void);"""
    ...
  def ggml_time_init() -> None:
    """    void ggml_time_init(void);"""
    ...
  def ggml_time_ms() -> int:
    """    int64_t ggml_time_ms(void);"""
    ...
  def ggml_time_us() -> int:
    """    int64_t ggml_time_us(void);"""
    ...
  def ggml_timestep_embedding(ctx: ffi.CData, timesteps: ffi.CData, dim: int, max_period: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_timestep_embedding(
                struct ggml_context * ctx,
                struct ggml_tensor * timesteps,
                int dim,
                int max_period);
    """
    ...
  def ggml_top_k(ctx: ffi.CData, a: ffi.CData, k: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_top_k(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int k);
    """
    ...
  def ggml_transpose(ctx: ffi.CData, a: ffi.CData) -> ffi.CData:
    """
        struct ggml_tensor * ggml_transpose(
                struct ggml_context * ctx,
                struct ggml_tensor * a);
    """
    ...
  def ggml_type_name(type: int) -> ffi.CData:
    """    const char * ggml_type_name(enum ggml_type type);"""
    ...
  def ggml_type_size(type: int) -> int:
    """    size_t ggml_type_size(enum ggml_type type);"""
    ...
  def ggml_type_sizef(type: int) -> float:
    """
        double ggml_type_sizef(enum ggml_type type)
                                      ;
    """
    ...
  def ggml_unary(ctx: ffi.CData, a: ffi.CData, op: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_unary(
                struct ggml_context * ctx,
                 struct ggml_tensor * a,
                 enum ggml_unary_op op);
    """
    ...
  def ggml_unary_inplace(ctx: ffi.CData, a: ffi.CData, op: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_unary_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor * a,
            enum ggml_unary_op op);
    """
    ...
  def ggml_unary_op_name(op: int) -> ffi.CData:
    """    const char * ggml_unary_op_name(enum ggml_unary_op op);"""
    ...
  def ggml_unravel_index(tensor: ffi.CData, i: int, i0: ffi.CData, i1: ffi.CData, i2: ffi.CData, i3: ffi.CData) -> None:
    """    void ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);"""
    ...
  def ggml_upscale(ctx: ffi.CData, a: ffi.CData, scale_factor: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_upscale(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int scale_factor);
    """
    ...
  def ggml_used_mem(ctx: ffi.CData) -> int:
    """    size_t ggml_used_mem(const struct ggml_context * ctx);"""
    ...
  def ggml_vec_dot_iq1_m_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq1_m_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq1_s_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq1_s_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq2_s_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq2_s_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq2_xs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq2_xs_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq2_xxs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq2_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq3_s_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq3_s_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq3_xxs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq3_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq4_nl_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq4_nl_q8_0 (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_iq4_xs_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_iq4_xs_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q2_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q2_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q3_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q3_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q4_0_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q4_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q4_1_q8_1(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q4_1_q8_1(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q4_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q4_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q5_0_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q5_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q5_1_q8_1(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q5_1_q8_1(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q5_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q6_K_q8_K(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q6_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_vec_dot_q8_0_q8_0(n: int, s: ffi.CData, bs: int, vx: ffi.CData, bx: int, vy: ffi.CData, by: int, nrc: int) -> None:
    """void ggml_vec_dot_q8_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc);"""
    ...
  def ggml_view_1d(ctx: ffi.CData, a: ffi.CData, ne0: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_view_1d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                size_t offset);
    """
    ...
  def ggml_view_2d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, nb1: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_view_2d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                size_t nb1,
                size_t offset);
    """
    ...
  def ggml_view_3d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_view_3d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                size_t nb1,
                size_t nb2,
                size_t offset);
    """
    ...
  def ggml_view_4d(ctx: ffi.CData, a: ffi.CData, ne0: int, ne1: int, ne2: int, ne3: int, nb1: int, nb2: int, nb3: int, offset: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_view_4d(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int64_t ne0,
                int64_t ne1,
                int64_t ne2,
                int64_t ne3,
                size_t nb1,
                size_t nb2,
                size_t nb3,
                size_t offset);
    """
    ...
  def ggml_view_tensor(ctx: ffi.CData, src: ffi.CData) -> ffi.CData:
    """    struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);"""
    ...
  def ggml_win_part(ctx: ffi.CData, a: ffi.CData, w: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_win_part(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int w);
    """
    ...
  def ggml_win_unpart(ctx: ffi.CData, a: ffi.CData, w0: int, h0: int, w: int) -> ffi.CData:
    """
        struct ggml_tensor * ggml_win_unpart(
                struct ggml_context * ctx,
                struct ggml_tensor * a,
                int w0,
                int h0,
                int w);
    """
    ...
  def gguf_add_tensor(ctx: ffi.CData, tensor: ffi.CData) -> None:
    """    void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);"""
    ...
  def gguf_find_key(ctx: ffi.CData, key: ffi.CData) -> int:
    """    int gguf_find_key(const struct gguf_context * ctx, const char * key);"""
    ...
  def gguf_find_tensor(ctx: ffi.CData, name: ffi.CData) -> int:
    """    int gguf_find_tensor (const struct gguf_context * ctx, const char * name);"""
    ...
  def gguf_free(ctx: ffi.CData) -> None:
    """    void gguf_free(struct gguf_context * ctx);"""
    ...
  def gguf_get_alignment(ctx: ffi.CData) -> int:
    """    size_t gguf_get_alignment (const struct gguf_context * ctx);"""
    ...
  def gguf_get_arr_data(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_arr_n(ctx: ffi.CData, key_id: int) -> int:
    """    int gguf_get_arr_n (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_arr_str(ctx: ffi.CData, key_id: int, i: int) -> ffi.CData:
    """    const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);"""
    ...
  def gguf_get_arr_type(ctx: ffi.CData, key_id: int) -> int:
    """    enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_data(ctx: ffi.CData) -> ffi.CData:
    """    void * gguf_get_data (const struct gguf_context * ctx);"""
    ...
  def gguf_get_data_offset(ctx: ffi.CData) -> int:
    """    size_t gguf_get_data_offset(const struct gguf_context * ctx);"""
    ...
  def gguf_get_key(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    const char * gguf_get_key (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_kv_type(ctx: ffi.CData, key_id: int) -> int:
    """    enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_meta_data(ctx: ffi.CData, data: ffi.CData) -> None:
    """    void gguf_get_meta_data(const struct gguf_context * ctx, void * data);"""
    ...
  def gguf_get_meta_size(ctx: ffi.CData) -> int:
    """    size_t gguf_get_meta_size(const struct gguf_context * ctx);"""
    ...
  def gguf_get_n_kv(ctx: ffi.CData) -> int:
    """    int gguf_get_n_kv(const struct gguf_context * ctx);"""
    ...
  def gguf_get_n_tensors(ctx: ffi.CData) -> int:
    """    int gguf_get_n_tensors (const struct gguf_context * ctx);"""
    ...
  def gguf_get_tensor_name(ctx: ffi.CData, i: int) -> ffi.CData:
    """    char * gguf_get_tensor_name (const struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_tensor_offset(ctx: ffi.CData, i: int) -> int:
    """    size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_tensor_type(ctx: ffi.CData, i: int) -> int:
    """    enum ggml_type gguf_get_tensor_type (const struct gguf_context * ctx, int i);"""
    ...
  def gguf_get_val_bool(ctx: ffi.CData, key_id: int) -> bool:
    """                         gguf_get_val_bool(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_data(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    const void * gguf_get_val_data(const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_f32(ctx: ffi.CData, key_id: int) -> float:
    """    float gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_f64(ctx: ffi.CData, key_id: int) -> float:
    """    double gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i16(ctx: ffi.CData, key_id: int) -> int:
    """    int16_t gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i32(ctx: ffi.CData, key_id: int) -> int:
    """    int32_t gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i64(ctx: ffi.CData, key_id: int) -> int:
    """    int64_t gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_i8(ctx: ffi.CData, key_id: int) -> int:
    """    int8_t gguf_get_val_i8 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_str(ctx: ffi.CData, key_id: int) -> ffi.CData:
    """    const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u16(ctx: ffi.CData, key_id: int) -> int:
    """    uint16_t gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u32(ctx: ffi.CData, key_id: int) -> int:
    """    uint32_t gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u64(ctx: ffi.CData, key_id: int) -> int:
    """    uint64_t gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_val_u8(ctx: ffi.CData, key_id: int) -> int:
    """    uint8_t gguf_get_val_u8 (const struct gguf_context * ctx, int key_id);"""
    ...
  def gguf_get_version(ctx: ffi.CData) -> int:
    """    int gguf_get_version (const struct gguf_context * ctx);"""
    ...
  def gguf_init_empty() -> ffi.CData:
    """    struct gguf_context * gguf_init_empty(void);"""
    ...
  def gguf_init_from_file(fname: ffi.CData, params: ffi.CData) -> ffi.CData:
    """    struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);"""
    ...
  def gguf_set_arr_data(ctx: ffi.CData, key: ffi.CData, type: int, data: ffi.CData, n: int) -> None:
    """    void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);"""
    ...
  def gguf_set_arr_str(ctx: ffi.CData, key: ffi.CData, data: ffi.CData, n: int) -> None:
    """    void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);"""
    ...
  def gguf_set_kv(ctx: ffi.CData, src: ffi.CData) -> None:
    """    void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);"""
    ...
  def gguf_set_tensor_data(ctx: ffi.CData, name: ffi.CData, data: ffi.CData, size: int) -> None:
    """    void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);"""
    ...
  def gguf_set_tensor_type(ctx: ffi.CData, name: ffi.CData, type: int) -> None:
    """    void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);"""
    ...
  def gguf_set_val_bool(ctx: ffi.CData, key: ffi.CData, val: bool) -> None:
    """
        void gguf_set_val_bool(struct gguf_context * ctx, const char * key,
                                                                                    _Bool
                                                                                             val);
    """
    ...
  def gguf_set_val_f32(ctx: ffi.CData, key: ffi.CData, val: float) -> None:
    """    void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float val);"""
    ...
  def gguf_set_val_f64(ctx: ffi.CData, key: ffi.CData, val: float) -> None:
    """    void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double val);"""
    ...
  def gguf_set_val_i16(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t val);"""
    ...
  def gguf_set_val_i32(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t val);"""
    ...
  def gguf_set_val_i64(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t val);"""
    ...
  def gguf_set_val_i8(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_i8 (struct gguf_context * ctx, const char * key, int8_t val);"""
    ...
  def gguf_set_val_str(ctx: ffi.CData, key: ffi.CData, val: ffi.CData) -> None:
    """    void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);"""
    ...
  def gguf_set_val_u16(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);"""
    ...
  def gguf_set_val_u32(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);"""
    ...
  def gguf_set_val_u64(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);"""
    ...
  def gguf_set_val_u8(ctx: ffi.CData, key: ffi.CData, val: int) -> None:
    """    void gguf_set_val_u8 (struct gguf_context * ctx, const char * key, uint8_t val);"""
    ...
  def gguf_type_name(type: int) -> ffi.CData:
    """    const char * gguf_type_name(enum gguf_type type);"""
    ...
  def gguf_write_to_file(ctx: ffi.CData, fname: ffi.CData, only_meta: bool) -> None:
    """
        void gguf_write_to_file(const struct gguf_context * ctx, const char * fname,
                                                                                             _Bool
                                                                                                  only_meta);
    """
    ...
  def iq2xs_free_impl(type: int) -> None:
    """void iq2xs_free_impl(enum ggml_type type);"""
    ...
  def iq2xs_init_impl(type: int) -> None:
    """void iq2xs_init_impl(enum ggml_type type);"""
    ...
  def iq3xs_free_impl(grid_size: int) -> None:
    """void iq3xs_free_impl(int grid_size);"""
    ...
  def iq3xs_init_impl(grid_size: int) -> None:
    """void iq3xs_init_impl(int grid_size);"""
    ...
  def quantize_iq1_m(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq1_m (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq1_s(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq1_s (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq2_s(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq2_s (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq2_xs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq2_xs (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq2_xxs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq2_xxs(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq3_s(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq3_s (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq3_xxs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq3_xxs(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq4_nl(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq4_nl (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_iq4_xs(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_iq4_xs (const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q2_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q2_K(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q3_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q3_K(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q4_0(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q4_0(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q4_1(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q4_1(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q4_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q4_K(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q5_0(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q5_0(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q5_1(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q5_1(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q5_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q5_K(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q6_K(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q6_K(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_q8_0(src: ffi.CData, dst: ffi.CData, nrows: int, n_per_row: int, imatrix: ffi.CData) -> int:
    """size_t quantize_q8_0(const float * restrict src, void * restrict dst, int nrows, int n_per_row, const float * imatrix);"""
    ...
  def quantize_row_iq2_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq2_s (const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_iq2_s_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq2_s_reference (const float * restrict x, block_iq2_s * restrict y, int k);"""
    ...
  def quantize_row_iq3_s(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_s (const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_iq3_s_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_s_reference (const float * restrict x, block_iq3_s * restrict y, int k);"""
    ...
  def quantize_row_iq3_xxs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_xxs(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_iq3_xxs_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq3_xxs_reference(const float * restrict x, block_iq3_xxs * restrict y, int k);"""
    ...
  def quantize_row_iq4_nl(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_nl (const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_iq4_nl_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_nl_reference (const float * restrict x, block_iq4_nl * restrict y, int k);"""
    ...
  def quantize_row_iq4_xs(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_xs (const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_iq4_xs_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_iq4_xs_reference (const float * restrict x, block_iq4_xs * restrict y, int k);"""
    ...
  def quantize_row_q2_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q2_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q2_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q2_K_reference(const float * restrict x, block_q2_K * restrict y, int k);"""
    ...
  def quantize_row_q3_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q3_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q3_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q3_K_reference(const float * restrict x, block_q3_K * restrict y, int k);"""
    ...
  def quantize_row_q4_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_0(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q4_0_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_0_reference(const float * restrict x, block_q4_0 * restrict y, int k);"""
    ...
  def quantize_row_q4_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_1(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q4_1_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_1_reference(const float * restrict x, block_q4_1 * restrict y, int k);"""
    ...
  def quantize_row_q4_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q4_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q4_K_reference(const float * restrict x, block_q4_K * restrict y, int k);"""
    ...
  def quantize_row_q5_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_0(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q5_0_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_0_reference(const float * restrict x, block_q5_0 * restrict y, int k);"""
    ...
  def quantize_row_q5_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_1(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q5_1_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_1_reference(const float * restrict x, block_q5_1 * restrict y, int k);"""
    ...
  def quantize_row_q5_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q5_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q5_K_reference(const float * restrict x, block_q5_K * restrict y, int k);"""
    ...
  def quantize_row_q6_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q6_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q6_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int k);"""
    ...
  def quantize_row_q8_0(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_0(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q8_0_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_0_reference(const float * restrict x, block_q8_0 * restrict y, int k);"""
    ...
  def quantize_row_q8_1(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_1(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q8_1_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_1_reference(const float * restrict x, block_q8_1 * restrict y, int k);"""
    ...
  def quantize_row_q8_K(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_K(const float * restrict x, void * restrict y, int k);"""
    ...
  def quantize_row_q8_K_reference(x: ffi.CData, y: ffi.CData, k: int) -> None:
    """void quantize_row_q8_K_reference(const float * restrict x, block_q8_K * restrict y, int k);"""
    ...