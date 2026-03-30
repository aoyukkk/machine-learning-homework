#!/bin/bash
# ============================================================
# 实验八：生成模型对比 — 服务器启动脚本
# ============================================================
# 用法:
#   chmod +x run_server.sh
#   ./run_server.sh                           # 自动检测所有可用 GPU
#   ./run_server.sh --gpus 0,2,5              # 只用第 0,2,5 号 GPU
#   ./run_server.sh --num-gpus 4              # 只用前 4 张可用 GPU
#   ./run_server.sh --quick-test              # 快速测试
#   ./run_server.sh --skip-train              # 跳过训练，仅评估
#   ./run_server.sh --gpus 0,1 --quick-test   # 指定 GPU + 快速测试
#
# 分布式多卡训练 (torchrun):
#   ./run_server.sh --ddp                      # 所有可用 GPU 上 DDP
#   ./run_server.sh --ddp --gpus 0,2,5         # 指定 GPU 上 DDP
#   ./run_server.sh --ddp --num-gpus 4         # 前 4 张 GPU 上 DDP
#
# 单卡模式 (无 torchrun, 用 --gpus 或 --num-gpus 选择):
#   ./run_server.sh --gpus 3                   # 只在 GPU 3 上跑
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/src"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate machine

# 确保进度条在分布式环境中正常显示
export FORCE_TQDM=1
# 防止 NCCL 超时
export NCCL_TIMEOUT=1800
# 优化 NCCL 性能
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# ---- 解析 --ddp 和 --gpus/--num-gpus 参数 ----
USE_DDP=false
GPU_IDS=""
NUM_GPUS=""
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --ddp)
            USE_DDP=true
            ;;
        --gpus=*)
            GPU_IDS="${arg#--gpus=}"
            ;;
        --num-gpus=*)
            NUM_GPUS="${arg#--num-gpus=}"
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

# 如果 --gpus 和 --num-gpus 是作为两个参数传入（带空格），也要处理
i=0
FINAL_ARGS=()
while [ $i -lt ${#PASSTHROUGH_ARGS[@]} ]; do
    arg="${PASSTHROUGH_ARGS[$i]}"
    case "$arg" in
        --gpus)
            i=$((i+1))
            GPU_IDS="${PASSTHROUGH_ARGS[$i]}"
            ;;
        --num-gpus)
            i=$((i+1))
            NUM_GPUS="${PASSTHROUGH_ARGS[$i]}"
            ;;
        *)
            FINAL_ARGS+=("$arg")
            ;;
    esac
    i=$((i+1))
done

echo "============================================================"
echo "  实验八：生成模型对比（Diffusion / Flow Matching / Autoregressive）"

if [ "$USE_DDP" = true ]; then
    # ---- DDP 多卡模式 (torchrun) ----

    # 确定要用哪些 GPU
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        # 计算 GPU 数量
        DDP_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
    elif [ -n "$NUM_GPUS" ]; then
        # 获取所有 GPU 数量，取 min
        ALL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
        DDP_GPUS=$(( ALL_GPUS < NUM_GPUS ? ALL_GPUS : NUM_GPUS ))
        # 选择前 N 张 GPU
        SEQ=$(python -c "print(','.join(str(i) for i in range($DDP_GPUS)))")
        export CUDA_VISIBLE_DEVICES="$SEQ"
    else
        DDP_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    fi

    echo "  模式: DDP 分布式训练"
    echo "  GPU 数量: $DDP_GPUS"
    [ -n "$CUDA_VISIBLE_DEVICES" ] && echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "============================================================"

    torchrun \
        --nproc_per_node=$DDP_GPUS \
        --master_port=29500 \
        train_compare_models.py \
        --profile multi_gpu_8x24g \
        "${FINAL_ARGS[@]}"
else
    # ---- 单进程模式 (python 直接启动, GPU 选择由代码内部处理) ----

    # 把 --gpus 和 --num-gpus 传给 Python 脚本
    PYTHON_EXTRA_ARGS=()
    [ -n "$GPU_IDS" ] && PYTHON_EXTRA_ARGS+=(--gpus "$GPU_IDS")
    [ -n "$NUM_GPUS" ] && PYTHON_EXTRA_ARGS+=(--num-gpus "$NUM_GPUS")

    DETECTED_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    echo "  模式: 单进程 (DataParallel)"
    echo "  系统检测到 GPU 数量: $DETECTED_GPUS"
    [ -n "$GPU_IDS" ] && echo "  指定使用 GPU: $GPU_IDS"
    [ -n "$NUM_GPUS" ] && echo "  限制最多使用 GPU: $NUM_GPUS 张"
    echo "============================================================"

    python train_compare_models.py \
        "${PYTHON_EXTRA_ARGS[@]}" \
        "${FINAL_ARGS[@]}"
fi

echo ""
echo "============================================================"
echo "  ✅ 训练完成！"
echo "  结果目录: $SCRIPT_DIR/outputs/"
echo "  图片目录: $SCRIPT_DIR/figures/"
echo "  报告文件: $SCRIPT_DIR/report8.tex"
echo "============================================================"
