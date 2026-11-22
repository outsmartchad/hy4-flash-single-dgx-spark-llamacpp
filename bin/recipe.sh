#!/usr/bin/env bash
set -euo pipefail

readonly source_repo=https://github.com/ggml-org/llama.cpp.git
readonly source_rev=2cbca389f515df455412dafa07e2b1359cc72a6d
readonly hub=https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/bcbc7db5dcf4162cc1e1b8ed6c65567e843ebe46
readonly patch_sha=6ec50722c8b181222a928e0a66cda0159e8dff3e72f8193334f6296f6213dfff
readonly data_dir=${HY4_DATA_DIR:-"$HOME/.cache/hy4-flash-spark"}
readonly source_dir="$data_dir/llama.cpp"
readonly build_dir="$source_dir/build-spark"
readonly port=${HY4_PORT:-18911}

case "${2:-standard}" in
    standard)
        weight=Hy4-flash-SPARK-4Bit.gguf
        weight_sha=f92c1b4014974e978b0e9a14fdead50b1f2b46c3ca21b8949775baa52e29f963
        ;;
    fast)
        weight=Hy4-flash-SPARK-4Bit-Fast.gguf
        weight_sha=b402322e67879ce1b65258e2dc1c3b3d6e25eeea074a391b5c29d2ac8b331613
        ;;
    *) printf 'Use standard or fast for the model variant.\n' >&2; exit 2 ;;
esac

select_cuda() {
    if [[ -z "${CUDACXX:-}" ]]; then
        export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
    fi
    test -x "$CUDACXX" || { printf 'Set CUDACXX to the absolute path to your CUDA compiler.\n' >&2; return 1; }
}

check_sha() {
    printf '%s  %s\n' "$2" "$1" | sha256sum --check --status
}

fetch() {
    local name=$1 digest=$2 final="$data_dir/${1##*/}" partial
    partial="$final.part"
    mkdir -p "$data_dir"
    if test -f "$final"; then
        check_sha "$final" "$digest" || { printf 'Checksum mismatch for %s\n' "$final" >&2; return 1; }
        return
    fi
    curl --fail --show-error --location --retry 4 --continue-at - "$hub/$name" --output "$partial"
    check_sha "$partial" "$digest" || { printf 'Checksum mismatch for %s\n' "$partial" >&2; return 1; }
    mv -n -- "$partial" "$final"
}

prepare_source() {
    fetch patch/hy4-flash-spark.patch "$patch_sha"
    if [[ ! -d "$source_dir/.git" ]]; then
        git init --quiet "$source_dir"
    fi
    if ! git -C "$source_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$source_dir" fetch --depth=1 "$source_repo" "$source_rev"
        git -C "$source_dir" checkout --detach FETCH_HEAD
    fi
    test "$(git -C "$source_dir" rev-parse HEAD)" = "$source_rev"
    if git -C "$source_dir" apply --reverse --check "$data_dir/hy4-flash-spark.patch" 2>/dev/null; then
        return
    fi
    git -C "$source_dir" diff --quiet
    git -C "$source_dir" apply --check "$data_dir/hy4-flash-spark.patch"
    git -C "$source_dir" apply "$data_dir/hy4-flash-spark.patch"
}

case "${1:-}" in
    doctor)
        [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]]
        for command in cmake git curl sha256sum nvidia-smi python3; do command -v "$command"; done
        select_cuda
        "$CUDACXX" --version
        nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
        mkdir -p "$data_dir"
        df -h "$data_dir"
        ;;
    fetch)
        fetch "$weight" "$weight_sha"
        ;;
    build)
        select_cuda
        prepare_source
        cmake -S "$source_dir" -B "$build_dir" \
            -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a \
            -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
        cmake --build "$build_dir" --config Release --target llama-server llama-bench --parallel "${HY4_BUILD_JOBS:-8}"
        ;;
    serve)
        test -s "$data_dir/$weight"
        test -x "$build_dir/bin/llama-server"
        profile=${HY4_SPECULATION:-mtp}
        case "$profile" in
            mtp) draft=(--spec-type draft-mtp --spec-draft-n-max "${HY4_DRAFT_MAX:-3}") ;;
            off) draft=(--spec-type none) ;;
            *) printf 'HY4_SPECULATION must be mtp or off.\n' >&2; exit 2 ;;
        esac
        export CUDA_VISIBLE_DEVICES=0
        export GGML_SCHED_BATCH_HOST_INPUTS=0
        export GGML_CUDA_FUSE_HC_GATE=1
        export GGML_CUDA_FUSE_HC_CONSUMER=1
        export GGML_CUDA_REUSE_Q8=1
        export GGML_CUDA_SPARK_MMQ_FUSION=1
        export GGML_CUDA_SPARK_MMQ_COMPACT_GRID=1
        export GGML_CUDA_SPARK_MMQ_TILE=64
        export GGML_CUDA_SPARK_MMQ_TILE_FUSED=64
        export GGML_CUDA_MMQ_PREC=q4
        export GGML_CUDA_CPY_F32_VEC=1
        export GGML_CUDA_SPARK_MMVQ_2W=1
        export GGML_CUDA_MOE_CONCURRENT=1
        export GGML_CUDA_FA_576_WHOLE_TILES=0
        exec "$build_dir/bin/llama-server" \
            --model "$data_dir/$weight" --alias hy4-flash --host 127.0.0.1 --port "$port" \
            --ctx-size "${HY4_CONTEXT:-196608}" --parallel "${HY4_PARALLEL:-1}" \
            --n-gpu-layers 999 --split-mode none \
            --override-tensor '^per_layer_token_embd\.weight$=CPU,^token_embd\.weight$=CUDA0' \
            --load-mode mmap --lazy-mode on --no-host --fit off --flash-attn on \
            --batch-size 2048 --ubatch-size "${HY4_UBATCH:-1024}" --threads 4 --threads-batch 4 \
            --cache-type-k f16 --cache-type-v f16 "${draft[@]}" --metrics --jinja
        ;;
    *)
        printf 'Usage: %s {doctor|fetch|build|serve} [standard|fast]\n' "${0##*/}" >&2
        exit 2
        ;;
esac
