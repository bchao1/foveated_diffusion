#!/bin/bash
# ============================================================================
# Z-Image Foveated Inference Script
# 
# Usage:
#   ./inference.sh <GPU_ID> <EXPERIMENT> [OPTIONS]
#
# Examples:
#   ./inference.sh 0 circular                    # Circular with defaults
#   ./inference.sh 0 radius_sweep                # Radius sweep with defaults
#   ./inference.sh 0 circular --no-foveation     # Circular without foveation (baseline)
#   ./inference.sh 0 baseline                    # Single baseline image
# ============================================================================

# Default configuration
MODEL_FOLDER="/miele/brian/modelscope/Tongyi-MAI/Z-Image"
NUM_FRAMES=100
ORBIT_RADIUS=0.25
MASK_RADIUS=0.30
RADIUS_START=0.1
RADIUS_END=1.0
VAE_DECODE_MODE="merge"
VAE_MERGE_DOWNSAMPLE_MODE="nearest"
USE_FOVEATION=true
OUTPUT_DIR=""

# Parse arguments
GPU_ID=${1:-0}
EXPERIMENT=${2:-circular}
shift 2 || shift $#

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-foveation)
            USE_FOVEATION=false
            shift
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --orbit-radius)
            ORBIT_RADIUS="$2"
            shift 2
            ;;
        --mask-radius)
            MASK_RADIUS="$2"
            shift 2
            ;;
        --radius-start)
            RADIUS_START="$2"
            shift 2
            ;;
        --radius-end)
            RADIUS_END="$2"
            shift 2
            ;;
        --vae-mode)
            VAE_DECODE_MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default output directory based on experiment type
if [ -z "$OUTPUT_DIR" ]; then
    if [ "$USE_FOVEATION" = true ]; then
        OUTPUT_DIR="./outputs/z_image_${EXPERIMENT}_foveated"
    else
        OUTPUT_DIR="./outputs/z_image_${EXPERIMENT}_baseline"
    fi
fi

# Build command
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python Z-Image-Foveated.py"
CMD="$CMD --model_folder $MODEL_FOLDER"
CMD="$CMD --experiment $EXPERIMENT"
CMD="$CMD --num_frames $NUM_FRAMES"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --vae_decode_mode $VAE_DECODE_MODE"
CMD="$CMD --vae_merge_downsample_mode $VAE_MERGE_DOWNSAMPLE_MODE"

# Add experiment-specific parameters
case $EXPERIMENT in
    circular)
        CMD="$CMD --orbit_radius $ORBIT_RADIUS"
        CMD="$CMD --mask_radius $MASK_RADIUS"
        ;;
    radius_sweep)
        CMD="$CMD --radius_start $RADIUS_START"
        CMD="$CMD --radius_end $RADIUS_END"
        ;;
    baseline)
        # No additional parameters needed
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: circular, radius_sweep, baseline"
        exit 1
        ;;
esac

# Add foveation flag
if [ "$USE_FOVEATION" = true ]; then
    CMD="$CMD --use_foveation"
fi

# Print and execute
echo "============================================================================"
echo "Running experiment: $EXPERIMENT"
echo "  GPU: $GPU_ID"
echo "  Foveation: $USE_FOVEATION"
echo "  Output: $OUTPUT_DIR"
echo "============================================================================"
echo ""
echo "Command: $CMD"
echo ""

eval $CMD
