# Trajectory Legibility VLM Evaluation - Results

## Summary

| Metric | v1 (Baseline) | v2 (Grounded) | Change |
|--------|:---:|:---:|:---:|
| **Overall Accuracy** | 18/40 (45.0%) | **39/40 (97.5%)** | **+52.5 pp** |
| Left Arc Accuracy | 9/20 (45.0%) | **19/20 (95.0%)** | +50.0 pp |
| Right Arc Accuracy | 9/20 (45.0%) | **20/20 (100.0%)** | +55.0 pp |
| Prompt Length | 26,000 chars | **1,180 chars** | 22x shorter |
| Temperature | 0.7 | **0.1** | - |
| Mean Legibility Score | 0.923 | **0.919** | comparable |

## Confusion Matrix (v2)

|  | Predicted LEFT | Predicted RIGHT |
|--|:-:|:-:|
| **GT LEFT** | **19** | 1 |
| **GT RIGHT** | 0 | **20** |

- Precision (LEFT): 19/19 = 100%
- Precision (RIGHT): 20/21 = 95.2%
- Recall (LEFT): 19/20 = 95.0%
- Recall (RIGHT): 20/20 = 100%

## Root Cause Analysis

The v1 system achieved only **45% accuracy** (random chance) due to 5 compounding issues:

### 1. Camera Perspective Confusion (Critical)
The PyBullet camera uses **yaw=135°**, placing the viewpoint in the +Y/-X quadrant. This causes a **left-right reversal**: the world "left" block (Y=+0.07) appears on the **right** side of the image, and vice versa. The v1 prompt contained no camera perspective information, so the VLM mapped image directions to world labels randomly.

### 2. Prompt Overload (26K chars)
The v1 prompt was a 26,000-character generic trajectory legibility framework with XML-structured sections for role definition, capabilities, analysis stages, and output schema. This overwhelmed the model with abstract theory instead of grounding it in the actual visual content.

### 3. Temperature Too High (0.7)
For a perceptual identification task, temperature 0.7 adds unnecessary randomness. The VLM's direction perception varied across runs even for identical videos.

### 4. No Visual Anchoring
The v1 system sent only the video with no reference frame. The VLM had to simultaneously identify blocks, track trajectories, and determine spatial relationships without any grounding.

### 5. Identical Block Appearance
Both blocks are identical red cubes, only 14cm apart. Without explicit markers, the VLM cannot distinguish them by appearance alone.

## v2 Fix: Grounded Visual Analysis

Five targeted fixes resolved all issues:

1. **Annotated Reference Frame**: First video frame with colored circles (blue=Block A, green=Block B) and labels identifying each block's image position
2. **Image-Space Reasoning**: VLM answers in image coordinates ("Block A or B"), evaluation harness maps to world coordinates using known camera calibration
3. **Focused Prompt** (1,180 chars): Describes the scene, references the annotated image, asks specific questions about endpoint and trajectory shape
4. **Low Temperature** (0.1): Deterministic perceptual judgments
5. **Multi-Modal Input**: [annotated_frame + video + prompt] instead of [video + 26K_prompt]

## Model & Configuration

- **Model**: Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`)
- **Input**: Annotated first frame (PNG, ~94KB) + Video (MP4, ~200KB, 640x480, 30fps, ~10s)
- **Thinking Budget**: 4,096 tokens
- **Temperature**: 0.1
- **Response Format**: Structured JSON

## Error Analysis

Single incorrect prediction: `cfg00_left_arc12.mp4`
- VLM predicted Block A (image-left = world-right), but ground truth is left
- Arc magnitude: 0.195 (moderate curvature)
- Error rate: 2.5% (1/40)

## Key Insight

The VLM's visual perception was never the bottleneck — even in v1, its observations were internally consistent (100% correlation between predicted lateral bias and target). The failure was entirely a **grounding problem**: without camera perspective information and visual anchoring, the VLM could not map image observations to world-coordinate labels. Providing minimal but correct grounding (annotated reference frame + short focused prompt) resolved this completely.
