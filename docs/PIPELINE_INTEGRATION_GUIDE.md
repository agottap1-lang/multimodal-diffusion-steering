# Gemini VLM Evaluation Pipeline Integration

## 📋 Pipeline Understanding

### Directory Structure
```
C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\
├── .env                          # API key storage (GEMINI_API_KEY)
├── .env.example
├── scripts/
│   ├── eval_dataset.py          # Main evaluation pipeline
│   ├── eval_video.py            # Single video evaluation
│   ├── evaluate_video.py
│   └── [15+ analysis scripts]
├── src/
│   └── gemini_vlm_eval/
│       ├── client.py            # GeminiClient class
│       ├── schema.py            # ManifestEntry, EvaluationResult
│       ├── prompt.py            # Legibility prompt generator
│       ├── video.py             # Frame extraction (OpenCV)
│       └── runner.py
├── data/                        # Video datasets
├── outputs/                     # Evaluation results
└── videos/                      # Input videos
```

## 🔄 How The Pipeline Works

### 1. **API Key Management**
The pipeline uses `python-dotenv` to load the Gemini API key from `.env`:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file from current directory
api_key = os.getenv("GEMINI_API_KEY")
```

**Current .env content:**
```dotenv
GEMINI_API_KEY=AIza...REDACTED...
```

⚠️ **ISSUE**: This API key has been leaked and revoked by Google.

### 2. **Frame Extraction** (video.py)

```python
def extract_frames(video_path: str, sample_rate_seconds: float = 1.0):
    """Extract frames from video at exact time intervals"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample at 0s, 1s, 2s, ...
    for target_time in [0.0, 1.0, 2.0, ...]:
        frame_num = int(target_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        success, jpeg_bytes = cv2.imencode('.jpg', frame)
        yield frame_num, jpeg_bytes, target_time
```

**Key features:**
- Extracts frames at **1-second intervals** by default
- Converts to **JPEG bytes** for VLM input
- Can cache frames to disk as PNG for visualization
- Returns dict: `{t_sec: {"frame_idx": int, "jpeg_bytes": bytes}}`

### 3. **Legibility Evaluation** (client.py + prompt.py)

**Prompt structure:**
```python
def get_instruction_prompt(goal_A, goal_B, t_sec, video_id, mode):
    """
    Generates prompt for Gemini to evaluate legibility
    
    Two modes:
    1. "single_frame": Evaluate ONLY the current frame at t_sec
    2. "prefix_frames": Evaluate ALL frames from 0 to t_sec
    """
```

**Example prompt (single_frame mode):**
```
You are evaluating LEGIBILITY: how easily a typical human observer can infer 
the actor's intended goal from what is visible NOW.

You are given ONLY ONE image: a single video frame at t=5s from video_id="rollout_001".

There are exactly two candidate goals:
Goal A: pick the left block
Goal B: pick the right block

Estimate probabilities:
- pA = P(Goal A | frames)
- pB = P(Goal B | frames)

Constraints: 0 <= pA,pB <= 1, pA + pB = 1

Provide ONE short visual cue and legibility status:
- "legible_now" if goal is inferable, else "not_legible_yet"

Output ONLY JSON: {"pA": 0.62, "pB": 0.38, "cue": "gripper aligned with left block", "legible": "legible_now"}
```

**GeminiClient.evaluate_frame():**
```python
def evaluate_frame(image_bytes, manifest_entry, t_sec, frame_idx, mode):
    """
    1. Build prompt from manifest_entry metadata
    2. Send image(s) + prompt to Gemini API
    3. Parse JSON response: {"pA": float, "pB": float, "cue": str, "legible": str}
    4. Compute derived metrics: choice (A/B/C), confidence (0-100)
    5. Return EvaluationResult with full API metadata
    """
```

**Retry logic:**
- 3 attempts per frame
- Exponential backoff (1s, 2s, 3s)
- Logs all failures

### 4. **Schemas** (schema.py)

**ManifestEntry** - Video metadata:
```python
@dataclass
class ManifestEntry:
    video_id: str          # "rollout_001"
    video_path: str        # "data/videos/rollout_001.mp4"
    goal_gt: str           # "A" or "B" (ground truth)
    goal_A: str            # "pick the left block"
    goal_B: str            # "pick the right block"
    scene_id: str          # "twoblockpick"
    task_family: str       # "block_pick"
    traj_type: str         # "optimal", "suboptimal", "predicted"
    notes: Optional[str]   # Free text
```

**EvaluationResult** - API response + metadata:
```python
@dataclass
class EvaluationResult:
    # Core evaluation
    video_id: str
    t_sec: int
    pA: float              # P(Goal A | frames)
    pB: float              # P(Goal B | frames)
    choice: str            # 'A', 'B', or 'C' (uncertain)
    confidence: int        # 0-100
    cue: str               # Visual cue from VLM
    legible: str           # "legible_now" or "not_legible_yet"
    
    # API metadata (reproducibility)
    model: str             # "gemini-2.5-flash"
    provider: str          # "google"
    endpoint: str
    temperature: float
    latency_ms: int
    http_status: int
    retry_count: int
    # ... and more
```

### 5. **Evaluation Pipeline** (eval_dataset.py)

**Main workflow:**
```python
# 1. Load manifest
entries = load_manifest("data/manifest.jsonl")

# 2. For each video:
for entry in entries:
    # 3. Extract frames at 1s intervals
    frames = extract_and_cache_frames(entry.video_path, k_seconds=10)
    
    # 4. For each timestamp:
    for t_sec in [0, 1, 2, ..., k]:
        # 5. Prepare image data
        if mode == "single_frame":
            image = frames[t_sec]["jpeg_bytes"]
        else:  # prefix_frames
            image = [frames[t]["jpeg_bytes"] for t in range(0, t_sec+1)]
        
        # 6. Evaluate with VLM
        result = client.evaluate_frame(image, entry, t_sec, frame_idx, mode)
        
        # 7. Save result as JSONL
        output_file.write(result.model_dump_json() + '\n')
```

**Output format:**
```jsonl
{"video_id": "rollout_001", "t_sec": 0, "pA": 0.5, "pB": 0.5, "choice": "C", ...}
{"video_id": "rollout_001", "t_sec": 1, "pA": 0.52, "pB": 0.48, "choice": "A", ...}
{"video_id": "rollout_001", "t_sec": 2, "pA": 0.68, "pB": 0.32, "choice": "A", ...}
...
```

## 🔗 Our Integration

### What We Did

1. **Created wrapper** ([scripts/vlm_client.py](c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick\scripts\vlm_client.py)):
   ```python
   class LegibilityScorer:
       def __init__(self):
           # Load API key from gemini_vlm_eval/.env
           load_dotenv(Path("C:/Users/anude/.../gemini_vlm_eval/.env"))
           self.client = GeminiClient()  # Uses existing pipeline
       
       def score_trajectory(self, image_bytes, goal_A, goal_B):
           # Create ManifestEntry for trajectory
           manifest = ManifestEntry(video_id="candidate_0", ...)
           
           # Use existing pipeline
           result = self.client.evaluate_frame(image_bytes, manifest, ...)
           
           # Return simplified dict
           return {"pA": result.pA, "pB": result.pB, ...}
   ```

2. **Created trajectory visualizer** ([scripts/trajectory_visualizer.py](c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick\scripts\trajectory_visualizer.py)):
   - Renders PyBullet environment frames
   - Overlays predicted trajectory as arrows
   - Converts to JPEG bytes for VLM input

3. **Created VLM-guided policy** ([scripts/vlm_guided_policy.py](c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick\scripts\vlm_guided_policy.py)):
   - Samples N trajectories from diffusion policy
   - Visualizes each trajectory
   - Queries VLM for legibility scores
   - Selects most legible trajectory

4. **Created evaluation script** ([scripts/eval_legibility_steering.py](c:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick\scripts\eval_legibility_steering.py)):
   - Runs full episodes with legibility steering
   - Saves videos with trajectory overlays
   - Collects metrics (success rate, legibility scores, VLM latency)

### Differences from Original Pipeline

| Original Pipeline | Our Integration |
|-------------------|-----------------|
| Evaluates **recorded videos** | Evaluates **real-time trajectories** |
| Frames extracted from MP4 | Frames rendered from PyBullet |
| Batch evaluation (all videos) | Online evaluation (per timestep) |
| Post-hoc analysis | Real-time trajectory selection |
| `eval_dataset.py` | `eval_legibility_steering.py` |

## 🚨 Current Issue: Leaked API Key

**Problem:**
```
403 PERMISSION_DENIED: Your API key was reported as leaked.
```

The API key in `gemini_vlm_eval/.env` is:
```
AIza...REDACTED...
```

This key has been exposed publicly and Google has revoked it.

**Solution:**

1. **Get new API key:**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Click "Create API Key"
   - Copy the key (format: `AIzaSy...`)

2. **Update .env file:**
   ```powershell
   # Edit the gemini_vlm_eval .env file
   notepad "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval\.env"
   ```
   
   Replace with:
   ```dotenv
   GEMINI_API_KEY=YOUR_NEW_KEY_HERE
   ```

3. **Verify it works:**
   ```powershell
   cd "C:\Users\anude\OneDrive\Documents\gemini_vlm_eval"
   python scripts/eval_image.py test_image.jpg
   ```

4. **Run legibility evaluation:**
   ```powershell
   cd "C:\Users\anude\OneDrive\Documents\multimodal diffusion for twoblockpick"
   py scripts/eval_legibility_steering.py --checkpoint runs/diffusion_20260222_195530/ckpt_ep100.pt --n_episodes 3 --n_samples 3 --save_videos
   ```

## 📊 Expected Behavior

Once API key is fixed, the system will:

1. **Sample trajectories**: Generate 3 candidate trajectories from diffusion policy
2. **Visualize each**: Render PyBullet frame with trajectory overlay
3. **Query VLM**: Send to Gemini 2.5 Flash (~1-2s per call)
4. **Get legibility scores**: 
   - `pA=0.75, pB=0.25` → Left block more legible
   - `pA=0.35, pB=0.65` → Right block more legible
   - `pA=0.52, pB=0.48` → Ambiguous (choose randomly)
5. **Select best**: Execute trajectory with highest max(pA, pB)
6. **Repeat**: Every replanning step (default: every 8 actions)

**Example output:**
```
Episode 1/3: Success=True, Reward=1.0, Steps=95, VLM Calls=12, Avg Legibility=0.823
Episode 2/3: Success=True, Reward=1.0, Steps=87, VLM Calls=11, Avg Legibility=0.791
Episode 3/3: Success=False, Reward=0.0, Steps=200, VLM Calls=25, Avg Legibility=0.612
=======================================================================
Success Rate: 66.7% (2/3)
Average Legibility Score: 0.742
Total VLM Calls: 48
Avg Latency per Call: 1234ms
=======================================================================
```

## 🔑 Key Takeaways

### Pipeline Architecture
- **Modular design**: Video extraction → VLM evaluation → Analysis
- **Full reproducibility**: All API metadata saved to JSONL
- **Two evaluation modes**: Single frame vs. prefix frames
- **Robust error handling**: 3 retries, detailed logging

### Integration Benefits
- **Reuses production code**: No reimplementation needed
- **Full metadata tracking**: Same reproducibility guarantees
- **Consistent prompts**: Same legibility definition
- **Same API client**: Handles retries, rate limits, errors

### Next Steps
1. ✅ Fix API key in `.env` file
2. ✅ Run quick test (3 episodes)
3. Compare to baseline (no steering)
4. Tune parameters (n_samples, rerank_frequency)
5. Full evaluation (20+ episodes)
6. Analyze videos and metrics

---

**Status**: Integration complete, waiting for valid API key ⏳
