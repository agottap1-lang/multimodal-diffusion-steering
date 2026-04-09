#!/usr/bin/env python3
"""
Test VLM Context Hypothesis
============================
For each video in data/demos/demo_videos_v2/:
  A) BLIND prompt  – current eval_cfg_vlm.py style (no target, no metrics)
  B) ENRICHED prompt – target identity + metric definitions + context

Compare: does enriched scoring align better with ground-truth behavior?

Video naming: {behavior}_{target}_{cfgNN}_{vNN}.mp4
  behavior: legible, predictable, safety_legible, safety_predictable, grounding
  target: left, right
  cfgNN: CFG scale used (cfg00=0, cfg02=2.0, ...)
  vNN: variant
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

GEMINI_API_KEY = "AIzaSyC49xYA6yGPLCQ0Ma-sFIIXYs-pfp38f0g"
GEMINI_MODEL = "gemini-3-pro-preview"

GOAL_A_DESC = "pick the LEFT (green) block"
GOAL_B_DESC = "pick the RIGHT (red) block"

# ── Parse video filename ──────────────────────────────────────────────

def parse_video_name(name: str) -> dict:
    """Extract behavior, target, cfg_scale from filename."""
    stem = Path(name).stem  # e.g. legible_right_cfg02_v00
    # Handle compound behaviors: safety_legible, safety_predictable
    if stem.startswith("safety_legible"):
        behavior = "safety_legible"
        rest = stem[len("safety_legible_"):]
    elif stem.startswith("safety_predictable"):
        behavior = "safety_predictable"
        rest = stem[len("safety_predictable_"):]
    elif stem.startswith("grounding"):
        behavior = "grounding"
        rest = stem[len("grounding_"):]
    elif stem.startswith("legible"):
        behavior = "legible"
        rest = stem[len("legible_"):]
    elif stem.startswith("predictable"):
        behavior = "predictable"
        rest = stem[len("predictable_"):]
    else:
        return {"behavior": "unknown", "target": "unknown", "cfg": 0, "var": 0}

    parts = rest.split("_")
    target = parts[0] if parts else "unknown"
    cfg_str = parts[1] if len(parts) > 1 else "cfg00"
    var_str = parts[2] if len(parts) > 2 else "v00"
    cfg_val = float(cfg_str.replace("cfg", "")) / 10.0 if "cfg" in cfg_str else 0.0
    # cfg00=0.0, cfg01=0.1, cfg02=0.2, ..., cfg09=0.9
    # But our actual lambda multiplier: cfg02 likely means lambda=2.0
    # Let's parse as integer then decide
    cfg_int = int(cfg_str.replace("cfg", "")) if "cfg" in cfg_str else 0
    var_int = int(var_str.replace("v", "")) if "v" in var_str else 0

    return {"behavior": behavior, "target": target,
            "cfg_int": cfg_int, "var": var_int, "stem": stem}


# ── Map behavior to VLM evaluation category ──────────────────────────

def get_eval_behavior(behavior: str) -> str:
    """Map video behavior to scoring category."""
    if behavior in ("legible", "safety_legible"):
        return "legibility"
    elif behavior in ("predictable", "safety_predictable"):
        return "predictability"
    elif behavior == "grounding":
        return "grounding"
    return "legibility"


# ── BLIND prompt (current eval_cfg_vlm.py style) ─────────────────────

def build_blind_prompt(eval_behavior: str, n_frames_note: str = "video") -> str:
    base = (
        f"You are a robotics expert evaluating a Franka Panda robot arm trajectory.\n"
        f"The robot is performing a block-picking task with two target blocks:\n"
        f"  Goal A: {GOAL_A_DESC}\n"
        f"  Goal B: {GOAL_B_DESC}\n\n"
        f"You are watching a {n_frames_note} of the trajectory (~5 seconds).\n"
        f"The arm starts from a home position above the table and moves toward one block.\n\n"
    )
    if eval_behavior == "legibility":
        return base + (
            "TASK: Rate how LEGIBLE this trajectory is.\n"
            "A legible trajectory clearly reveals which block the robot intends to pick "
            "as EARLY as possible. The arm should curve toward the target so an observer "
            "can predict the goal from the first few seconds.\n\n"
            "Rate:\n"
            "  pA = probability robot is going for Goal A (left block)\n"
            "  pB = probability robot is going for Goal B (right block)\n"
            "  legibility = 0.0 (ambiguous) to 1.0 (crystal clear from start)\n\n"
            'Output ONLY valid JSON: {"pA": X, "pB": X, "legibility": X, '
            '"cue": "brief description"}'
        )
    elif eval_behavior == "predictability":
        return base + (
            "TASK: Rate how PREDICTABLE this trajectory is.\n"
            "A predictable trajectory takes the SHORTEST, MOST DIRECT path.\n"
            "No arc, no sweep, no unnecessary lateral motion.\n\n"
            "Rate:\n"
            "  pA = probability going for Goal A\n"
            "  pB = probability going for Goal B\n"
            "  predictability = 0.0 (erratic/curved) to 1.0 (perfectly straight)\n\n"
            'Output ONLY valid JSON: {"pA": X, "pB": X, "predictability": X, '
            '"cue": "brief description"}'
        )
    elif eval_behavior == "grounding":
        return base + (
            "TASK: Rate how well this trajectory follows a spatial instruction.\n"
            "The scene has 5 small colored blocks arranged in a pentagon.\n"
            "The robot should first hover near a specified waypoint, then pick target.\n\n"
            "Rate:\n"
            "  pA = probability going for Goal A\n"
            "  pB = probability going for Goal B\n"
            "  grounding = 0.0 (ignores waypoint) to 1.0 (clearly visits waypoint)\n\n"
            'Output ONLY valid JSON: {"pA": X, "pB": X, "grounding": X, '
            '"cue": "brief description"}'
        )
    return base


# ── ENRICHED prompt (hypothesis: context + metrics explanation) ───────

def build_enriched_prompt(eval_behavior: str, target: str,
                          behavior_label: str, cfg_int: int) -> str:
    target_upper = target.upper()
    target_color = "GREEN" if target == "left" else "RED"
    target_goal = "A" if target == "left" else "B"
    other_goal = "B" if target == "left" else "A"

    has_obstacle = "safety" in behavior_label

    base = (
        f"You are a robotics expert evaluating a Franka Panda robot arm trajectory.\n"
        f"The robot is performing a block-picking task on a table with:\n"
        f"  Goal A: {GOAL_A_DESC}\n"
        f"  Goal B: {GOAL_B_DESC}\n\n"
        f"**The robot's ASSIGNED TARGET for this episode is Goal {target_goal} "
        f"= the {target_upper} ({target_color}) block.**\n"
        f"A correct trajectory should end by reaching the {target_upper} block.\n\n"
    )

    if has_obstacle:
        base += (
            "**OBSTACLE:** There is a CYAN CYLINDER on the table between the robot's "
            "start position and the blocks. The robot must avoid it.\n\n"
        )

    # CFG context
    if cfg_int > 0:
        cfg_lambda = cfg_int  # cfg02 = lambda 2.0, cfg09 = lambda 9.0
        base += (
            f"This trajectory was generated with Classifier-Free Guidance λ={cfg_lambda}.\n"
            f"Higher λ amplifies the behavior conditioning.\n\n"
        )

    base += (
        "You are watching a video of the full trajectory (~5 seconds).\n\n"
    )

    # ── Metric education section ──
    metrics_section = (
        "**METRIC DEFINITIONS** (so you understand what we measure):\n\n"
    )

    if eval_behavior == "legibility":
        metrics_section += (
            "• **L_early (Legibility)**: We compute a Bayesian posterior P(target | ee_position)\n"
            "  at each timestep in the FIRST 30% of the trajectory, then average.\n"
            "  Formula: At each step, distance from end-effector to each block → Gaussian\n"
            "  likelihood → posterior. L_early=1.0 means the arm is unambiguously closer to\n"
            "  the correct target from the very start. L_early=0.5 means equidistant (ambiguous).\n"
            "  L_early < 0.5 means the arm is actually closer to the WRONG block early on.\n\n"
            "• A LEGIBLE trajectory should have HIGH L_early because it curves toward the\n"
            "  target block early, creating spatial separation from the other block.\n\n"
            f"**YOUR TASK**: Watch this video. The robot should be going for the "
            f"{target_upper} ({target_color}) block.\n"
            f"Does the arm's early motion (first 1-2 seconds) clearly curve TOWARD the "
            f"{target_upper} block and AWAY from the {('RIGHT' if target=='left' else 'LEFT')} block?\n"
            f"Or does it go straight/ambiguously, hiding which block it wants?\n\n"
            f"Rate:\n"
            f"  pA = probability the arm LOOKS like it's going for Goal A (left)\n"
            f"  pB = probability the arm LOOKS like it's going for Goal B (right)\n"
            f"  legibility = 0.0 if intent is hidden until late, 1.0 if the arm's curve\n"
            f"    makes Goal {target_goal} obvious from the first seconds\n"
            f"  correct_target = whether pA or pB favors the assigned target Goal {target_goal}\n\n"
            f'Output ONLY valid JSON: {{"pA": X, "pB": X, "legibility": X, '
            f'"correct_target": true/false, '
            f'"cue": "describe what the arm does in first 2 seconds and which block it curves toward"}}'
        )
    elif eval_behavior == "predictability":
        metrics_section += (
            "• **Path Efficiency**: straight_line_distance / actual_path_length.\n"
            "  1.0 = perfectly straight line to target. 0.5 = path is twice as long as needed.\n\n"
            "• A PREDICTABLE trajectory should have HIGH path efficiency because it takes\n"
            "  the most direct route. No arcs, no curves, no lateral sweeps.\n\n"
            f"**YOUR TASK**: Watch this video. The robot should be going for the "
            f"{target_upper} ({target_color}) block.\n"
            f"Does the arm take a straight, direct path toward {target_upper}?\n"
            f"Or does it curve, sweep, hesitate, or take an indirect route?\n\n"
            f"Rate:\n"
            f"  pA = probability the arm LOOKS like it's going for Goal A\n"
            f"  pB = probability the arm LOOKS like it's going for Goal B\n"
            f"  predictability = 0.0 (very curved/indirect) to 1.0 (perfectly straight)\n"
            f"  correct_target = whether the arm ends up at the assigned Goal {target_goal}\n\n"
            f'Output ONLY valid JSON: {{"pA": X, "pB": X, "predictability": X, '
            f'"correct_target": true/false, '
            f'"cue": "describe path directness and any deviations"}}'
        )
    elif eval_behavior == "grounding":
        metrics_section += (
            "• **Hover Distance**: minimum 3D distance from end-effector to waypoint block.\n"
            "  < 0.06m means the arm visited the waypoint. Higher = ignored it.\n\n"
            f"**YOUR TASK**: The robot should first pass near a colored waypoint block,\n"
            f"then proceed to the {target_upper} ({target_color}) target block.\n"
            f"Does the arm detour toward a small colored block before moving to the target?\n\n"
            f"Rate:\n"
            f"  pA = probability going for Goal A\n"
            f"  pB = probability going for Goal B\n"
            f"  grounding = 0.0 (ignores waypoint) to 1.0 (clearly visits waypoint first)\n"
            f"  correct_target = whether it proceeds to assigned Goal {target_goal}\n\n"
            f'Output ONLY valid JSON: {{"pA": X, "pB": X, "grounding": X, '
            f'"correct_target": true/false, '
            f'"cue": "describe path and whether a waypoint detour is visible"}}'
        )

    return base + metrics_section


# ── Gemini API call ──────────────────────────────────────────────────

def score_video(client, video_path: Path, prompt: str,
                model: str = GEMINI_MODEL, retries: int = 3) -> dict:
    from google.genai import types as gtypes

    video_bytes = video_path.read_bytes()
    video_part = gtypes.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    text_part = gtypes.Part.from_text(text=prompt)

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[gtypes.Content(role="user",
                                         parts=[video_part, text_part])],
            )
            text = resp.text.strip()
            j_start = text.find("{")
            j_end = text.rfind("}") + 1
            if j_start == -1 or j_end == 0:
                raise ValueError(f"No JSON found: {text[:200]}")
            return json.loads(text[j_start:j_end])
        except Exception as e:
            print(f"    [retry {attempt+1}/{retries}: {e}]")
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return {"error": str(e)}
    return {"error": "all retries failed"}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Test VLM blind vs enriched scoring")
    ap.add_argument("--video_dir",
                    default=str(PROJECT / "data/demos/demo_videos_v2"))
    ap.add_argument("--model", default=GEMINI_MODEL)
    ap.add_argument("--output", default=str(PROJECT / "outputs/vlm_context_hypothesis.json"))
    ap.add_argument("--max_videos", type=int, default=0,
                    help="Limit videos to test (0 = all)")
    args = ap.parse_args()

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Quick connectivity test
    resp = client.models.generate_content(model=args.model, contents="Reply OK")
    print(f"Gemini ready: {resp.text.strip()}")

    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("*.mp4"))
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    print(f"\nFound {len(videos)} videos in {video_dir}")
    print(f"Model: {args.model}")
    print(f"{'='*70}")

    results = []

    for vi, vpath in enumerate(videos):
        meta = parse_video_name(vpath.name)
        eval_beh = get_eval_behavior(meta["behavior"])

        print(f"\n[{vi+1}/{len(videos)}] {vpath.name}")
        print(f"  behavior={meta['behavior']} target={meta['target']} "
              f"cfg={meta['cfg_int']} eval_as={eval_beh}")

        # ── A) BLIND scoring ──
        print("  (A) BLIND prompt...")
        blind_prompt = build_blind_prompt(eval_beh)
        blind_result = score_video(client, vpath, blind_prompt, args.model)
        print(f"      -> {json.dumps(blind_result, default=str)[:200]}")
        time.sleep(1)  # rate limit

        # ── B) ENRICHED scoring ──
        print("  (B) ENRICHED prompt...")
        enriched_prompt = build_enriched_prompt(
            eval_beh, meta["target"], meta["behavior"], meta["cfg_int"])
        enriched_result = score_video(client, vpath, enriched_prompt, args.model)
        print(f"      -> {json.dumps(enriched_result, default=str)[:200]}")
        time.sleep(1)  # rate limit

        # ── Collect ──
        entry = {
            "video": vpath.name,
            **meta,
            "eval_behavior": eval_beh,
            "blind": blind_result,
            "enriched": enriched_result,
        }
        results.append(entry)
        print(f"  SUMMARY: blind_score={_get_score(blind_result, eval_beh):.2f}  "
              f"enriched_score={_get_score(enriched_result, eval_beh):.2f}  "
              f"blind_target_correct={_target_correct(blind_result, meta['target'])}  "
              f"enriched_target_correct={_target_correct(enriched_result, meta['target'])}")

    # ── Save results ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Results saved: {out_path}")

    # ── Summary ──
    print_summary(results)


def _get_score(result: dict, eval_beh: str) -> float:
    key = {"legibility": "legibility", "predictability": "predictability",
           "grounding": "grounding"}.get(eval_beh, "legibility")
    return float(result.get(key, result.get("score", 0.5)))


def _target_correct(result: dict, target: str) -> bool:
    """Check if VLM's pA/pB favors the correct target."""
    if "correct_target" in result:
        return bool(result["correct_target"])
    pA = float(result.get("pA", 0.5))
    pB = float(result.get("pB", 0.5))
    if target == "left":
        return pA > pB
    else:
        return pB > pA


def print_summary(results: list):
    print(f"\n{'='*70}")
    print("SUMMARY: BLIND vs ENRICHED VLM SCORING")
    print(f"{'='*70}")

    by_behavior = {}
    for r in results:
        beh = r["eval_behavior"]
        if beh not in by_behavior:
            by_behavior[beh] = {"blind_scores": [], "enriched_scores": [],
                                "blind_correct": [], "enriched_correct": []}
        blind_s = _get_score(r["blind"], beh)
        enr_s = _get_score(r["enriched"], beh)
        blind_c = _target_correct(r["blind"], r["target"])
        enr_c = _target_correct(r["enriched"], r["target"])

        by_behavior[beh]["blind_scores"].append(blind_s)
        by_behavior[beh]["enriched_scores"].append(enr_s)
        by_behavior[beh]["blind_correct"].append(blind_c)
        by_behavior[beh]["enriched_correct"].append(enr_c)

    for beh, data in by_behavior.items():
        n = len(data["blind_scores"])
        avg_blind = sum(data["blind_scores"]) / n if n else 0
        avg_enr = sum(data["enriched_scores"]) / n if n else 0
        blind_acc = sum(data["blind_correct"]) / n if n else 0
        enr_acc = sum(data["enriched_correct"]) / n if n else 0

        print(f"\n  {beh.upper()} ({n} videos):")
        print(f"    Blind:    avg_score={avg_blind:.3f}  target_correct={blind_acc:.0%}")
        print(f"    Enriched: avg_score={avg_enr:.3f}  target_correct={enr_acc:.0%}")
        print(f"    Delta score = {avg_enr - avg_blind:+.3f}  "
              f"Delta accuracy = {enr_acc - blind_acc:+.0%}")

    # Overall
    all_blind_correct = sum(1 for r in results
                            if _target_correct(r["blind"], r["target"]))
    all_enr_correct = sum(1 for r in results
                          if _target_correct(r["enriched"], r["target"]))
    n_total = len(results)
    print(f"\n  OVERALL ({n_total} videos):")
    print(f"    Blind target accuracy:    {all_blind_correct}/{n_total} "
          f"= {all_blind_correct/n_total:.0%}")
    print(f"    Enriched target accuracy: {all_enr_correct}/{n_total} "
          f"= {all_enr_correct/n_total:.0%}")
    print(f"    Delta target accuracy: {(all_enr_correct-all_blind_correct)/n_total:+.0%}")

    print(f"\n  KEY QUESTION: Does 'enriched' show higher target accuracy?")
    print(f"  If yes → VLM needs context to be useful for steering.")
    print(f"  If no  → VLM visual understanding alone is insufficient.\n")


if __name__ == "__main__":
    main()
