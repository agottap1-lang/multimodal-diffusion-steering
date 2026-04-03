"""
VLM Client for Legibility Evaluation
Uses existing gemini_vlm_eval pipeline for real-time trajectory scoring
"""

import os
import sys
import logging
from typing import Optional, List, Union, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load API key from gemini_vlm_eval .env file
GEMINI_VLM_PATH = Path(os.getenv(
    "GEMINI_VLM_EVAL_PATH",
    r"C:\Users\anude\OneDrive\Documents\gemini_vlm_eval"
))
ENV_FILE = GEMINI_VLM_PATH / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    logging.info(f"Loaded API key from {ENV_FILE}")
else:
    logging.warning(f"No .env file found at {ENV_FILE}")

# Add gemini_vlm_eval to path
SRC_PATH = GEMINI_VLM_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from gemini_vlm_eval.client import GeminiClient
    from gemini_vlm_eval.schema import ManifestEntry, EvaluationResult
except ImportError as e:
    raise ImportError(
        f"Failed to import gemini_vlm_eval. Make sure the package is available at {SRC_PATH}\n"
        f"Error: {e}"
    )

logger = logging.getLogger(__name__)


class LegibilityScorer:
    """
    Wrapper around existing GeminiClient for trajectory legibility scoring
    Uses your production VLM pipeline from gemini_vlm_eval
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-2.5-flash"
    ):
        """
        Initialize using existing gemini_vlm_eval client
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Gemini model name (default: gemini-2.5-flash)
        """
        # Use existing GeminiClient
        self.client = GeminiClient(api_key=api_key, model=model)
        self.model_name = model
        self.api_key = self.client.api_key
        
        logger.info(f"Initialized LegibilityScorer using gemini_vlm_eval pipeline (model: {model})")
    
    def score_trajectory(
        self,
        image_bytes: Union[bytes, List[bytes]],
        goal_A: str,
        goal_B: str,
        mode: str = "single_frame",
        video_id: str = "rollout",
        t_sec: float = 0.0,
        target_goal: Optional[str] = None,
    ) -> Dict:
        """
        Score a trajectory's legibility using existing VLM pipeline
        
        Args:
            image_bytes: Single frame or list of frames (prefix_frames)
            goal_A: First goal description (e.g., "pick left block")
            goal_B: Second goal description (e.g., "pick right block")
            mode: "single_frame" or "prefix_frames"
            video_id: Identifier for this trajectory
            t_sec: Timestamp in seconds
            target_goal: "A" or "B" for direction-aware scoring
            
        Returns:
            Dict with keys: pA, pB, legibility_score, confidence, cue, choice
        """
        # Create ManifestEntry for the trajectory
        manifest_entry = ManifestEntry(
            video_id=video_id,
            video_path=f"trajectory_{video_id}",
            goal_gt="A",  # Dummy value, not used for scoring
            goal_A=goal_A,
            goal_B=goal_B,
            scene_id="twoblockpick",
            task_family="block_pick",
            traj_type="predicted",
            notes="Real-time trajectory evaluation"
        )
        
        try:
            # Use existing GeminiClient.evaluate_frame method
            result: EvaluationResult = self.client.evaluate_frame(
                image_bytes=image_bytes,
                manifest_entry=manifest_entry,
                t_sec=int(t_sec),
                frame_idx=0,
                mode=mode
            )
            
            # Convert EvaluationResult to our simplified dict format
            # Direction-aware legibility: P(correct goal | observation)
            if target_goal == "A":
                directed_leg = result.pA
            elif target_goal == "B":
                directed_leg = result.pB
            else:
                directed_leg = max(result.pA, result.pB)  # fallback: undirected

            return {
                'video_id': result.video_id,
                't_sec': result.t_sec,
                'mode': mode,
                'pA': result.pA,
                'pB': result.pB,
                'clarity': getattr(result, 'clarity', max(result.pA, result.pB)),
                'legibility_score': directed_leg,
                'undirected_legibility': max(result.pA, result.pB),
                'target_goal': target_goal,
                'confidence': result.confidence,
                'choice': result.choice,
                'cue': result.cue,
                'legible': result.legible,
                'vlm_error': False,
                'legibility_class': 'legible' if directed_leg >= 0.7 else
                                   'somewhat_legible' if directed_leg >= 0.55 else
                                   'not_legible_yet',
                'latency_ms': result.latency_ms,
                'model': result.model
            }
            
        except Exception as e:
            logger.error(f"VLM evaluation failed for {video_id} at t={t_sec}s: {e}")
            return self._default_response(video_id, t_sec, mode)
    
    def score_trajectory_batch(
        self,
        image_bytes_list: List[Union[bytes, List[bytes]]],
        goal_A: str,
        goal_B: str,
        mode: str = "single_frame"
    ) -> List[Dict]:
        """
        Score multiple trajectory candidates for reranking
        
        Args:
            image_bytes_list: List of image sequences (one per candidate)
            goal_A: First goal description
            goal_B: Second goal description
            mode: Evaluation mode
            
        Returns:
            List of scoring dicts, one per candidate
        """
        results = []
        for i, img_bytes in enumerate(image_bytes_list):
            result = self.score_trajectory(
                img_bytes, 
                goal_A, 
                goal_B, 
                mode=mode,
                video_id=f"candidate_{i}",
                t_sec=0.0
            )
            results.append(result)
        
        return results
    
    def score_trajectory_progressive(
        self,
        image_bytes_early: bytes,
        image_bytes_full: bytes,
        goal_A: str,
        goal_B: str,
        video_id: str = "rollout",
        early_weight: float = 0.6
    ) -> Dict:
        """
        Progressive legibility scoring (research-backed: Dragan et al. 2015)
        
        Key insight: Legibility is determined by EARLY trajectory clarity (first 30-40%).
        This method queries VLM at both early (30%) and full (100%) trajectory,
        then combines scores with higher weight on early clarity.
        
        Args:
            image_bytes_early: Visualization of first 30% of trajectory
            image_bytes_full: Visualization of full trajectory
            goal_A: First goal description
            goal_B: Second goal description
            video_id: Trajectory identifier
            early_weight: Weight for early score (0.6 = 60% early, 40% final)
            
        Returns:
            Dict with:
                - early_confidence: Confidence at 30% (most important!)
                - final_confidence: Confidence at 100%
                - legibility_score: Weighted combination (higher = more legible)
                - consistent: Whether goal inference is same at both points
                - goal: Most likely goal
        """
        try:
            # Score early trajectory (30%)
            early_result = self.score_trajectory(
                image_bytes_early,
                goal_A=goal_A,
                goal_B=goal_B,
                mode="single_frame",
                video_id=video_id,
                t_sec=0.3
            )
            
            # Score full trajectory (100%)
            full_result = self.score_trajectory(
                image_bytes_full,
                goal_A=goal_A,
                goal_B=goal_B,
                mode="single_frame",
                video_id=video_id,
                t_sec=1.0
            )
            
            # Extract confidences
            early_conf = max(early_result['pA'], early_result['pB'])
            final_conf = max(full_result['pA'], full_result['pB'])
            
            # Determine goals
            early_goal = 'A' if early_result['pA'] > early_result['pB'] else 'B'
            final_goal = 'A' if full_result['pA'] > full_result['pB'] else 'B'
            
            # Weighted legibility score (early clarity weighted higher)
            legibility_score = early_weight * early_conf + (1 - early_weight) * final_conf
            
            # Check consistency (same goal at 30% and 100%?)
            consistent = (early_goal == final_goal)
            
            # Bonus for consistency (legible trajectories should be consistent)
            if consistent:
                legibility_score *= 1.05  # 5% bonus
            
            return {
                'video_id': video_id,
                'early_confidence': early_conf,
                'final_confidence': final_conf,
                'legibility_score': legibility_score,
                'consistent': consistent,
                'early_goal': early_goal,
                'final_goal': final_goal,
                'goal': final_goal,
                'pA': full_result['pA'],
                'pB': full_result['pB'],
                'latency_ms': early_result['latency_ms'] + full_result['latency_ms'],
                'model': self.model_name,
                'method': 'progressive',
                'early_weight': early_weight
            }
            
        except Exception as e:
            logger.error(f"Progressive scoring failed for {video_id}: {e}")
            # Fallback to default
            return {
                'video_id': video_id,
                'early_confidence': 0.5,
                'final_confidence': 0.5,
                'legibility_score': 0.5,
                'consistent': False,
                'goal': 'unknown',
                'latency_ms': 0,
                'method': 'progressive',
                'error': str(e)
            }
    
    def score_trajectory_progressive_batch(
        self,
        early_full_pairs: List[tuple],  # [(early_bytes, full_bytes), ...]
        goal_A: str,
        goal_B: str,
        early_weight: float = 0.6
    ) -> List[Dict]:
        """
        Batch version of progressive scoring for multiple candidates
        
        Args:
            early_full_pairs: List of (early_image_bytes, full_image_bytes) tuples
            goal_A: First goal description
            goal_B: Second goal description
            early_weight: Weight for early score
            
        Returns:
            List of progressive scoring dicts
        """
        results = []
        for i, (early_bytes, full_bytes) in enumerate(early_full_pairs):
            result = self.score_trajectory_progressive(
                early_bytes,
                full_bytes,
                goal_A=goal_A,
                goal_B=goal_B,
                video_id=f"candidate_{i}",
                early_weight=early_weight
            )
            results.append(result)
        
        return results
    
    def _default_response(self, video_id: str, t_sec: float, mode: str) -> Dict:
        """Return default response when VLM call fails."""
        return {
            'video_id': video_id,
            't_sec': t_sec,
            'mode': mode,
            'pA': 0.5,
            'pB': 0.5,
            'legibility_score': 0.5,
            'undirected_legibility': 0.5,
            'target_goal': None,
            'confidence': 0,
            'choice': 'C',
            'cue': 'ERROR: VLM call failed',
            'legible': 'unknown',
            'vlm_error': True,
            'legibility_class': 'unknown',
            'latency_ms': 0,
            'model': self.model_name
        }


if __name__ == "__main__":
    # Test the client
    import PIL.Image
    import io
    import json
    
    print("Testing LegibilityScorer with existing gemini_vlm_eval pipeline...\n")
    
    # Create a dummy image
    img = PIL.Image.new('RGB', (96, 96), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    
    try:
        # Test scoring
        scorer = LegibilityScorer()
        print(f"✓ Initialized scorer with model: {scorer.model_name}\n")
        
        # Note: This would make an actual API call
        print("To test VLM call, run with --test-api flag")
        print("Example: python scripts/vlm_client.py --test-api")
        
        # result = scorer.score_trajectory(
        #     img_bytes,
        #     goal_A="pick the left block",
        #     goal_B="pick the right block",
        #     mode="single_frame"
        # )
        # print("Test Result:")
        # print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
