"""
Trajectory Visualization for VLM Input
Renders predicted actions on environment frames for legibility assessment
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import io
from PIL import Image, ImageDraw, ImageFont


class TrajectoryVisualizer:
    """Visualize predicted action sequences for VLM evaluation"""
    
    def __init__(self, img_size: Tuple[int, int] = (480, 480)):
        """
        Args:
            img_size: Output image size (width, height)
        """
        self.img_size = img_size
        
    def render_frame_with_trajectory(
        self,
        env,
        obs: np.ndarray,
        action_sequence: np.ndarray,
        n_steps: int = 8,
        show_future: bool = True
    ) -> bytes:
        """
        Render current environment state with predicted trajectory overlay
        
        Args:
            env: TwoBlockPick environment
            obs: Current observation (ee_pos, ee_quat, grip, block_poses)
            action_sequence: Predicted actions (H, act_dim)
            n_steps: Number of future steps to visualize
            show_future: Whether to show ghosted future poses
            
        Returns:
            JPEG image bytes
        """
        # Get current environment render
        img = env.render(mode='rgb_array')  # (H, W, 3)
        
        if show_future and n_steps > 0:
            # Parse observation to get current EE and block positions
            ee_pos = obs[:3]  # (x, y, z)
            left_block_pos = obs[9:12]  # Block L position
            right_block_pos = obs[16:19]  # Block R position
            
            # Extract position deltas from actions (first 3 components)
            position_deltas = action_sequence[:n_steps, :3]  # (n_steps, 3)
            
            # Visualize trajectory as arrows/path
            img = self._overlay_trajectory(
                img, 
                ee_pos, 
                position_deltas,
                left_block_pos,
                right_block_pos
            )
        
        # Add text overlay with step info
        img = self._add_text_overlay(img, f"Predicted Next {n_steps} Steps")
        
        # Convert to JPEG bytes
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize(self.img_size, Image.Resampling.LANCZOS)
        
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG', quality=90)
        return img_bytes.getvalue()
    
    def render_trajectory_comparison(
        self,
        env,
        obs: np.ndarray,
        candidates: List[np.ndarray],
        scores: Optional[List[float]] = None
    ) -> List[bytes]:
        """
        Render multiple candidate trajectories for comparison
        
        Args:
            env: Environment
            obs: Current observation
            candidates: List of action sequences
            scores: Optional legibility scores for each candidate
            
        Returns:
            List of JPEG image bytes, one per candidate
        """
        images = []
        
        for i, action_seq in enumerate(candidates):
            # Render this candidate
            img_bytes = self.render_frame_with_trajectory(
                env, obs, action_seq, n_steps=8, show_future=True
            )
            
            # If scores provided, add them
            if scores is not None and i < len(scores):
                img = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img)
                
                # Add score text
                score_text = f"Candidate {i+1} | Legibility: {scores[i]:.2f}"
                img_array = self._add_text_overlay(
                    img_array, 
                    score_text,
                    position='bottom'
                )
                
                # Re-encode
                img_pil = Image.fromarray(img_array)
                img_bytes_new = io.BytesIO()
                img_pil.save(img_bytes_new, format='JPEG', quality=90)
                img_bytes = img_bytes_new.getvalue()
            
            images.append(img_bytes)
        
        return images
    
    def _overlay_trajectory(
        self,
        img: np.ndarray,
        ee_pos: np.ndarray,
        position_deltas: np.ndarray,
        left_block_pos: np.ndarray,
        right_block_pos: np.ndarray
    ) -> np.ndarray:
        """
        Overlay trajectory visualization on image
        
        Uses proper 3D-to-2D projection matching TwoBlockPick camera setup:
        - Angled view: yaw=135°, pitch=-30°
        - Camera target: [0.50, 0.0, table_z]
        - Distance: 0.9
        """
        import pybullet as p
        
        img = img.copy()
        h, w = img.shape[:2]
        
        # Match TwoBlockPick environment camera setup (from twoblockpick_env.py line 437)
        _TABLE_TOP_Z = 0.625  # Typical table height
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.50, 0.0, _TABLE_TOP_Z],
            distance=0.9,
            yaw=135,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=w / h,
            nearVal=0.1,
            farVal=3.0
        )
        
        def world_to_pixel(pos_3d):
            """Project 3D world coordinates to 2D pixel coordinates"""
            # Convert to homogeneous coordinates
            pos_4d = [pos_3d[0], pos_3d[1], pos_3d[2], 1.0]
            
            # Apply view matrix (world -> camera space)
            view_mat = np.array(view_matrix).reshape(4, 4).T
            cam_space = view_mat @ pos_4d
            
            # Apply projection matrix (camera -> clip space)
            proj_mat = np.array(proj_matrix).reshape(4, 4).T
            clip_space = proj_mat @ cam_space
            
            # Perspective divide (clip -> NDC)
            if abs(clip_space[3]) < 1e-6:
                return None  # Behind camera
            ndc = clip_space[:3] / clip_space[3]
            
            # NDC (-1 to 1) -> pixel coordinates (0 to w/h)
            x_px = int((ndc[0] + 1.0) * 0.5 * w)
            y_px = int((1.0 - ndc[1]) * 0.5 * h)  # Flip Y (OpenGL convention)
            
            # Clamp to image bounds
            x_px = max(0, min(w - 1, x_px))
            y_px = max(0, min(h - 1, y_px))
            
            return (x_px, y_px)
        
        # Draw block positions with HIGH VISIBILITY
        left_px = world_to_pixel(left_block_pos)
        right_px = world_to_pixel(right_block_pos)
        
        cv2.circle(img, left_px, 25, (0, 255, 0), -1)  # Green for left (bigger)
        cv2.circle(img, left_px, 25, (255, 255, 255), 3)  # White border
        cv2.circle(img, right_px, 25, (0, 0, 255), -1)  # Red for right (bigger)
        cv2.circle(img, right_px, 25, (255, 255, 255), 3)  # White border
        
        cv2.putText(img, "L", (left_px[0]-12, left_px[1]+12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, "R", (right_px[0]-12, right_px[1]+12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Calculate trajectory metrics for annotation
        total_dy = np.sum(position_deltas[:, 1])  # Y-axis displacement
        arc_magnitude = np.sqrt(np.sum(position_deltas[:, :2]**2, axis=1)).sum()
        
        # Classify curve for annotation
        if arc_magnitude < 0.05:
            curve_type = "STRAIGHT"
        elif arc_magnitude < 0.15:
            curve_type = "MODERATE CURVE"
        else:
            curve_type = "LARGE CURVE"
        
        # Draw trajectory path with HIGH CONTRAST
        current_pos = ee_pos.copy()
        prev_px = world_to_pixel(current_pos)
        
        # Use CYAN for maximum visibility against typical backgrounds
        trajectory_color = (0, 255, 255)  # Cyan (BGR format)
        
        for i, delta in enumerate(position_deltas):
            current_pos = current_pos + delta
            curr_px = world_to_pixel(current_pos)
            
            # Draw THICK arrow - constant thickness, no fading
            thickness = 8  # Much thicker than before (was 4→1)
            cv2.arrowedLine(img, prev_px, curr_px, trajectory_color, 
                          thickness, tipLength=0.4, line_type=cv2.LINE_AA)
            
            # Draw position markers with BRIGHT YELLOW
            cv2.circle(img, curr_px, 10, (0, 255, 255), -1)  # Cyan filled
            cv2.circle(img, curr_px, 10, (255, 255, 255), 2)  # White border
            
            prev_px = curr_px
        
        # Draw current EE position with MAGENTA
        ee_px = world_to_pixel(ee_pos)
        cv2.circle(img, ee_px, 18, (255, 0, 255), -1)  # Magenta for EE (bigger)
        cv2.circle(img, ee_px, 18, (255, 255, 255), 3)  # Thick white border
        cv2.putText(img, "START", (ee_px[0]-30, ee_px[1]-25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # NO TEXT ANNOTATIONS - Force VLM to do true visual analysis
        # The VLM should discriminate based on trajectory curve visibility alone
        
        return img
    
    def _add_text_overlay(
        self, 
        img: np.ndarray, 
        text: str,
        position: str = 'top'
    ) -> np.ndarray:
        """Add text overlay to image"""
        img = img.copy()
        h, w = img.shape[:2]
        
        # Add semi-transparent background
        overlay = img.copy()
        if position == 'top':
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
            text_pos = (10, 28)
        else:  # bottom
            cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
            text_pos = (10, h-12)
        
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        # Add text
        cv2.putText(
            img, text, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return img
    
    def create_trajectory_grid(
        self,
        images: List[bytes],
        scores: Optional[List[float]] = None,
        grid_cols: int = 2
    ) -> bytes:
        """
        Create a grid visualization of multiple trajectories
        
        Args:
            images: List of JPEG image bytes
            scores: Optional scores to display
            grid_cols: Number of columns in grid
            
        Returns:
            JPEG bytes of grid image
        """
        if not images:
            raise ValueError("No images provided")
        
        # Load images
        pil_images = [Image.open(io.BytesIO(img)) for img in images]
        
        # Calculate grid dimensions
        n_images = len(pil_images)
        grid_rows = (n_images + grid_cols - 1) // grid_cols
        
        # Get individual image size
        img_w, img_h = pil_images[0].size
        
        # Create grid canvas
        grid_w = img_w * grid_cols
        grid_h = img_h * grid_rows
        grid = Image.new('RGB', (grid_w, grid_h), color='black')
        
        # Paste images
        for idx, img in enumerate(pil_images):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * img_w
            y = row * img_h
            grid.paste(img, (x, y))
        
        # Convert to bytes
        grid_bytes = io.BytesIO()
        grid.save(grid_bytes, format='JPEG', quality=92)
        return grid_bytes.getvalue()


if __name__ == "__main__":
    # Test visualization
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from envs.twoblockpick_env import TwoBlockPickEnv
    
    # Create environment
    env = TwoBlockPickEnv(render_mode='rgb_array')
    obs, _ = env.reset()
    
    # Create dummy action sequence
    action_seq = np.random.randn(16, 5) * 0.01
    
    # Visualize
    visualizer = TrajectoryVisualizer()
    img_bytes = visualizer.render_frame_with_trajectory(
        env, obs, action_seq, n_steps=8
    )
    
    # Save test image
    with open('test_trajectory_viz.jpg', 'wb') as f:
        f.write(img_bytes)
    
    print(f"Saved test visualization (size: {len(img_bytes)} bytes)")
    
    env.close()
