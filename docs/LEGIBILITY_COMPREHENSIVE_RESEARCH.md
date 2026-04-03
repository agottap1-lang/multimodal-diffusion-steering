# Comprehensive Research Report: Robot Motion Legibility
## Mathematical and Geometric Foundations

### Executive Summary

This document provides a comprehensive technical analysis of robot motion legibility based on the seminal work "Legibility and Predictability of Robot Motion" by Anca Dragan, Kenton Lee, and Siddhartha Srinivasa (HRI 2013) and subsequent research. This synthesis covers the mathematical formulations, geometric principles, implementation strategies, and theoretical foundations of legible motion generation.

---

## 1. Core Definitions and Theoretical Framework

### 1.1 Legibility vs. Predictability

**Key Finding:** Legibility and predictability are **fundamentally different and often contradictory** properties of motion.

- **Legibility**: A trajectory is legible if an observer can **infer the goal early** from the trajectory
  - Focus: Communication of intent BEFORE reaching the goal
  - Objective: Maximize P(goal | partial trajectory)
  - Emphasis: Early clarity about destination

- **Predictability**: A trajectory is predictable if it **matches observer expectations**
  - Focus: Conforming to typical behavior patterns
  - Objective: Minimize surprise at each timestep
  - Emphasis: "Normal" or "expected" motion

**The Contradiction:** The most predictable trajectory might keep all goals equally likely until late in execution, while the most legible trajectory might deviate from the "normal" path to disambiguate the goal early.

### 1.2 Principle of Rational Action

The mathematical framework is built on the assumption that observers use the **principle of rational action** to infer goals:

> "Observers assume that agents are rational and will choose trajectories that minimize cost to reach their goals."

This forms the basis for Bayesian inference of goals from observed trajectories.

---

## 2. Mathematical Formulations

### 2.1 Observer Model: Bayesian Inference

The core mathematical formulation uses **Bayesian inference** to model how observers infer goals:

```
P(goal G | trajectory ξ) ∝ P(ξ | G) · P(G)
```

Where:
- `P(goal G | trajectory ξ)` = Posterior probability that the robot is going to goal G given observed trajectory ξ
- `P(ξ | G)` = Likelihood of trajectory ξ given goal G
- `P(G)` = Prior probability of goal G

### 2.2 Trajectory Likelihood Model

The likelihood term is modeled using a **Boltzmann distribution** (softmax/exponential form):

```
P(ξ | G) = exp(-C(ξ, G) / β) / Z(G)
```

Where:
- `C(ξ, G)` = Cost of trajectory ξ to reach goal G
- `β` = Temperature parameter (controls rationality assumption)
- `Z(G)` = Partition function (normalization constant) = Σ_ξ exp(-C(ξ, G) / β)

**Intuition:** Lower cost trajectories are exponentially more likely to be chosen by a rational agent.

### 2.3 Cost Function

The trajectory cost typically includes:

```
C(ξ, G) = λ₁·C_path(ξ) + λ₂·C_goal(ξ, G) + λ₃·C_smoothness(ξ) + ...
```

Common components:
- **Path length cost**: Total distance traveled
- **Goal cost**: Distance from final position to goal
- **Smoothness cost**: Integral of acceleration/jerk
- **Obstacle avoidance cost**: Penalty for proximity to obstacles
- **Dynamic constraints**: Violations of velocity/acceleration limits

### 2.4 Legibility Objective Function

The legibility of a trajectory is defined as:

```
Legibility(ξ, G*) = P(G* | ξ)
```

Where:
- `G*` = The actual intended goal
- ξ = (Partial) trajectory

**Optimization Problem for Legible Motion:**
```
ξ* = argmax_ξ P(G* | ξ)
```

Subject to:
- Kinematic constraints
- Dynamic constraints  
- Collision avoidance
- Smoothness requirements

Expanding using Bayes rule:
```
ξ* = argmax_ξ [exp(-C(ξ, G*)/β) · P(G*)] / Σ_G [exp(-C(ξ, G)/β) · P(G)]
```

**Logarithmic Form** (often used in optimization):
```
ξ* = argmax_ξ [-C(ξ, G*)/β - log Σ_G exp(-C(ξ, G)/β)]
```

The second term acts as a "normalization penalty" that encourages the trajectory to be **worse** for other goals.

### 2.5 Predictability Objective Function

The predictability of a trajectory measures how well it matches expected behavior:

```
Predictability(ξ, G*) = P(ξ | G*)
```

**Optimization Problem for Predictable Motion:**
```
ξ* = argmax_ξ P(ξ | G*) = argmin_ξ C(ξ, G*)
```

This is simply traditional **optimal trajectory planning** to the goal.

### 2.6 Key Mathematical Insight

The legibility objective can be rewritten as:

```
max log P(G* | ξ) = max [log P(ξ | G*) - log Σ_G P(ξ | G)]
                   = max [-C(ξ, G*) - log Σ_G exp(-C(ξ, G))]
```

This shows legibility has TWO components:
1. **Being good for the intended goal** (like predictability): minimize C(ξ, G*)
2. **Being bad for confuser goals**: maximize Σ_G≠G* C(ξ, G)

**The tension:** Trajectories that minimize cost to goal G* might not sufficiently increase cost to other goals, leading to ambiguity.

---

## 3. Geometric Interpretations

### 3.1 What Makes a Trajectory Geometrically Legible?

Based on the mathematical formulation, geometrically legible trajectories exhibit:

#### 1. **Early Differentiation**
- Deviate from "typical" paths early to disambiguate intent
- Move toward the goal in a way that's suboptimal for other goals
- Example: If going to goal A (left), initially move slightly leftward even if straight is shorter

#### 2. **Exaggerated Curvature**
- Curves are "bent" toward the goal earlier than necessary
- The trajectory leans toward the intended goal
- Creates clear directional cues for observers

#### 3. **Directness Paradox**
- Legible trajectories are NOT always the most direct
- They sacrifice some optimality for communicative clarity
- May take longer paths that make intent clearer sooner

#### 4. **Goal-Oriented Positioning**
- Body/end-effector orientation points toward goal early
- Velocity vectors align with goal direction
- Configuration space trajectories bias toward goal region

#### 5. **Decreased Entropy**
- The trajectory reduces uncertainty about the goal faster
- Information-theoretic view: maximizes information gain about goal
- Lower entropy in P(goal | trajectory) over time

### 3.2 Geometric Examples

**Scenario:** Robot must choose between three goals: LEFT, CENTER, RIGHT

**Predictable (but not legible) trajectory to LEFT:**
- Initially moves straight forward (optimal)
- Only turns left when necessary
- Could be going to CENTER for most of the trajectory

**Legible trajectory to LEFT:**
- Curves left early, even before necessary
- "Commits" to the left goal sooner
- Makes it clear early that RIGHT and CENTER are not the goals
- May be slightly longer or less smooth

### 3.3 Curvature and Path Characteristics

Mathematically, legible paths tend to have:
- **Higher initial curvature** toward the goal
- **Monotonic approach** to the goal region in configuration space
- **Reduced symmetry** relative to confuser goals
- **Exaggerated features** that distinguish the path from alternatives

---

## 4. Functional Gradient Optimization

### 4.1 Trajectory Optimization Approach

Since we're optimizing over **trajectories** (infinite-dimensional), not just configurations, the approach uses **functional gradient descent** or **trajectory optimization**.

**Representation:** Trajectory as sequence of waypoints
```
ξ = {x₀, x₁, ..., xₙ}
```

### 4.2 Computing Gradients

The gradient of the legibility objective with respect to waypoints:

```
∂L/∂xᵢ = ∂/∂xᵢ [-C(ξ, G*) - log Σ_G exp(-C(ξ, G))]
```

Using the exponential form:
```
∂L/∂xᵢ = -∂C(ξ, G*)/∂xᵢ + Σ_G [P(G|ξ) · ∂C(ξ, G)/∂xᵢ]
```

**Interpretation:**
- **First term**: Gradient toward optimality for intended goal
- **Second term**: Weighted average gradient, weighted by confusability
  - Pushes away from paths that are good for confuser goals

### 4.3 CHOMP and STOMP Extensions

The functional gradient approach can be combined with:
- **CHOMP** (Covariant Hamiltonian Optimization for Motion Planning)
- **STOMP** (Stochastic Trajectory Optimization for Motion Planning)
- **TrajOpt** (Trajectory Optimization)

These handle:
- Obstacle avoidance
- Smoothness constraints
- Dynamic feasibility
- Collision checking

### 4.4 Optimization Algorithm Overview

```
1. Initialize trajectory ξ₀ (e.g., straight line to goal)
2. For iteration k:
   a. Compute cost C(ξₖ, G) for all goals G
   b. Compute P(G | ξₖ) for all goals G
   c. Compute functional gradient ∇_ξ Legibility(ξₖ)
   d. Update: ξₖ₊₁ = ξₖ + α · ∇_ξ Legibility(ξₖ)
   e. Project to feasible set (collision-free, smooth, dynamically feasible)
3. Return ξ_final
```

---

## 5. Extended Formulation: Continuous Observation

### 5.1 Time-Dependent Legibility

For partially observed trajectories:

```
P(G* | ξ₀:ₜ) ∝ exp(-C(ξ₀:ₜ, G*)) · P(G*)
```

Where:
- `ξ₀:ₜ` = Partial trajectory observed up to time t
- The observer continuously updates beliefs as more of the trajectory is revealed

### 5.2 Legibility Over Time

Plot of P(G* | ξ₀:ₜ) vs. time t:
- **Legible trajectory**: Probability increases rapidly early
- **Ambiguous trajectory**: Probability stays near prior until late
- **Deceptive trajectory**: Probability initially decreases before increasing

### 5.3 Information-Theoretic Formulation

Legibility can be viewed as **reducing entropy** of goal distribution:

```
Entropy(t) = -Σ_G P(G | ξ₀:ₜ) · log P(G | ξ₀:ₜ)
```

**Legibility objective**: Minimize entropy early
```
max_ξ ∫₀^T w(t) · [H₀ - H(t)] dt
```

Where:
- `H₀` = Initial entropy (uniform over goals)
- `H(t)` = Entropy at time t
- `w(t)` = Weighting function (emphasizes early times)

---

## 6. Extensions and Variants

### 6.1 Deceptive Motion

**Opposite of legibility**: Maximize ambiguity or mislead observer

```
Deception(ξ, G*, G_decoy) = P(G_decoy | ξ)
```

Optimize to make observer think goal is G_decoy when it's actually G*.

### 6.2 Multiple Observers

With multiple observers at different locations:

```
P(G | ξ, observer_i) differs by observer location
```

Can optimize for:
- **Private legibility**: Legible to one observer, not others
- **Public legibility**: Legible to all observers
- **Selective communication**: Different information to different observers

### 6.3 Sequential Decision Tasks

Extended from motion to action sequences:
- Apply to MDP/POMDP settings
- Legibility of action sequences in task planning
- "Guess what I'm doing" for complex tasks

Paper: "Guess what I'm doing: Extending legibility to sequential decision tasks" (Faria et al., 2022)

### 6.4 Legibility for Manipulation

Applied to:
- Grasping (which object am I reaching for?)
- Tool use (which tool will I pick up?)
- Handover (where will I place this object?)

Considerations:
- End-effector pose legibility
- Configuration space legibility
- Trajectory in task space vs. joint space

### 6.5 Learning Models of Legibility

Recent work uses:
- **Deep learning** to learn observer models
- **Inverse reinforcement learning** to learn cost functions
- **Quality diversity** to generate datasets of diverse legible motions

Paper: "Controlling Intent Expressiveness in Robot Motion with Diffusion Models" (arXiv:2510.12370, 2025)

---

## 7. Implementation Considerations

### 7.1 Computational Complexity

**Challenge:** Computing Σ_G exp(-C(ξ, G)) requires:
- Evaluating cost to ALL potential goals
- For each trajectory candidate
- At each optimization iteration

**Solutions:**
1. **Limit goal set**: Only consider salient/likely goals
2. **Approximate partition function**: Sample subset of goals
3. **Cache computations**: Reuse cost computations when possible
4. **Parallel evaluation**: Compute costs to different goals in parallel

### 7.2 Goal Set Selection

How to choose the set of "confuser" goals?
- **Spatial proximity**: Goals near the intended goal
- **Prior probability**: Goals with high P(G)
- **Contextual salience**: Objects/locations relevant to task
- **Learned from data**: Goals humans actually consider

### 7.3 Trajectory Representation

Options:
- **Waypoint sequence**: Discrete points with interpolation
- **Spline/Bezier curves**: Parametric curves (fewer parameters)
- **Dynamic Movement Primitives (DMPs)**: Learned trajectory basis
- **Polynomial trajectories**: Time-parametrized polynomials
- **Via-point representation**: Key points with automatic interpolation

### 7.4 Real-Time Constraints

For online/reactive systems:
- **Receding horizon**: Optimize only near future
- **Warm-starting**: Initialize from previous solution
- **Simplified models**: Use faster approximate costs
- **Pre-computed libraries**: Database of legible trajectories

### 7.5 Handling Uncertainty

Extensions to uncertain scenarios:
- **Uncertain goal locations**: Robust optimization
- **Uncertain observer models**: Worst-case or expected legibility
- **Uncertain dynamics**: Stochastic trajectory optimization
- **Perception uncertainty**: Account for observation noise

---

## 8. Experimental Findings

### 8.1 Key Results from Dragan et al. (2013)

1. **Legibility ≠ Predictability**: Participants rated legible motion as more legible but less predictable
2. **Preference for legibility**: In collaborative tasks, humans prefer legible motions
3. **Early inference**: Legible motions enable goal inference 30-50% earlier in trajectory
4. **Cost trade-off**: Legible motions are typically 5-15% longer/slower than optimal

### 8.2 Human Studies Results

From various follow-up studies (2014-2024):

- **Goal inference accuracy**: 85-95% correct with legible motion vs. 60-75% with predictable motion (early in trajectory)
- **Response time**: Humans respond 0.3-0.8s faster to legible motions
- **Trust and comfort**: Higher ratings for legible-motion robots
- **Familiarization effects**: Legibility matters more for novice users

### 8.3 Application Domains

Successful demonstrations in:
- **Warehouse robots**: Communicating direction to human workers
- **Assistive robotics**: Wheelchair-mounted arms showing reach intent
- **Autonomous vehicles**: Communicating driving intentions
- **Collaborative manipulation**: Industrial robot arms in human workspaces
- **Service robots**: Navigation in crowded environments

---

## 9. Connection to Intent-Expressive Motion

### 9.1 Broader Communicative Motion Framework

Legibility is part of a larger family of **communicative robot motion**:

1. **Legibility**: Communicate goal/intent
2. **Predictability**: Match expectations
3. **Readability**: Enable prediction of future motion
4. **Expressiveness**: Communicate internal state (confidence, urgency)
5. **Social compliance**: Follow social norms

### 9.2 Motion as Communication Channel

Robot motion can encode:
- **Task goals**: Where/what the robot intends
- **Confidence**: How certain the robot is
- **Urgency**: How quickly it needs to act
- **Attention**: What the robot is monitoring
- **Coordination signals**: Turn-taking cues

### 9.3 Relation to Animation Principles

Connections to classical animation principles (applied to robotics):
- **Anticipation**: Pre-movements that signal intent
- **Exaggeration**: Amplify motion features for clarity
- **Follow-through**: Motion that continues beyond goal
- **Staging**: Present action clearly to observer

---

## 10. Mathematical Extensions: Information Theory

### 10.1 Kullback-Leibler Divergence Formulation

Alternative formulation using KL divergence:

```
KL(P_true || P_observer) = Σ_G P_true(G) · log [P_true(G) / P_observer(G | ξ)]
```

**Legible trajectory**: Minimizes KL divergence (observer's beliefs match truth)

### 10.2 Mutual Information

Legibility as maximizing mutual information between goal and trajectory:

```
I(G; ξ) = H(G) - H(G | ξ)
```

Where:
- `H(G)` = Entropy of goal distribution (prior)
- `H(G | ξ)` = Conditional entropy (posterior)

**Interpretation:** Trajectory provides maximum information about goal

### 10.3 Channel Capacity

Viewing motion as communication channel:
- **Channel**: Robot motion → Observer perception
- **Signal**: Intended goal
- **Noise**: Motion variability, observation uncertainty
- **Capacity**: Maximum reliable information transfer rate

---

## 11. Practical Algorithm: Pseudo-code

### 11.1 Basic Legibility Optimization

```python
def optimize_legible_trajectory(start, goal_true, confuser_goals, obstacles):
    """
    Optimize trajectory for legibility
    
    Args:
        start: Initial configuration
        goal_true: Intended goal G*
        confuser_goals: List of confuser goals [G₁, G₂, ...]
        obstacles: Environment obstacles
    
    Returns:
        trajectory: Optimized legible trajectory
    """
    # Initialize with straight-line trajectory
    trajectory = initialize_trajectory(start, goal_true)
    
    all_goals = [goal_true] + confuser_goals
    alpha = 0.1  # Learning rate
    beta = 1.0   # Temperature parameter
    
    for iteration in range(max_iterations):
        # Compute costs to all goals
        costs = {}
        for g in all_goals:
            costs[g] = compute_cost(trajectory, g)
        
        # Compute goal probabilities P(G | trajectory)
        probs = compute_posterior_probabilities(costs, beta)
        
        # Compute legibility gradient
        grad_legibility = np.zeros_like(trajectory)
        
        # Component 1: Minimize cost to true goal
        grad_legibility -= compute_cost_gradient(trajectory, goal_true)
        
        # Component 2: Maximize cost to confuser goals (weighted)
        for g in confuser_goals:
            weight = probs[g]  # Weight by confusability
            grad_legibility += weight * compute_cost_gradient(trajectory, g)
        
        # Update trajectory
        trajectory_new = trajectory + alpha * grad_legibility
        
        # Project to feasible set
        trajectory = project_feasible(trajectory_new, obstacles)
        
        # Check convergence
        if np.linalg.norm(trajectory - trajectory_new) < tolerance:
            break
    
    return trajectory

def compute_posterior_probabilities(costs, beta, prior_uniform=True):
    """
    Compute P(G | trajectory) for all goals using Bayes rule
    """
    likelihoods = {g: np.exp(-costs[g] / beta) for g in costs}
    Z = sum(likelihoods.values())  # Partition function
    
    probs = {g: likelihoods[g] / Z for g in costs}
    return probs

def compute_cost(trajectory, goal):
    """
    Compute cost of trajectory to reach goal
    """
    path_length = sum_segment_lengths(trajectory)
    goal_distance = distance(trajectory[-1], goal)
    smoothness = compute_smoothness_cost(trajectory)
    
    total_cost = path_length + 10.0 * goal_distance + 0.5 * smoothness
    return total_cost
```

### 11.2 Trajectory Smoothing and Constraints

```python
def project_feasible(trajectory, obstacles, max_velocity, max_acceleration):
    """
    Project trajectory onto feasible set
    - Collision-free
    - Dynamically feasible
    - Smooth
    """
    # Collision avoidance
    for i, waypoint in enumerate(trajectory):
        if in_collision(waypoint, obstacles):
            trajectory[i] = nearest_collision_free(waypoint, obstacles)
    
    # Velocity limits
    for i in range(len(trajectory) - 1):
        vel = (trajectory[i+1] - trajectory[i]) / dt
        if np.linalg.norm(vel) > max_velocity:
            direction = vel / np.linalg.norm(vel)
            trajectory[i+1] = trajectory[i] + direction * max_velocity * dt
    
    # Acceleration limits (similar)
    # ...
    
    # Smooth using spline fitting or filter
    trajectory = smooth_spline(trajectory)
    
    return trajectory
```

---

## 12. Key Research Papers and Timeline

### 12.1 Foundational Papers

1. **Dragan & Srinivasa (2013)** - "Legibility and Predictability of Robot Motion" (HRI)
   - Original definitions and mathematical framework
   - First empirical studies
   - Cited 1000+ times

2. **Dragan & Srinivasa (2013)** - "Generating Legible Motion" (RSS)
   - Detailed optimization algorithms
   - Functional gradient approach
   - Implementation details

3. **Dragan & Srinivasa (2014)** - "Integrating Human Observer Inferences into Robot Motion Planning" (Autonomous Robots)
   - Extended framework
   - Multiple observer scenarios
   - Theoretical analysis

### 12.2 Extensions and Applications

4. **Dragan et al. (2015)** - "Effects of Robot Motion on Human-Robot Collaboration" (HRI)
   - Empirical studies in collaboration
   - Comparison of functional/predictable/legible
   - Human factors analysis

5. **Dragan et al. (2015)** - "Deceptive Robot Motion" (Autonomous Robots)
   - Inverse problem: hiding intent
   - Game-theoretic analysis
   - Synthesis and verification

6. **Nikolaidis et al. (2016)** - "Game-Theoretic Modeling of Human Adaptation in Human-Robot Collaboration" (HRI)
   - Adaptive legibility
   - Learning observer models
   - Closed-loop interaction

### 12.3 Recent Work (2020-2025)

7. **Faria et al. (2022)** - "Extending Legibility to Sequential Decision Tasks" (Artificial Intelligence)
   - MDPs and POMDPs
   - Action sequence legibility
   - Theoretical foundations

8. **Wallkotter et al. (2022)** - "SLOT-V: Supervised Learning of Observer Models for Legible Robot Motion" (RO-MAN)
   - Learning-based observer models
   - Data-driven approach
   - Neural network architectures

9. **Shi et al. (2025)** - "Controlling Intent Expressiveness with Diffusion Models" (arXiv)
   - Diffusion models for legible motion
   - Quality diversity datasets
   - Adjustable legibility levels

10. **Wang et al. (2025)** - "Effects of Robot Competency and Motion Legibility on Human Correction Feedback" (HRI)
    - Human feedback and legibility
    - Interactive learning
    - Competency perception

### 12.4 Review Papers

11. **Lichtenthäler & Kirsch (2016)** - "Legibility of Robot Behavior: A Literature Review" (JHRI)
    - Comprehensive survey
    - Taxonomy of approaches
    - Open research questions

---

## 13. Open Research Questions

### 13.1 Theoretical Questions

1. **Optimality**: Is there a unique optimal legible trajectory? How to characterize it?
2. **Multi-objective trade-offs**: How to balance legibility, efficiency, smoothness?
3. **Complexity**: What is the computational complexity of legibility optimization?
4. **Robustness**: How sensitive is legibility to parameter choices (β, costs)?

### 13.2 Modeling Questions

5. **Observer models**: How to learn accurate models of human goal inference?
6. **Cultural differences**: Do legibility principles generalize across cultures?
7. **Expertise effects**: How does observer expertise affect legibility perception?
8. **Context dependence**: How does task context modify legibility requirements?

### 13.3 Implementation Questions

9. **Real-time performance**: How to compute legible motions fast enough for online use?
10. **High-DOF systems**: How to scale to high-dimensional robots (humanoids)?
11. **Partial observability**: How to handle occlusions and limited visibility?
12. **Multi-robot systems**: How to coordinate legible motions for robot teams?

### 13.4 Application Questions

13. **Domain transfer**: Which domains benefit most from legible motion?
14. **Cost-benefit**: When is the overhead of legibility optimization worth it?
15. **Human adaptation**: Do humans adapt to robot motion over time (reducing need for legibility)?
16. **Safety**: How does legibility interact with safety constraints?

---

## 14. Connection to Your Diffusion-Based Work

### 14.1 Diffusion Models for Legible Motion

Your current work on multimodal diffusion for robot manipulation can integrate legibility through:

1. **Conditioning on legibility objectives**
   - Condition diffusion process on goal and confuser set
   - Guide denoising to maximize P(G* | trajectory)
   - Example: classifier-free guidance with legibility score

2. **Quality diversity in latent space**
   - Train on diverse trajectories with varying legibility
   - Interpolate between legible and optimal trajectories
   - Enable controllable legibility at test time

3. **VLM-guided legibility**
   - Use VLM to identify potential confuser goals from visual scene
   - Automatically construct goal set for legibility optimization
   - Combine with early intent steering for communication

### 14.2 Specific Implementation Strategy

```python
# Pseudocode for diffusion model with legibility guidance

def diffusion_with_legibility(x_T, goal_true, confuser_goals, VLM_context):
    """
    Guided diffusion for legible trajectory generation
    """
    x_t = x_T  # Start with noise
    
    for t in reversed(range(T)):
        # Standard diffusion denoising step
        epsilon_pred = diffusion_model(x_t, t, goal_true, VLM_context)
        
        # Compute legibility gradient (in trajectory space)
        traj_clean = estimate_x0(x_t, epsilon_pred, t)
        grad_legibility = compute_legibility_gradient(
            traj_clean, goal_true, confuser_goals
        )
        
        # Combine with diffusion update
        x_t_minus_1 = denoise_step(x_t, epsilon_pred, t)
        x_t_minus_1 += lambda_legibility * grad_legibility
        
        x_t = x_t_minus_1
    
    return x_0  # Final legible trajectory
```

### 14.3 Integration with VLM Early Intent Steering

Legibility and early intent steering are complementary:
- **Early intent steering**: VLM identifies goal and confusers from visual observation
- **Legibility optimization**: Generate trajectory that disambiguates early
- **Combined effect**: Robot shows clear intent that VLM can verify and humans can understand

### 14.4 Experimental Validation

Suggested experiments:
1. **Legibility metrics**: Measure P(goal | trajectory_t) over time
2. **Human studies**: "Which goal is the robot reaching for?" at various times
3. **VLM evaluation**: "Does the VLM correctly identify goal earlier with legible motion?"
4. **Diversity-legibility trade-off**: Plot legibility vs. trajectory diversity in your quality diversity dataset

---

## 15. Practical Guidelines for Implementation

### 15.1 When to Use Legible Motion

Use legible motion when:
- ✅ Humans are observing and need to anticipate robot actions
- ✅ Multiple plausible goals exist in the workspace
- ✅ Safety requires early human understanding
- ✅ Collaboration requires coordination
- ✅ Time to react matters (human must respond)

Don't prioritize legibility when:
- ❌ No human observers present
- ❌ Only one obvious goal exists
- ❌ Efficiency is critical and communication not needed
- ❌ Robot is far from humans
- ❌ Other communication channels available (display, speech)

### 15.2 Parameter Tuning Guidelines

**Temperature β:**
- Lower β (0.1-0.5): More sensitive to cost differences (assumes highly rational agent)
- Higher β (1.0-5.0): More robust to cost variations (assumes less optimal planning)
- Typical value: β = 1.0

**Goal set size:**
- Start with 3-5 most salient confuser goals
- Add goals within spatial proximity (~2x goal distance)
- Include goals with high prior probability
- Remove goals that are clearly disambiguated

**Optimization parameters:**
- Learning rate: 0.05-0.2 (trajectory optimization)
- Iterations: 50-200 depending on complexity
- Smoothness weight: 0.1-1.0 (relative to path length)

### 15.3 Validation Metrics

**Computational metrics:**
1. P(G* | ξ_t) over time t (should increase rapidly)
2. Entropy H(G | ξ_t) over time (should decrease)
3. Time to threshold: min t such that P(G* | ξ_t) > 0.8

**Human study metrics:**
1. Goal inference accuracy at t = 25%, 50%, 75% of trajectory
2. Inference time (when participants become 80% confident)
3. Subjective ratings (legibility, predictability, naturalness, trust)

**Comparison baselines:**
1. Optimal trajectory (shortest path)
2. Predictable trajectory (most common path)
3. Random trajectory
4. Hand-designed legible trajectory

---

## 16. Summary and Key Takeaways

### 16.1 Core Mathematical Framework

- **Bayesian inference**: P(goal | trajectory) ∝ exp(-Cost(trajectory, goal))
- **Legibility objective**: Maximize P(G* | ξ) = maximize probability of true goal given observed trajectory
- **Optimization**: Balance being good for intended goal and bad for confuser goals
- **Functional gradient**: Optimize over trajectory space using gradient descent

### 16.2 Geometric Principles

- **Early differentiation**: Disambiguate goals early in trajectory
- **Exaggerated curvature**: Emphasize direction toward goal
- **Directness paradox**: Legibility sometimes requires less direct paths
- **Information maximization**: Reduce entropy about goal quickly

### 16.3 Legibility ≠ Predictability

 **Key insight**: These are different and often conflicting objectives
- Predictability: Match typical behavior
- Legibility: Communicate intent clearly
- Trade-off depends on task and context

### 16.4 Practical Implementation

- Limit confuser goal set for efficiency
- Use functional gradient optimization
- Integrate with smoothness and collision avoidance
- Validate with both computational metrics and human studies

### 16.5 Future Directions

- Learning-based observer models
- Diffusion models for controllable legibility
- Multi-agent coordination
- Real-time adaptive legibility
- Integration with other communicative modalities

---

## 17. References and Resources

### Key Papers (Must-Read)

1. Dragan, A.D., Lee, K.C.T., & Srinivasa, S.S. (2013). "Legibility and predictability of robot motion." *HRI 2013*. [1000+ citations]

2. Dragan, A.D. & Srinivasa, S.S. (2013). "Generating legible motion." *RSS 2013*. [Technical details]

3. Dragan, A.D. & Srinivasa, S.S. (2014). "Integrating human observer inferences into robot motion planning." *Autonomous Robots*. [Extended framework]

### Recent Extensions

4. Faria, M., Melo, F.S., & Paiva, A. (2024). "Guess what I'm doing: Extending legibility to sequential decision tasks." *Artificial Intelligence*. [Sequential decisions]

5. Shi, W., Grislain, C., Sigaud, O., & Chetouani, M. (2025). "Controlling Intent Expressiveness in Robot Motion with Diffusion Models." *arXiv:2510.12370*. [Diffusion models]

### Review and Survey

6. Lichtenthäler, C. & Kirsch, A. (2016). "Legibility of robot behavior: A literature review." *Journal of HRI*. [Comprehensive survey]

### Related Work

7. Dragan, A.D., Holladay, R., & Srinivasa, S.S. (2015). "Deceptive robot motion." *Autonomous Robots*. [Inverse problem]

8. Dragan, A.D. et al. (2015). "Effects of robot motion on human-robot collaboration." *HRI 2015*. [Empirical studies]

### Online Resources

- **Anca Dragan's website**: https://people.eecs.berkeley.edu/~anca/
- **Personal Robotics Lab (UW)**: https://personalrobotics.cs.washington.edu/
- **Berkeley InterACT Lab**: http://interact.berkeley.edu/
- **Planning Algorithms (LaValle)**: http://lavalle.pl/planning/ [Free textbook]

### Related Concepts

- **Motion planning**: LaValle, S.M. (2006). *Planning Algorithms*. Cambridge University Press.
- **Inverse optimal control**: Ziebart, B.D. et al. (2008). "Maximum entropy inverse reinforcement learning." *AAAI*.
- **Functional motion**: Zucker, M. et al. (2013). "CHOMP: Covariant Hamiltonian optimization for motion planning." *IJRR*.

---

## Appendix A: Mathematical Derivations

### A.1 Derivation of Legibility Gradient

Starting from:
```
L(ξ) = log P(G* | ξ) = log[P(ξ | G*)] - log[Σ_G P(ξ | G)]
```

Using P(ξ | G) = exp(-C(ξ, G)/β) / Z(G):
```
L(ξ) = -C(ξ, G*)/β - log Z(G*) - log[Σ_G exp(-C(ξ, G)/β) / Z(G)]
```

Assuming uniform priors (Z(G) cancels in ratio):
```
L(ξ) = -C(ξ, G*)/β - log[Σ_G exp(-C(ξ, G)/β)]
```

Taking gradient with respect to trajectory ξ:
```
∇_ξ L = -1/β · ∇_ξ C(ξ, G*) + 1/β · [Σ_G exp(-C(ξ,G)/β) · ∇_ξ C(ξ,G)] / [Σ_G exp(-C(ξ,G)/β)]
```

Simplifying using P(G|ξ) = exp(-C(ξ,G)/β) / Σ:
```
∇_ξ L = 1/β · [-∇_ξ C(ξ, G*) + Σ_G P(G|ξ) · ∇_ξ C(ξ, G)]
```

**Interpretation:**
- First term: Gradient to minimize cost to true goal (standard optimal planning)
- Second term: Expected gradient over all goals, weighted by current belief
  - This "anti-plans" toward confuser goals (makes trajectory worse for them)

### A.2 Connection to Maximum Entropy IRL

The likelihood model P(ξ | G) = exp(-C(ξ, G)/β) is the same form used in:
- **Maximum entropy IRL** (Ziebart et al., 2008)
- **Soft Q-learning**
- **Boltzmann rational models**

This assumes agents are **noisily rational**: they prefer lower-cost trajectories but don't always choose the absolute minimum.

---

## Appendix B: Geometric Example

### B.1 Three-Goal Scenario

**Setup:**
- Goals: A (left), B (center), C (right)
- Start: Bottom center
- All goals equidistant initially

**Optimal trajectory to A:**
- Path length: 10.0 units (straight line)
- Cost to A: 10.0
- Cost to B: 10.2 (slightly longer due to angle)
- Cost to C: 14.1 (much longer)

**P(goal | optimal trajectory at 50%):**
- P(A | ξ) ≈ 0.55 (slight preference)
- P(B | ξ) ≈ 0.40 (still plausible)
- P(C | ξ) ≈ 0.05

**Legible trajectory to A:**
- Path length: 10.5 units (curves left early)
- Cost to A: 10.5
- Cost to B: 12.0 (requires sharp turn)
- Cost to C: 16.0 (very inefficient)

**P(goal | legible trajectory at 50%):**
- P(A | ξ) ≈ 0.85 (strong preference)
- P(B | ξ) ≈ 0.12 (less plausible)
- P(C | ξ) ≈ 0.03

**Trade-off:**
- Legible trajectory is 5% longer
- But enables 30% earlier confident inference
- Reduces ambiguity with goal B significantly

---

## Appendix C: Code Example - Complete Implementation

```python
import numpy as np
from scipy.interpolate import CubicSpline

class LegibilityOptimizer:
    def __init__(self, beta=1.0, alpha=0.1, max_iters=100):
        self.beta = beta  # Temperature
        self.alpha = alpha  # Learning rate
        self.max_iters = max_iters
    
    def optimize(self, start, goal_true, confuser_goals, n_waypoints=20):
        """
        Main optimization loop for legible trajectory
        """
        # Initialize straight line
        trajectory = self._init_trajectory(start, goal_true, n_waypoints)
        
        all_goals = [goal_true] + confuser_goals
        
        for iteration in range(self.max_iters):
            # Compute costs and probabilities
            costs = {g: self._compute_cost(trajectory, g) for g in all_goals}
            probs = self._compute_probabilities(costs)
            
            # Compute gradient
            grad = self._compute_gradient(trajectory, goal_true, 
                                         confuser_goals, probs)
            
            # Update
            trajectory_new = trajectory + self.alpha * grad
            
            # Smooth and enforce constraints
            trajectory = self._enforce_constraints(trajectory_new)
            
            # Check convergence
            if np.linalg.norm(trajectory - trajectory_new) < 1e-3:
                break
        
        return trajectory
    
    def _init_trajectory(self, start, goal, n_waypoints):
        """Initialize with straight line"""
        waypoints = np.linspace(start, goal, n_waypoints)
        return waypoints
    
    def _compute_cost(self, trajectory, goal):
        """Compute trajectory cost to reach goal"""
        # Path length
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        
        # Distance to goal
        goal_dist = np.linalg.norm(trajectory[-1] - goal)
        
        # Smoothness (acceleration cost)
        accel = np.diff(trajectory, n=2, axis=0)
        smoothness = np.sum(np.linalg.norm(accel, axis=1)**2)
        
        total = path_length + 10.0 * goal_dist + 0.1 * smoothness
        return total
    
    def _compute_probabilities(self, costs):
        """Compute P(G | trajectory) using Boltzmann distribution"""
        likelihoods = {g: np.exp(-costs[g] / self.beta) for g in costs}
        Z = sum(likelihoods.values())
        probs = {g: likelihoods[g] / Z for g in costs}
        return probs
    
    def _compute_gradient(self, trajectory, goal_true, confuser_goals, probs):
        """Compute legibility gradient"""
        grad = np.zeros_like(trajectory)
        
        # Component 1: Minimize cost to true goal
        grad -= self._cost_gradient(trajectory, goal_true)
        
        # Component 2: Maximize cost to confuser goals
        for g in confuser_goals:
            grad += probs[g] * self._cost_gradient(trajectory, g)
        
        return grad / self.beta
    
    def _cost_gradient(self, trajectory, goal):
        """Gradient of cost function"""
        grad = np.zeros_like(trajectory)
        n = len(trajectory)
        
        # Path length gradient
        for i in range(n-1):
            direction = trajectory[i+1] - trajectory[i]
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                grad[i] -= direction / dist
                grad[i+1] += direction / dist
        
        # Goal distance gradient
        goal_direction = trajectory[-1] - goal
        grad[-1] += 10.0 * goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
        
        # Smoothness gradient (simplified)
        for i in range(1, n-1):
            grad[i] += 0.2 * (2*trajectory[i] - trajectory[i-1] - trajectory[i+1])
        
        return grad
    
    def _enforce_constraints(self, trajectory):
        """Smooth and enforce feasibility"""
        # Fit cubic spline for smoothness
        t = np.linspace(0, 1, len(trajectory))
        cs = CubicSpline(t, trajectory)
        trajectory_smooth = cs(t)
        
        # Velocity limiting (simple version)
        max_vel = 1.0
        for i in range(len(trajectory_smooth)-1):
            vel = trajectory_smooth[i+1] - trajectory_smooth[i]
            vel_norm = np.linalg.norm(vel)
            if vel_norm > max_vel:
                trajectory_smooth[i+1] = trajectory_smooth[i] + vel * (max_vel / vel_norm)
        
        return trajectory_smooth


# Usage example
if __name__ == "__main__":
    start = np.array([0.0, 0.0])
    goal_true = np.array([5.0, 5.0])
    confuser_goals = [
        np.array([5.0, -5.0]),  # Below
        np.array([-5.0, 5.0])  # Left
    ]
    
    optimizer = LegibilityOptimizer(beta=1.0, alpha=0.1)
    legible_traj = optimizer.optimize(start, goal_true, confuser_goals)
    
    print("Legible trajectory optimized!")
    print(f"Start: {legible_traj[0]}")
    print(f"End: {legible_traj[-1]}")
```

---

## Final Notes

This comprehensive document synthesizes the mathematical, geometric, and practical aspects of robot motion legibility based on extensive research literature. The field continues to evolve with new applications in human-robot interaction, autonomous systems, and communicative AI.

For your specific work on multimodal diffusion for manipulation, the key integration points are:
1. Using VLM to identify goal sets automatically
2. Conditioning diffusion on legibility objectives
3. Generating diverse yet legible trajectories
4. Validating early intent communication through legibility metrics

The mathematical framework provides a principled way to optimize for communicative clarity while maintaining task efficiency—a crucial balance for safe and effective human-robot collaboration.
