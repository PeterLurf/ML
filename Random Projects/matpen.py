
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import animation

#############################################
# 1. System and Simulation Parameters
#############################################
# Physical parameters
m1 = 1.0         # mass of the first pendulum bob
m2 = 1.0         # mass of the second pendulum bob
l1 = 1.0         # length of the first rod
l2 = 1.0         # length of the second rod
g  = 9.81        # gravitational acceleration

# Simulation parameters
dt = 0.005       # integration time step (seconds)
num_steps = 2000 # number of integration steps (~10 seconds total)

#############################################
# 2. Hyperparameters for Loss Terms
#############################################
# (These can be tuned to get the desired “interesting” behavior.)
# Penalty to avoid trivial starting angles (e.g. hanging down)
trivial_penalty_weight = 1.0
trivial_penalty_scale  = 0.01

# Penalty if starting angles are too close to each other
separation_weight = 1.0
separation_scale  = 0.1

# Rewards/Penalties for early chaotic phase vs. late stability:
early_variation_weight = 1.0  # reward high variation in early phase
late_variation_weight  = 1.0   # penalize high variation in late phase

early_kinetic_weight   = 0.5   # reward high kinetic energy early on
late_kinetic_weight    = 1.0   # penalize high kinetic energy late (for stability)

# Reward for non-smooth (irregular) early dynamics
irregularity_weight    = 0.1

# Penalty if overall motion is too static (low variation over the whole simulation)
static_penalty_weight  = 1.0

#############################################
# 3. Dynamics and RK4 Integration
#############################################
def double_pendulum_dynamics(state):
    """
    Given state = [theta1, theta2, omega1, omega2], compute the time derivatives.
    (This formulation is one common version of the double pendulum equations.)
    """
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2

    denom1 = l1 * (2*m1 + m2 - m2 * torch.cos(2*theta1 - 2*theta2))
    num1   = (- g * (2*m1 + m2) * torch.sin(theta1)
              - m2 * g * torch.sin(theta1 - 2*theta2)
              - 2 * torch.sin(delta) * m2 * (omega2**2 * l2 + omega1**2 * l1 * torch.cos(delta)))
    domega1 = num1 / denom1

    denom2 = l2 * (2*m1 + m2 - m2 * torch.cos(2*theta1 - 2*theta2))
    num2   = (2 * torch.sin(delta) *
              (omega1**2 * l1 * (m1 + m2)
               + g * (m1 + m2) * torch.cos(theta1)
               + omega2**2 * l2 * m2 * torch.cos(delta)))
    domega2 = num2 / denom2

    return torch.stack([omega1, omega2, domega1, domega2])

def simulate_double_pendulum_rk4(theta1_init, theta2_init, num_steps, dt):
    """
    Uses a 4th order Runge–Kutta (RK4) method to integrate the double pendulum dynamics.
    Returns a tensor of shape (num_steps, 4) recording the full state:
      [theta1, theta2, omega1, omega2] at each time step.
    """
    # Start with initial angles and zero angular velocities.
    state = torch.stack([theta1_init, theta2_init, torch.tensor(0.0), torch.tensor(0.0)])
    states = []
    for _ in range(num_steps):
        states.append(state)
        k1 = double_pendulum_dynamics(state)
        k2 = double_pendulum_dynamics(state + dt/2 * k1)
        k3 = double_pendulum_dynamics(state + dt/2 * k2)
        k4 = double_pendulum_dynamics(state + dt * k3)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return torch.stack(states)

#############################################
# 4. Loss Function Definition
#############################################
def loss_fn(params):
    """
    The loss function is designed to encourage a system that:
      - Does not start in the trivial (hanging) state.
      - Has initial angles that are not too close to each other.
      - Exhibits a long chaotic (high–variation, high kinetic energy) early phase.
      - Settles into a stable (low–variation, low kinetic energy) late phase.
      - Has irregular (jerky) early behavior.
      - Is not completely static overall.
    
    The loss is constructed as:
      loss = trivial_penalty + separation_penalty + static_penalty +
             (late_variation_weight * variation_late + late_kinetic_weight * kinetic_late) 
             - (early_variation_weight * variation_early + early_kinetic_weight * kinetic_early)
             - (irregularity_weight * irregularity)
    """
    theta1_init, theta2_init = params[0], params[1]
    
    # 4.1 Trivial penalty: penalize if either starting angle is near zero.
    trivial_penalty = trivial_penalty_weight * (torch.exp(-theta1_init**2 / trivial_penalty_scale) +
                                                torch.exp(-theta2_init**2 / trivial_penalty_scale))
    
    # 4.2 Separation penalty: penalize if starting angles are too close.
    separation_penalty = separation_weight * torch.exp(-((theta1_init - theta2_init)**2) / separation_scale)
    
    # 4.3 Run the simulation using the RK4 integrator.
    full_states = simulate_double_pendulum_rk4(theta1_init, theta2_init, num_steps, dt)
    angles = full_states[:, :2]      # shape: (num_steps, 2)
    velocities = full_states[:, 2:4] # shape: (num_steps, 2)
    
    # 4.4 Split the trajectory into early and late halves.
    half = num_steps // 2
    early_angles = angles[:half]
    late_angles  = angles[half:]
    early_vel    = velocities[:half]
    late_vel     = velocities[half:]
    
    # 4.5 Compute variation (mean absolute differences) in early and late phases.
    early_diff = early_angles[1:] - early_angles[:-1]
    variation_early = torch.mean(torch.abs(early_diff))
    
    late_diff = late_angles[1:] - late_angles[:-1]
    variation_late = torch.mean(torch.abs(late_diff))
    
    # 4.6 Kinetic energy calculation (using a standard double pendulum formula)
    def kinetic_energy(ang, vel):
        theta1 = ang[:, 0]
        theta2 = ang[:, 1]
        omega1 = vel[:, 0]
        omega2 = vel[:, 1]
        T = 0.5*(m1 + m2)*(l1*omega1)**2 + 0.5*m2*(l2*omega2)**2 \
            + m2 * l1 * l2 * omega1 * omega2 * torch.cos(theta1 - theta2)
        return torch.mean(T)
    
    kinetic_early = kinetic_energy(early_angles, early_vel)
    kinetic_late  = kinetic_energy(late_angles, late_vel)
    
    # 4.7 Irregularity: measure the "jerkiness" via second differences on the early phase.
    early_diff2 = early_diff[1:] - early_diff[:-1]
    irregularity = torch.mean(early_diff2**2)
    
    # 4.8 Static penalty: if the overall variation is too low, add extra loss.
    overall_variation = torch.mean(torch.abs(angles[1:] - angles[:-1]))
    static_penalty = static_penalty_weight * (1.0 - torch.tanh(overall_variation))
    
    # 4.9 Combine the terms:
    # Reward early chaotic behavior (subtract early metrics) and penalize late chaos (add late metrics)
    total_loss = (trivial_penalty +
                  separation_penalty +
                  static_penalty +
                  late_variation_weight * variation_late +
                  late_kinetic_weight  * kinetic_late -
                  early_variation_weight * variation_early -
                  early_kinetic_weight  * kinetic_early -
                  irregularity_weight   * irregularity)
    
    return total_loss

#############################################
# 5. Optimization: Finding "Interesting" Behavior
#############################################
# We optimize over the two initial angles.

#############################################
# 6. Simulation and Plots of the Optimized System
#############################################
# Run a final simulation with the optimized initial angles.
full_states = simulate_double_pendulum_rk4(torch.tensor(0.1695), torch.tensor(0.6076), num_steps, dt)

angles = full_states[:, :2].detach().numpy()
time_axis = dt * torch.arange(num_steps).numpy()

# Plot the time evolution of the angles.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time_axis, angles[:, 0], label='theta1')
plt.plot(time_axis, angles[:, 1], label='theta2')
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Time Evolution of Angles")
plt.legend()

# Phase plot: theta1 vs. theta2.
plt.subplot(1, 2, 2)
plt.plot(angles[:, 0], angles[:, 1], 'o-', markersize=2)
plt.xlabel("theta1 (rad)")
plt.ylabel("theta2 (rad)")
plt.title("Phase Plot")
plt.tight_layout()
plt.show()

# Plot the loss history.
plt.figure()

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.show()

#############################################
# 7. Animation: Visualize the Motion
#############################################
def compute_positions(theta1, theta2):
    """
    Convert angles to Cartesian coordinates.
      First bob:  x1 = l1*sin(theta1), y1 = -l1*cos(theta1)
      Second bob: x2 = x1 + l2*sin(theta1+theta2), y2 = y1 - l2*cos(theta1+theta2)
    """
    x1 = l1 * torch.sin(theta1)
    y1 = -l1 * torch.cos(theta1)
    x2 = x1 + l2 * torch.sin(theta1 + theta2)
    y2 = y1 - l2 * torch.cos(theta1 + theta2)
    return x1, y1, x2, y2

# Prepare the (x, y) positions for each time step.
theta1_np = full_states[:, 0].detach().numpy()
theta2_np = full_states[:, 1].detach().numpy()
x1_list, y1_list, x2_list, y2_list = [], [], [], []
for t1, t2 in zip(theta1_np, theta2_np):
    t1_tensor = torch.tensor(t1)
    t2_tensor = torch.tensor(t2)
    x1, y1, x2, y2 = compute_positions(t1_tensor, t2_tensor)
    x1_list.append(x1.item())
    y1_list.append(y1.item())
    x2_list.append(x2.item())
    y2_list.append(y2.item())

# Set up the animation.
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(- (l1 + l2 + 0.5), l1 + l2 + 0.5)
ax.set_ylim(- (l1 + l2 + 0.5), l1 + l2 + 0.5)
ax.set_aspect('equal')
ax.grid()

rod_line, = ax.plot([], [], 'o-', lw=2, color='blue')
trace_line, = ax.plot([], [], '-', lw=1, color='gray')
trace_x, trace_y = [], []

def init_anim():
    rod_line.set_data([], [])
    trace_line.set_data([], [])
    return rod_line, trace_line

def animate_frame(i):
    x0, y0 = 0, 0  # pivot at the origin
    x1, y1 = x1_list[i], y1_list[i]
    x2, y2 = x2_list[i], y2_list[i]
    rod_line.set_data([x0, x1, x2], [y0, y1, y2])
    trace_x.append(x2)
    trace_y.append(y2)
    trace_line.set_data(trace_x, trace_y)
    return rod_line, trace_line

ani = animation.FuncAnimation(fig, animate_frame, frames=len(x1_list),
                              init_func=init_anim, interval=20, blit=True)
plt.title("Animation of a System with a Long Chaotic Phase")
plt.show()
