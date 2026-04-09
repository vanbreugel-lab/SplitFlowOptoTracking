
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

def find_value_at_T(values,jumptimes, T):
    idx = np.where(jumptimes == T)[-1][-1]
    return values[idx]

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

# Scale trajectories to desired ranges
def scale_to_range(data, new_min, new_max):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:  # Prevent division by zero
        return np.full_like(data, (new_min + new_max)/2)
    return ((data - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min



def brownian_motion_sim( dt, T, index, mu=0, sigma_x=.1, sigma_y=.1, x0=0, ):
    N = int(T/dt)
    x_vals = np.random.normal(mu*dt, sigma_x*np.sqrt(dt), N)
    y_vals = np.random.normal(mu*dt, sigma_y*np.sqrt(dt), N)
    BM_x = np.append(np.zeros(1),np.cumsum(x_vals))
    BM_xdt = np.append(np.zeros(1),x_vals)
    BM_y = np.append(np.zeros(1),np.cumsum(y_vals))
    BM_ydt = np.append(np.zeros(1),y_vals)
    if x0 is not None:
        BM_x= np.cumsum(np.append(x0, x_vals))
        BM_xdt= np.append(x0, x_vals)
        BM_y= np.cumsum(np.append(x0, y_vals))
        BM_ydt= np.append(x0, y_vals)
    time = np.linspace(0,T, N+1)    
    traj_id = 'brownian_' + str(index)
    #print(traj_id)
    return pd.DataFrame({'x': BM_x, 'y': BM_y, 'xvel': BM_xdt, 'yvel': BM_ydt, 'time stamp': time*1000, 'obj_id_unique': traj_id })



def generate_random_turns_variable_straights(total_time=3.01, straight_duration_range=(.15,0.6), dt=0.01, 
                             min_angle=50, max_angle=120, turn_duration=0.12, meanspeed=np.random.normal(.4, .15)):
    """
    Generate a smooth trajectory with controlled turns and velocity information.
    Modified to stay within bounds: y ∈ [-0.25, 0.25], x ∈ [-0.5, 0.5]
    Now with random straight durations between turns.
    
    Parameters:
    - total_time: total duration in seconds
    - straight_duration_range: tuple of (min, max) duration of straight movements in seconds
    - dt: time step (0.01s default)
    - min_angle: minimum turn angle in degrees (60 default)
    - max_angle: maximum turn angle in degrees (80 default)
    - turn_duration: time over which turn is completed (0.1s default)
    - meanspeed: mean speed of movement (0.4 default)
    
    Returns:
    - times: array of time points
    - x, y: arrays of coordinates
    - vx, vy: arrays of x and y velocities
    - angles: array of movement angles (radians)
    """
    # Convert angles to radians
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    
    # Calculate steps
    turn_steps = int(turn_duration / dt)
    total_steps = int(total_time / dt)
    
    speed = np.random.normal(meanspeed, 0.1)
    # Initialize arrays
    times = np.arange(0, total_time, dt)
    angles = np.zeros(total_steps)
    x = np.zeros(total_steps)
    y = np.zeros(total_steps)
    vx = np.zeros(total_steps)
    vy = np.zeros(total_steps)
    
    # Initial conditions (start near center)
    current_angle = 2 * np.pi * np.random.random()
    x[0] = 0
    y[0] = 0
    
    # Initialize turn variables
    turning = False
    turn_start_angle = 0
    turn_target_angle = 0
    turn_progress = 0
    
    # Initialize straight movement variables
    steps_until_next_turn = 0
    current_straight_duration = np.round(np.random.choice(np.linspace(straight_duration_range[0],straight_duration_range[1],7)),2)
    steps_per_straight = int(current_straight_duration / dt)
    
    for i in range(1, total_steps):
        # Check if we should start a new turn
        if not turning and (steps_until_next_turn <= 0 or i == 1):
            turning = True
            turn_start_angle = current_angle
            # Randomly choose turn direction (left or right)
            turn_sign = 1#np.sign(np.random.randn())
            turn_magnitude = min_angle_rad + (max_angle_rad - min_angle_rad) * np.random.random()
            turn_target_angle = turn_start_angle + turn_sign * turn_magnitude
        
            turn_progress = 0
            # Set new random straight duration for after this turn
            current_straight_duration = np.round(np.random.choice(np.linspace(straight_duration_range[0],straight_duration_range[1],7)),2)
            #print(current_straight_duration)
            steps_per_straight = int(current_straight_duration / dt)
            steps_until_next_turn = steps_per_straight

        # Handle turning
        if turning:
            turn_progress += 1
            # Cubic easing function for smooth turn
            t = min(turn_progress / turn_steps, 1.0)
            t_ease = t**2 * (3 - 2*t)  # Smoothstep function
            current_angle = turn_start_angle + (turn_target_angle - turn_start_angle) * t_ease
            
            if turn_progress >= turn_steps:
                turning = False
                current_angle = turn_target_angle  # Ensure exact final angle
        else:
            steps_until_next_turn -= 1
        
        angles[i] = current_angle
        
        # Calculate velocities
        vx[i] = np.cos(current_angle) * speed + np.random.normal(0.0, 0.06)
        vy[i] = np.sin(current_angle) * speed + np.random.normal(0.0, 0.06)
        
        # Update position using velocities
        x[i] = x[i-1] + vx[i] * dt
        y[i] = y[i-1] + vy[i] * dt
        
    return times, x, y, vx, vy, angles

def generate_random_turn_trajectory(total_time=3.01, straight_duration=np.random.normal(0.3,0.15), direction=None, dt=0.01, 
                              min_angle=60, max_angle=120, turn_duration=0.12, meanspeed=np.random.normal(.4, .15)):
    """
    Generate a smooth trajectory with controlled turns and velocity information.
    
    Parameters:
    - total_time: total duration in seconds
    - straight_duration: duration of straight movements (300ms default)
    - dt: time step (0.01s default)
    - min_angle: minimum turn angle in degrees (60 default)
    - max_angle: maximum turn angle in degrees (80 default)
    - turn_duration: time over which turn is completed (0.1s default)
    - speed: constant speed of movement (1.0 default)
    
    Returns:
    - times: array of time points
    - x, y: arrays of coordinates
    - vx, vy: arrays of x and y velocities
    - angles: array of movement angles (radians)
    """
    # Convert angles to radians
    min_angle_rad = np.deg2rad(min_angle)
    max_angle_rad = np.deg2rad(max_angle)
    
    # Calculate steps
    steps_per_segment = int(straight_duration / dt)
    turn_steps = int(turn_duration / dt)
    total_steps = int(total_time / dt)
    
    speed  = np.random.normal(meanspeed,0.15)
    # Initialize arrays
    times = np.arange(0, total_time, dt)
    angles = np.zeros(total_steps)
    x = np.zeros(total_steps)
    y = np.zeros(total_steps)
    vx = np.zeros(total_steps)
    vy = np.zeros(total_steps)
    
    # Initial conditions
    current_angle = 2 * np.pi * np.random.random()
    
    # Generate angle sequence first
    segment_count = 0
    turning = False
    turn_start_angle = 0
    turn_target_angle = 0
    turn_progress = 0
    
    for i in range(1, total_steps):
        # Check if we should start a new turn
        if not turning and (i % steps_per_segment == 0 or i == 1):
            turning = True
            turn_start_angle = current_angle
            # Randomly choose turn direction (left or right)
            if direction is None:
                turn_sign = np.sign(np.random.randn())
            else: 
                turn_sign= direction

            #print(turn_sign)    
            turn_magnitude = min_angle_rad + (max_angle_rad - min_angle_rad) * np.random.random()
            turn_target_angle = turn_start_angle + turn_sign * turn_magnitude
            turn_progress = 0
        
        # Handle turning
        if turning:
            turn_progress += 1
            # Cubic easing function for smooth turn
            t = min(turn_progress / turn_steps, 1.0)
            t_ease = t**2 * (3 - 2*t)  # Smoothstep function
            current_angle = turn_start_angle + (turn_target_angle - turn_start_angle) * t_ease
            
            if turn_progress >= turn_steps:
                turning = False
                current_angle = turn_target_angle  # Ensure exact final angle
        
        angles[i] = current_angle
        
        # Calculate velocities
        vx[i] = np.cos(current_angle) * speed + np.random.normal(0.0,0.06)
        vy[i] = np.sin(current_angle) * speed + np.random.normal(0.0,0.06)
        
        # Update position using velocities
        x[i] = x[i-1] + vx[i] * dt
        y[i] = y[i-1] + vy[i] * dt
    
    
    return times, x, y, vx, vy, angles
def levy_step(alpha=2, scale=0.03):
    """Generate a step length from a Lévy distribution."""
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    step = scale * (u ** (-1 / alpha)) * np.sin(alpha * v) / (np.cos(v) ** (1 / alpha))
    return np.abs(step)

def levy_flight_fruit_fly(n_steps=301, alpha=2, scale=0.03, turn_angle_std=0.2, dt=0.01):
    """Generate a Lévy flight trajectory resembling fruit fly movement and compute velocity."""
    x, y = [0], [0]  # Starting at origin
    velocities = []   # Stores speed (magnitude of velocity)
    vx, vy = [0], [0]   # Stores velocity components
    time = np.arange(0,n_steps+1)*dt
    for _ in range(n_steps):
        # Decide between short (Brownian-like) or long (Lévy) step
        if np.random.rand() < 0.2:  # 20% chance of short step (local search)
            step_length = np.random.exponential(scale=2)
        else:  # 80% chance of long step 
            step_length = levy_step(alpha, scale)
        
        # Add directional persistence (slight turning angle)
        if len(x) > 1:
            last_angle = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])
        else:
            last_angle = np.random.uniform(0, 2 * np.pi)
        
        angle = last_angle + np.random.normal(0, turn_angle_std)
        
        # Update position
        x_new = x[-1] + step_length * np.cos(angle)
        y_new = y[-1] + step_length * np.sin(angle)
        
        # Compute velocity (displacement / dt)
        dx = (x_new - x[-1])
        dy = (y_new - y[-1])
        velocity_x = dx / dt
        velocity_y = dy / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Store results
        x.append(x_new)
        y.append(y_new)
        velocities.append(speed)
        vx.append(velocity_x)
        vy.append(velocity_y)
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    vx = np.array(vx)
    vy = np.array(vy)
    speeds = np.array(velocities)
    
    # Scale positions to fit within specified bounds while keeping start at (0,0)
    # Find maximum absolute values
    max_x = np.max(np.abs(x))
    max_y = np.max(np.abs(y))
    
    # Scale x to [-0.5, 0.5] while keeping x[0] = 0
    if max_x > 0:
        x_scaled = x / max_x * 0.5  # This maintains x[0] = 0
    else:
        x_scaled = x
    
    # Scale y to [-0.25, 0.25] while keeping y[0] = 0
    if max_y > 0:
        y_scaled = y / max_y * 0.25  # This maintains y[0] = 0
    else:
        y_scaled = y
    
    # Velocity scaling
    max_vx = np.max(np.abs(vx))
    max_vy = np.max(np.abs(vy))
    
    if max_vx > 0:
        vx_scaled = vx / max_vx *0.5  + 0.3 # Scale to [-1,1]
    else:
        vx_scaled = vx 
        
    if max_vy > 0:
        vy_scaled = vy / max_vy  *0.5 + 0.3# Scale to [-1,1]
    else:
        vy_scaled = vy 
    
    # Recalculate speeds with scaled velocities
    speeds_scaled = np.sqrt(vx_scaled**2 + vy_scaled**2)
    
    return time, x_scaled, y_scaled, speeds_scaled, vx_scaled, vy_scaled




def run_and_tumble(total_time, dt, run_time, tumble_time, speed, seed=None):
    """
    Generate a 2D trajectory using a run-and-tumble algorithm.
    
    Parameters:
    total_time (float): Total simulation time.
    dt (float): Time step for simulation.
    run_time (float): Mean duration of straight runs (exponential distribution).
    tumble_time (float): Fixed duration of tumbling periods.
    speed (float): Constant speed during both runs and tumbles.
    seed (int, optional): Seed for reproducibility.
    
    Returns:
    tuple: (time_points, positions, velocities)
        time_points: 1D array of time stamps.
        positions: 2D array of (x, y) positions at each time point.
        velocities: 2D array of (vx, vy) velocities at each time point.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize arrays
    n_steps = int(total_time / dt) + 1
    time_points = np.linspace(0, total_time, n_steps)
    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))
    
    # Initial direction (radians)
    angle = np.random.uniform(0, 2 * np.pi)
    
    t = 0
    step = 0
    is_running = True
    run_duration = np.random.exponential(run_time)
    tumble_end = 0
    
    while step < n_steps - 1:
        step += 1
        t = step * dt
        
        # Switch between run and tumble states
        if is_running:
            if t >= run_duration:
                is_running = False
                tumble_end = t + tumble_time
        else:
            if t >= tumble_end:
                is_running = True
                run_duration = t + np.random.exponential(run_time)
                angle = np.random.uniform(0, 2 * np.pi)  # New run direction
        
        # Update direction during tumble
        if not is_running:
            angle = np.random.uniform(0, 2 * np.pi)
        
        # Compute velocity components
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        velocities[step] = [vx, vy]
        
        # Update position
        positions[step] = positions[step-1] + velocities[step] * dt
    #positions[:,0],positions[:,1] = scale_trajectory(positions[:,0],positions[:,1]) 
    return time_points, positions, velocities


def plot_trajectory_with_velocities(times, x, y, vx, vy, angles):
    """Plot the smooth trajectory with velocity information."""
    plt.figure(figsize=(14, 10))
    
    # Main trajectory plot
    plt.subplot(2, 2, (1, 2))
    plt.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, label='Path')
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    plt.ylim(-0.25,0.25)
    plt.xlim(-0.5,0.5)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Trajectory with 60-120° Turns')

    plt.legend()
    
    #plt.grid(True, alpha=0.3)
    #plt.axis('equal')
    plt.ylim(-0.25,0.25)
    plt.xlim(-0.5,0.5)
    
    
    # Velocity components plot
    plt.subplot(2, 2, 3)
    plt.plot(times, vx, 'r-', label='X Velocity')
    plt.plot(times, vy, 'b-', label='Y Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('Velocity Components')

    plt.legend()
    plt.grid(True, alpha=0.3)
    # Speed and angle plot
    plt.subplot(2, 2, 4)
    plt.plot(times, wrap_angle(np.rad2deg(angles)), 'm-', label='Heading Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle')
    plt.title('Speed and Heading Direction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.plot(times, np.sqrt(vx**2 + vy**2), 'r-', label='Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad) / Speed (m/s)')
    plt.title('Speed and Heading Direction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
