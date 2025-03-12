import numpy as np
import math
from typing import Tuple, List

def APF(WIDTH: int, HEIGHT: int, q_star: int, num_sensors: int, number_of_drones: int, max_steps: int=100, count: int=0, epsilon: float = 5.0):
    screen_array = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
    path = np.zeros((max_steps, number_of_drones, 2))  # Initialize the path array

    class APFMethods:
        """
            Class containing methods to calculate forces for a potential field algorithm and
            to update the position according to these forces.
            It handles obstacle avoidance and goal attraction using artificial potential fields.
            """

        @staticmethod
        def check_boundaries(ex: int, ey: int, nx: float, ny: float) -> bool:
            """
                   Check if the new coordinates are within the boundaries of the environment.

                   Args:
                       ex (int): Environment width boundary.
                       ey (int): Environment height boundary.
                       nx (float): New x-coordinate.
                       ny (float): New y-coordinate.

                   Returns:
                       bool: True if the new coordinates are within boundaries, False otherwise.
                   """
            return 0 <= nx < ex and 0 <= ny < ey

        @staticmethod
        def check_obstacles(arr: np.ndarray, check_x: int, check_y: int, color: int) -> bool:
            """
                    Check if there is an obstacle at the specified coordinates.

                    Args:
                        arr (np.ndarray): The environment array with obstacles and free space.
                        check_x (int): X-coordinate to check.
                        check_y (int): Y-coordinate to check.
                        color (int): The color channel representing the obstacle.

                    Returns:
                        bool: True if an obstacle is detected, False otherwise.
                    """
            return arr[check_x][check_y][0] != color and arr[check_x][check_y][0] != 0

        @staticmethod
        def dist(sx: int, sy: int, x: int, y: int, theta: float, arr: np.ndarray, q_star_value: int, color_channel: int,
                 search_radius: int) -> tuple[int, int]:
            """
                Calculate the distance to the nearest obstacle from a starting point (sx, sy) in a given direction (theta).

                Args:
                    sx (int): Starting x-coordinate.
                    sy (int): Starting y-coordinate.
                    x (int): Width of the environment.
                    y (int): Height of the environment.
                    theta (float): Direction angle in radians.
                    arr (np.ndarray): The environment array representing the space.
                    q_star_value (int): Maximum distance to search for an obstacle.
                    color_channel (int): Color channel to use for identifying obstacles.
                    search_radius (int): Initial distance to start checking from.

                Returns:
                    tuple[int, int]: (dx, dy) distances from (sx, sy) to the nearest obstacle, or (-1, -1) if no obstacle
                    is found within q_star.
                """
            search_radius = search_radius
            while True:
                if search_radius > q_star_value:
                    return -1, -1  # No obstacle found within q_star

                # Calculate the new position in the direction of theta
                new_x = sx + search_radius * math.cos(theta)
                new_y = sy + search_radius * math.sin(theta)

                # Round to nearest integers for array indexing
                int_new_x, int_new_y = int(round(new_x)), int(round(new_y))

                # Check if the new coordinates are within bounds
                if not APFMethods.check_boundaries(x, y, new_x, new_y):
                    break  # Exit if out of bounds

                # print(f"Checked index: {int_ansx}, {int_ansy}")

                # Check if there's an obstacle at the new position
                if APFMethods.check_obstacles(arr, int_new_x, int_new_y, color_channel):
                    return int_new_x - sx, int_new_y - sy  # Return force if obstacle found

                search_radius += 1

            return -1, -1

        @staticmethod
        def calculate_sensor_positions(center: Tuple[int, int], radius: int,
                                       sensor_num: int, ) -> List[Tuple[int, int]]:
            """
                Calculate the positions of sensors placed evenly around a circle at a specified radius from the center.

                Args:
                    center (Tuple[int, int]): The (x, y) coordinates representing the center of the manipulator.
                    radius (int): The distance from the center to each sensor (i.e., the radius of the circle).
                    sensor_num (int): The number of sensors to place around the circle.

                Returns:
                    List[Tuple[int, int]]: A list of sensor positions as (x, y) tuples, calculated in equal angular intervals.
                """
            sensor_num = sensor_num  # For 60° intervals
            angle_increment = 360 / sensor_num  # Degrees between each sensor
            sensor_positions = []  # Initialize an empty list for sensor positions

            # Calculate each sensor position using polar coordinates and convert to Cartesian
            for i in range(sensor_num):
                angle = np.radians(i * angle_increment)  # Convert the angle to radians
                x = int(center[0] + radius * np.cos(angle))  # X-coordinate
                y = int(center[1] + radius * np.sin(angle))  # Y-coordinate
                sensor_positions.append((x, y))  # Store the sensor position as a tuple

            return sensor_positions

        @staticmethod
        def obstacle_force(arr: np.ndarray, sx: int, sy: int, q_star_value: int, color_channel: int,
                           velocity_dict: dict, search_radius: int, amplification_factor: int, neta: int = 30000000) -> \
                tuple[float, float]:
            """
                Calculate the repulsive force exerted by surrounding obstacles in the environment on the manipulator.

                Args:
                    arr (np.ndarray): A 3D array representing the environment with obstacle locations and free space.
                    sx (int): Start x-coordinate of the manipulator's current position.
                    sy (int): Start y-coordinate of the manipulator's current position.
                    q_star_value (int): Maximum distance within which obstacles exert a repulsive force.
                    color_channel (int): The color channel used to identify obstacles in the environment array.
                    velocity_dict (dict): A dictionary mapping obstacle coordinates to their velocity and direction (theta).
                    search_radius (int): Initial count used in distance calculations.
                    amplification_factor (int): Factor to amplify the repulsive force when an obstacle is within a critical distance.
                    neta (int, optional): Scaling factor for repulsive force calculations (default is 30,000,000).

                Returns:
                    tuple[float, float]: The x and y components of the total repulsive force acting on the manipulator.

                The function computes the repulsive force from obstacles within a given range (q_star) around the manipulator's
                position. It checks 8 directions (in 45-degree intervals) for obstacles, calculates the repulsive force based
                on the obstacle's distance and velocity (if available), and returns the resultant x and y force components.
                """
            force_x, force_y = 0, 0
            neta = neta  # Scaling factor for repulsive force
            x, y, z = arr.shape  # Get the dimensions of the environment

            for i in range(8):  # 8 directions (every 45 degrees)
                (ox, oy) = APFMethods.dist(sx, sy, (x - 1), (y - 1), i * math.pi / 4, arr, q_star_value, color_channel, search_radius)
                theta = i * math.pi / 4

                if ox == -1 or oy == -1:
                    continue

                ox, oy = abs(ox), abs(oy)

                # Check if there's a velocity recorded for the obstacle
                if (ox, oy) in velocity_dict:
                    obst_velocity, theta = velocity_dict[(ox, oy)]  # Get the velocity and theta from the dict
                    ox = ox + obst_velocity * math.sin(theta)
                    oy = oy + obst_velocity * math.cos(theta)  # Modify according to your logic

                effective_distance = math.hypot(abs(ox), abs(oy))

                # Prevent division by zero and ensure effective distance is positive
                effective_distance = max(effective_distance, 1e-6)  # Small value to avoid division by zero

                # Calculate the repulsive force based on effective distance
                f = (neta * (1.0 / q_star_value - 1.0 / effective_distance)) / (effective_distance * effective_distance)

                safe_distance = q_star_value / 2
                amplification_factor = amplification_factor

                if effective_distance < safe_distance:
                    f *= amplification_factor

                force_x += f * math.sin(theta)
                force_y += f * math.cos(theta)

            return force_x, force_y

        @staticmethod
        def sum_obstacle_forces(center: Tuple[int, int], radius: int, arr: np.ndarray, q_star_value: int,
                                color_channel: int, velocity_dict: dict, sensors_num: int, count_num: int,
                                amplification_factor: int = 10) -> \
                Tuple[float, float]:
            """
                Calculate the cumulative obstacle forces acting on a manipulator based on sensor readings around it.

                Args:
                    center (Tuple[int, int]): The (x, y) coordinates of the manipulator's center position.
                    radius (int): The distance from the center at which sensors are positioned.
                    arr (np.ndarray): A 3D array representing the environment, including obstacles and free space.
                    q_star_value (int): The maximum distance within which obstacles exert a significant influence on the manipulator.
                    color_channel (int): The specific color channel used to identify obstacles in the environment for simulation purposes.
                    velocity_dict (dict): A dictionary mapping obstacle coordinates to their velocities, used for calculating obstacle influence.
                    sensors_num (int): The number of sensors placed around the manipulator, determining the density of readings.
                    count_num (int): An initial count used in calculations to help with distance measurements.
                    amplification_factor (int, optional): Factor to amplify the repulsive forces when obstacles are close (default is 10).

                Returns:
                    Tuple[float, float]: The total cumulative obstacle force components in the x and y directions.

                This function calculates the total repulsive forces from obstacles detected by sensors positioned in a circular
                pattern around the manipulator. It first computes the sensor positions and then iterates through each sensor
                to calculate the individual obstacle forces, which are accumulated to provide a net force vector acting on the
                manipulator.
                """
            # Calculate the sensor positions for the current manipulator
            sensor_positions = APFMethods.calculate_sensor_positions(center, radius, sensors_num)

            total_obst_force_x = 0
            total_obst_force_y = 0

            # Loop through each sensor position for the current manipulator
            for sensor_position in sensor_positions:
                # Extract the (x, y) coordinates of the sensor
                x_sensor, y_sensor = sensor_position

                # Calculate obstacle force for the sensor position
                obst_force_x, obst_force_y = APFMethods.obstacle_force(
                    arr,
                    x_sensor,
                    y_sensor,
                    q_star_value,
                    color_channel,
                    velocity_dict,
                    count_num,
                    amplification_factor
                )

                # Accumulate the total obstacle forces in both x and y directions
                total_obst_force_x += obst_force_x
                total_obst_force_y += obst_force_y

            return total_obst_force_x, total_obst_force_y

        @staticmethod
        def goal_force(sx: int, sy: int, dx: int, dy: int, tau: int = 20) -> tuple[float, float]:
            """
                Calculate the attractive force exerted on the manipulator towards a specified goal position.

                Args:
                    sx (int): The x-coordinate of the current position of the manipulator.
                    sy (int): The y-coordinate of the current position of the manipulator.
                    dx (int): The x-coordinate of the target goal position.
                    dy (int): The y-coordinate of the target goal position.
                    tau (int, optional): A scaling factor that adjusts the magnitude of the attractive force (default is 20).

                Returns:
                    tuple[float, float]: The x and y components of the attractive force directed towards the goal.

                This function calculates the attractive force towards a goal by computing the difference in x and y coordinates
                between the current position and the goal position, scaled by the factor tau. The resulting force components
                indicate how much the manipulator should be directed towards the goal in both the x and y directions.
                """
            tau = tau  # Scaling factor for attractive force
            # Apply attractive force directly proportional to the distance
            force_x = ((dx - sx) * tau)
            force_y = ((dy - sy) * tau)

            return force_x, force_y

        @staticmethod
        def determine_direction(arr: np.ndarray, start_x: int, start_y: int, obst_force_x: float, obst_force_y: float,
                                goal_force_x: float, goal_force_y: float, v: int = 10, theta: float = 0,
                                theta_const: float = math.pi * 30 / 180, flx: int = 100000, fly: int = 100000,
                                v_min: int = 1, v_max: int = 20, alpha: float = 0.1) -> tuple[int, int, float, float]:
            """
                Calculate the new position and direction of the manipulator based on the attractive and repulsive forces.

                Args:
                    arr (np.ndarray): The environment array representing the space where the manipulator operates.
                    start_x (int): The current x-coordinate of the manipulator's position.
                    start_y (int): The current y-coordinate of the manipulator's position.
                    obst_force_x (float): The x-component of the repulsive force from surrounding obstacles.
                    obst_force_y (float): The y-component of the repulsive force from surrounding obstacles.
                    goal_force_x (float): The x-component of the attractive force towards the goal.
                    goal_force_y (float): The y-component of the attractive force towards the goal.
                    v (int, optional): The current speed of the manipulator (default is 10).
                    theta (float, optional): The current direction angle of the manipulator in radians (default is 0).
                    theta_const (float, optional): The maximum allowable change in direction angle (default is 30 degrees in radians).
                    flx (int, optional): The maximum force limit in the x-direction (default is 100,000).
                    fly (int, optional): The maximum force limit in the y-direction (default is 100,000).
                    v_min (int, optional): The minimum speed the manipulator can achieve (default is 1).
                    v_max (int, optional): The maximum speed the manipulator can achieve (default is 20).
                    alpha (float, optional): The adjustment factor for speed change (default is 0.1).

                Returns:
                    tuple[int, int, float, float]: The updated x and y coordinates, the new direction angle, and the updated speed.

                This method calculates the new position and direction of the manipulator by:
                - Summing the attractive and repulsive forces to determine the total forces acting on the manipulator.
                - Clamping these forces to specified limits to ensure they remain within a manageable range.
                - Calculating the resultant force's magnitude and adjusting the manipulator's speed accordingly.
                - Determining the new direction based on the angle of the resultant force and updating the angle while constraining it to a maximum allowable change.
                - Updating the position based on the new speed and direction, and ensuring the position remains within the boundaries of the environment.
                """
            # Calculate total forces
            total_force_x = goal_force_x + obst_force_x
            total_force_y = goal_force_y + obst_force_y

            # Bound the total forces within max allowable values
            total_force_x = max(min(total_force_x, flx), -flx)
            total_force_y = max(min(total_force_y, fly), -fly)

            # Compute the resultant force magnitude
            f_total = math.sqrt(total_force_x ** 2 + total_force_y ** 2)

            # Adjust speed based on resultant force
            target_speed = max(v_min, min(int(v_max * math.exp(-f_total / max(flx, fly))), v_max))
            v = v + alpha * (target_speed - v)

            # Calculate direction angle and update theta
            force_angle = math.atan2(total_force_x, total_force_y)
            angle_diff = force_angle - theta
            theta_change = math.atan2(f_total * math.sin(angle_diff), v + f_total * math.cos(angle_diff))
            theta_change = max(min(theta_change, theta_const), -theta_const)
            theta = (theta + theta_change + 2 * math.pi) % (2 * math.pi)

            # Update position
            start_x += int(v * math.sin(theta))
            start_y += int(v * math.cos(theta))

            # Clamp position within boundaries
            max_x, max_y = arr.shape[0] - 1, arr.shape[1] - 1
            start_x = max(0, min(start_x, max_x))
            start_y = max(0, min(start_y, max_y))

            return start_x, start_y, theta, v

        def reached_goal(self, position: Tuple[int, int], goal: Tuple[int, int], epsilon_value: float = 5.0) -> bool:
            """
                Check if the manipulator has reached its goal within a specified distance.

                Args:
                    position (Tuple[int, int]): The current (x, y) coordinates of the manipulator.
                    goal (Tuple[int, int]): The target (x, y) coordinates representing the goal position.
                    epsilon_value (float, optional): The distance threshold within which the goal is considered reached (default is 1.0).

                Returns:
                    bool: True if the manipulator is within the specified distance (epsilon) of the goal, False otherwise.

                This method computes the Euclidean distance between the manipulator's current position and the goal.
                If the distance is less than or equal to the epsilon threshold, the method returns True, indicating that
                the manipulator has effectively reached the goal; otherwise, it returns False.
                """
            # Calculate Euclidean distance between the current position and the goal
            distance = np.linalg.norm(np.array(position) - np.array(goal))
            return distance <= epsilon_value

        def form_circle(self, x: int, y: int, radius: int, arr: np.ndarray, x_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
            """
                Creates a filled circle of a specified radius centered at (x, y) within the environment grid.

                This method utilizes the symmetry properties of circles to optimize the computation by calculating points
                in only one octant of the circle and reflecting them to the other octants. It then ensures that the
                resulting coordinates are within the bounds of the provided environment array.

                Args:
                    x (int): The x-coordinate of the center of the circle.
                    y (int): The y-coordinate of the center of the circle.
                    radius (int): The radius of the circle.
                    arr (np.ndarray): The environment array, used to ensure the coordinates are valid within its bounds.
                    x_offset (int, optional): The initial offset in the x-direction for the circle drawing process (default is 0).

                Returns:
                    Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the x and y coordinates, respectively,
                                                    that form the filled circle within the environment bounds.

                This method implements the Midpoint Circle Algorithm to efficiently compute the points of the circle.
                It first determines the boundary points and then fills in the circle by including all points between
                the edge and the center. The resulting coordinates are filtered to ensure they lie within the limits
                of the given environment array.
                """
            valid_coords = []

            # Use symmetry to calculate points in one octant and reflect to other octants
            def add_symmetric_points(xc, yc, x_offset_value, y_offset_value):
                points = [
                    (xc + x_offset_value, yc + y_offset_value),  # 1st octant
                    (xc - x_offset_value, yc + y_offset_value),  # 2nd octant
                    (xc + x_offset_value, yc - y_offset_value),  # 7th octant
                    (xc - x_offset_value, yc - y_offset_value),  # 8th octant
                    (xc + y_offset_value, yc + x_offset_value),  # 4th octant
                    (xc - y_offset_value, yc + x_offset_value),  # 3rd octant
                    (xc + y_offset_value, yc - x_offset_value),  # 5th octant
                    (xc - y_offset_value, yc - x_offset_value)  # 6th octant
                ]
                valid_coords.extend(points)

            # Midpoint circle algorithm for calculating the edge points
            y_offset = radius
            d = 1 - radius

            # Add initial symmetric points (on the boundary of the circle)
            add_symmetric_points(x, y, x_offset, y_offset)

            while x_offset < y_offset:
                if d < 0:
                    d += 2 * x_offset + 3
                else:
                    d += 2 * (x_offset - y_offset) + 5
                    y_offset -= 1
                x_offset += 1

                # Add all symmetric points for each new offset
                add_symmetric_points(x, y, x_offset, y_offset)

            # Now, fill the circle by including all points between the edge and the center
            for r in range(radius):
                x_offset = 0
                y_offset = r
                d = 1 - r

                # Add symmetric points for each smaller radius
                add_symmetric_points(x, y, x_offset, y_offset)

                while x_offset < y_offset:
                    if d < 0:
                        d += 2 * x_offset + 3
                    else:
                        d += 2 * (x_offset - y_offset) + 5
                        y_offset -= 1
                    x_offset += 1

                    # Add all symmetric points for the smaller radius
                    add_symmetric_points(x, y, x_offset, y_offset)

            # Convert valid_coords to numpy array
            valid_coords = np.array(valid_coords)

            # Ensure the new coordinates are within the bounds of the environment array
            valid_mask = (0 <= valid_coords[:, 0]) & (valid_coords[:, 0] < arr.shape[0]) & \
                         (0 <= valid_coords[:, 1]) & (valid_coords[:, 1] < arr.shape[1])

            return valid_coords[valid_mask, 0].astype(int), valid_coords[valid_mask, 1].astype(int)

        @staticmethod
        def color_area(valid_x: np.ndarray, valid_y: np.ndarray, color: list[int], arr: np.ndarray) -> None:
            """
                Colors the specified coordinates in the environment grid with the given color.

                This method modifies the provided environment array by filling in the areas defined
                by the valid x and y coordinates with the specified color value.

                Args:
                    valid_x (np.ndarray): A 1D array of x-coordinates of the points to color.
                    valid_y (np.ndarray): A 1D array of y-coordinates of the points to color.
                    color (list[int]): A list of integers representing the color value to fill
                                       the specified areas in the environment array.
                    arr (np.ndarray): The environment array to be modified. This should be a
                                      2D array representing the grid where colors are applied.

                Returns:
                    None: This method does not return a value. It directly modifies the input
                          environment array in place.
                """
            # Color the valid coordinates in the environment array
            arr[valid_x, valid_y] = color

    class Manipulator(APFMethods):
        """
            The Manipulator class inherits from APFMethods and represents a manipulator that can navigate
            in an environment towards a goal while avoiding obstacles using artificial potential fields (APF).

            Attributes:
                goal (tuple): The goal position (x, y) that the manipulator needs to reach.
                path (np.ndarray): A 3D array that stores the manipulator's path over multiple steps.
                arr (np.ndarray): The 2D grid representing the environment (obstacles and free space).
                q_star (int): Maximum distance to check for obstacles.
                theta (float): The manipulator's current direction in radians (angle).
                number (int): The identifier of the manipulator (for tracking or distinguishing multiple manipulators).
                reached_goal_bool (bool): A flag indicating whether the manipulator has reached its goal.
                epsilon (float): The small distance threshold to determine when the manipulator has effectively reached the goal.
                radius (int): The radius of the manipulator's circular representation.
                sensor_positions (np.ndarray): The positions of sensors for detecting obstacles.
                Kp (float): Proportional gain for PID control.
                Ki (float): Integral gain for PID control.
                Kd (float): Derivative gain for PID control.
                previous_error (np.ndarray): The previous error vector (x, y) used for derivative calculation.
                integral_error (np.ndarray): The accumulated error vector (x, y) used for integral calculation.
            """

        def __init__(self, start: Tuple[int, int], goal: Tuple[int, int], arr: np.ndarray, q_star_value: int, radius: int,
                 serial_number: int, number_of_drones: int, path: np.ndarray, theta: float = 0,
                 epsilon_value: float = 5.0, Kp: float = 0.01, Ki: float = 0.001, Kd: float = 0.0002) -> None:
            """
                    Initializes the Manipulator object with the starting position, goal, environment, and other parameters.

                    Args:
                        start (tuple): The starting position (x, y) of the manipulator.
                        goal (tuple): The goal position (x, y) of the manipulator.
                        arr (np.ndarray): The 2D array representing the environment with obstacles and free space.
                        radius (int): The radius of the manipulator's circular representation.
                        serial_number (int): The identifier for the manipulator.
                        number_of_drones (int): The total number of drones in the simulation.
                        path (np.ndarray): A 3D array to store the path taken by the manipulator over time.
                        q_star_value (int, optional): Maximum distance to check for obstacles (default is 30).
                        theta (float, optional): The initial angle of movement (default is 0).
                        epsilon_value (float, optional): The small threshold distance for determining when the goal is reached (default is 5).
                        Kp (float, optional): Proportional gain for PID control (default is 0.01).
                        Ki (float, optional): Integral gain for PID control (default is 0.001).
                        Kd (float, optional): Derivative gain for PID control (default is 0.0002).
                    """
            self.center: Tuple[int, int] = start  # Set the initial position
            self.goal: Tuple[int, int] = goal  # Set the goal position
            self.path: np.ndarray = path  # Initialize the path array
            self.number_of_drones = number_of_drones
            self.current_step: int = 0  # Track current step index

            self.arr: np.ndarray = arr  # Environment array where obstacles and free space are defined
            self.q_star: int = q_star_value  # Maximum distance to check for obstacles
            self.theta: float = theta  # Initial angle of movement
            self.number: int = serial_number  # Identifier for this manipulator
            self.reached_goal_bool: bool = False  # Flag to check if the goal has been reached
            self.epsilon: float = epsilon_value  # Small threshold distance for determining when the goal is reached
            self.velocity_dict = {}  # Store velocities of drones

            # Circle parameters
            self.radius: int = radius  # Define the radius of the drone's circular representation
            self.circle_points: tuple[np.ndarray, np.ndarray] = self.form_circle(self.center[0], self.center[1],
                                                                                 self.radius,
                                                                                 arr)  # Generate the circle points
            self.sensor_positions = self.calculate_sensor_positions(self.center, self.radius, num_sensors)

            # PID variables
            self.Kp: float = Kp  # Proportional gain
            self.Ki: float = Ki  # Integral gain
            self.Kd: float = Kd  # Derivative gain
            self.previous_error: np.ndarray = np.array([0.0, 0.0])  # Previous error
            self.integral_error: np.ndarray = np.array([0.0, 0.0])  # Accumulated error

        def move(self, arr: np.ndarray, color: int) -> Tuple[int, int]:
            """
                    Moves the manipulator towards the goal while avoiding obstacles using artificial potential fields.

                    Args:
                        arr (np.ndarray): The environment array containing obstacles and free space.
                        color (int): The color or marker to indicate the manipulator's influence on the environment
                                     (for rendering).

                    Returns:
                        tuple: The new position (x, y) of the manipulator after moving.
                    """

            # Unpack current position (sx, sy) and goal position (dx, dy)
            start_x, start_y = self.center
            destination_x, destination_y = self.goal

            self.sensor_positions = self.calculate_sensor_positions(self.center, self.radius, num_sensors)

            if self.reached_goal((start_x, start_y), (destination_x, destination_y), self.epsilon):
                self.reached_goal_bool = True
                return self.center

            # Calculate obstacle repulsion forces
            obst_force_x, obst_force_y = self.sum_obstacle_forces(self.center, self.radius, self.arr, self.q_star,
                                                                  color,
                                                                  self.velocity_dict, num_sensors, count)

            # Calculate attractive forces towards the goal
            goal_force_x, goal_force_y = self.goal_force(self.center[0], self.center[1], self.goal[0], self.goal[1])

            # Calculate the current error
            current_error = np.array([goal_force_x - start_x, goal_force_y - start_y])

            # PID control
            self.integral_error += current_error  # Update integral of error
            derivative_error = current_error - self.previous_error  # Calculate derivative of error

            # PID output
            pid_output = (self.Kp * current_error) + (self.Ki * self.integral_error) + (self.Kd * derivative_error)

            # Update previous error
            self.previous_error = current_error

            max_velocity = 3.0  # Maximum allowable change in position per step
            if np.linalg.norm(pid_output) > max_velocity:
                pid_output = pid_output / np.linalg.norm(pid_output) * max_velocity

            # Apply PID output to determine new velocities
            start_x += pid_output[0]  # Update sx based on PID output for x
            start_y += pid_output[1]  # Update sy based on PID output for y

            # Determine the new direction using APF methods
            start_x, start_y, self.theta, v = self.determine_direction(
                arr=arr, start_x=start_x, start_y=start_y,
                obst_force_x=obst_force_x, obst_force_y=obst_force_y, goal_force_x=goal_force_x,
                goal_force_y=goal_force_y, theta=self.theta
            )

            self.velocity_dict[(start_x, start_y)] = (v, self.theta)

            # Update the manipulator's position with the new coordinates
            self.center = (start_x, start_y)
            self.sensor_positions = self.calculate_sensor_positions(self.center, self.radius, num_sensors)

            # Append the current position to the path array
            if self.current_step < self.path.shape[0]:  # Prevent overflow
                print(f"Added center value for {self.number + 1}: {self.center}")
                self.path[self.current_step, self.number - 1, 0] = self.center[0]  # x-coordinate
                self.path[self.current_step, self.number - 1, 1] = self.center[1]  # y-coordinate
                # print(self.path.shape)
                self.current_step += 1  # Move to the next step
            else:
                # Optionally expand the path if needed
                self.path = np.vstack((self.path, np.zeros((1, self.number_of_drones, 2))))  # Add a new row

            return self.center

        def color_circular_area(self, color: list[int], x: int, y: int) -> None:
            """
            Colors the area of influence of the manipulator in the environment.

            Args:
                color (list[int]): The color or marker to use in the environment array.
                x (int): The x-coordinate of the area center (current manipulator position).
                y (int): The y-coordinate of the area center (current manipulator position).
            """
            valid_x, valid_y = self.form_circle(x=x, y=y, radius=self.q_star, arr=screen_array)
            self.color_area(valid_x=valid_x, valid_y=valid_y, color=color, arr=screen_array)

    class Simulation(APFMethods):
        """
            The Simulation class manages the environment and the movement of multiple manipulators
            within it, checking for collisions and coloring areas of influence based on their positions.

            Attributes:
                manipulators (list): A list of Manipulator objects that will move within the environment.
                arr (np.ndarray): A 2D array representing the environment where manipulators operate.
                colors (np.ndarray): An array of colors assigned to each manipulator for rendering purposes.
            """

        def __init__(self, manipulators: List['Manipulator'], arr: np.ndarray) -> None:
            """
                    Initializes the Simulation object with the given manipulators and environment array.

                    Args:
                        manipulators (list): A list of Manipulator objects to be included in the simulation.
                        arr (np.ndarray): The 2D array representing the environment where the simulation occurs.
                    """
            self.manipulators: List['Manipulator'] = manipulators
            self.arr: np.ndarray = arr
            self.colors: np.ndarray = np.arange(0, 255, max(1, 255 // len(self.manipulators)))

        def run(self) -> None:
            # Step 1: Store the old positions (sensor positions) and assign colors to each manipulator
            old_positions = []  # List to store old sensor positions for all manipulators
            new_positions = []  # List to store new sensor positions for all manipulators
            colors = []  # List to store the colors for each manipulator

            # Iterate through each manipulator to store old positions and calculate new positions
            for manipulator in self.manipulators:
                # Store old sensor positions
                old_sensors = manipulator.sensor_positions
                old_positions.append(old_sensors)

                # Assign a color based on manipulator number
                color_value = int(self.colors[manipulator.number - 1])  # Red scale for visualization
                color = [color_value, 0, 0]
                colors.append(color)  # Store the color for later

                # If the manipulator has not moved yet, color its sensors' areas of influence
                if len(manipulator.path) == 0:
                    for sensor_x, sensor_y in old_sensors:
                        manipulator.color_circular_area(color, sensor_x, sensor_y)
                # Color the initial sensor positions

                # Move the manipulator (this computes the new position and updates sensor positions)
                manipulator.move(self.arr, color_value)

                # Store the new sensor positions after the move
                new_sensors = manipulator.sensor_positions
                new_positions.append(new_sensors)

            # Step 2: Clear the old sensor positions and update the new ones
            for i, manipulator in enumerate(self.manipulators):
                # Clear the old sensor positions in the array (set them to black)
                for sensor_x, sensor_y in old_positions[i]:
                    manipulator.color_circular_area([0, 0, 0], sensor_x, sensor_y)

                # Apply the new sensor positions in the environment, but only if the manipulator hasn't reached its goal
                for sensor_x, sensor_y in new_positions[i]:
                    manipulator.color_circular_area(colors[i], sensor_x, sensor_y)  # Color the new sensor positions

                # Step 3: Manage the velocity_dict by removing unnecessary values
                if len(manipulator.velocity_dict) > len(self.manipulators):
                    first_key = list(manipulator.velocity_dict.keys())[0]  # Get the first key
                    del manipulator.velocity_dict[first_key]  # Remove the corresponding entry

            # Step 4: Check for collisions after all movements are done
            self.check_collisions()

        def check_collisions(self) -> str:
            """
                Checks for collisions among the manipulators by tracking their positions.

                Returns:
                    str: A message indicating whether any collisions were detected.
                """
            # Initialize a set to track all positions visited by the manipulators
            visited_positions = set()

            # Iterate through each manipulator to check their positions
            for manipulator in self.manipulators:
                start_x, start_y = manipulator.center

                # Get the coordinates forming a circle of radius 1 around the manipulator's position
                valid_x, valid_y = manipulator.circle_points

                # Iterate through all valid positions and check for collisions
                for x, y in zip(valid_x, valid_y):
                    if (x, y) in visited_positions:
                        return "Collision detected among manipulators!"
                    visited_positions.add((x, y))

            return "There is no collision."

    manipulator_test_x_initial_locations = np.arange(50, WIDTH - 50, WIDTH / (number_of_drones + 1))
    manipulator_test_x_final_locations = manipulator_test_x_initial_locations[::-1]

    # This should give you integer values, but casting ensures safety
    manipulators = [Manipulator((30, (int(x_init))), (770, int(x_final)), screen_array, q_star, 5, a,
                                number_of_drones, path, number_of_drones)
                    for a, (x_init, x_final) in
                    enumerate(zip(manipulator_test_x_initial_locations, manipulator_test_x_final_locations))]

    simulation = Simulation(manipulators, screen_array)

    # Initialize the simulation with the manipulators and the environment array
    number = 0
    while True:
        simulation.run()
        if all(manipulator.reached_goal_bool for manipulator in manipulators):
            print(np.array2string(path, threshold=10**9, max_line_width=10**9))
            print(path[-1])
            break
        number += 1

# Example implementation
APF(800, 800, 20, 6, 8, 150)

