from __future__ import annotations
from typing import List, Tuple
import math
from src.models import CarPose, Cone, Path2D

class PathPlanning:
    """Student-implemented path planner.

    You are given the car pose and an array of detected cones, each cone with (x, y, color)
    where color is 0 for yellow (right side) and 1 for blue (left side). The goal is to
    generate a sequence of path points that the car should follow.

    Implement ONLY the generatePath function.
    """
    def __init__(self, car_pose: CarPose, cones: List[Cone]):
        self.car_pose = car_pose
        self.cones = cones

    def generatePath(self) -> Path2D:
        """Return a list of path points (x, y) in world frame.

        Requirements and notes:
        - Cones: color==0 (yellow) are on the RIGHT of the track; color==1 (blue) are on the LEFT.
        - You may be given 2, 1, or 0 cones on each side.
        - Use the car pose (x, y, yaw) to seed your path direction if needed.
        - Return a drivable path that stays between left (blue) and right (yellow) cones.
        - The returned path will be visualized by PathTester.

        The path can contain as many points as you like, but it should be between 5-10 meters,
        with a step size <= 0.5. Units are meters.

        Replace the placeholder implementation below with your algorithm.
        """
        path: Path2D = []
        step_size = 0.5
        min_length = 5.0
        max_length = 10.0

        cx = self.car_pose.x
        cy = self.car_pose.y
        yaw = self.car_pose.yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        unit_left = (-1.0, 0.0)  # Global left
        unit_right = (1.0, 0.0)  # Global right

        # Start path at car position
        path.append((cx, cy))

        # Separate cones
        blue_cones = [c for c in self.cones if c.color == 1]
        yellow_cones = [c for c in self.cones if c.color == 0]

        # Filter cones in front
        def get_proj(c: Cone) -> float:
            dx = c.x - cx
            dy = c.y - cy
            return dx * cos_y + dy * sin_y

        blue_cones = [c for c in blue_cones if get_proj(c) >= 0]
        yellow_cones = [c for c in yellow_cones if get_proj(c) >= 0]
        all_front_cones = blue_cones + yellow_cones

        if not all_front_cones:
            for i in range(1, int(max_length / step_size) + 1):
                dist = i * step_size
                x = cx + dist * cos_y
                y = cy + dist * sin_y
                path.append((x, y))
            return path

        # Sort all front cones by Euclidean distance
        def get_dist(c: Cone) -> float:
            return math.sqrt((c.x - cx)**2 + (c.y - cy)**2)

        all_front_cones.sort(key=get_dist)
        used = set()  # Set of id(cone)
        current_x, current_y = cx, cy

        while all_front_cones and len(path) < int(max_length / step_size) + 1:
            if not all_front_cones:
                break
            cone = all_front_cones.pop(0)
            if id(cone) in used:
                continue

            # Offset 1m based on color (opposite directions)
            if cone.color == 0:  # Yellow, offset left and up to keep on right
                dx, dy = unit_left
                dy_offset = 1.0
            else:  # Blue, offset right and down to keep on left
                dx, dy = unit_right
                dy_offset = -1.0
            offset_x = cone.x + dx * 1.0
            offset_y = cone.y + dy * 1.0 + dy_offset
            path.append((offset_x, offset_y))
            current_x, current_y = offset_x, offset_y
            used.add(id(cone))  

            # Check for opposite-color cone with same x-coordinate
            opposite_cones = yellow_cones if cone.color == 1 else blue_cones
            candidates = [op for op in opposite_cones if id(op) not in used and abs(op.x - cone.x) < 1e-6]
            if candidates:
                paired = min(candidates, key=get_dist)
                mid_x = (cone.x + paired.x) / 2
                mid_y = (cone.y + paired.y) / 2
                path.append((mid_x, mid_y))
                current_x, current_y = mid_x, mid_y
                used.add(id(paired))
            else:
                # Move to next nearest if available
                if all_front_cones:
                    next_cone = all_front_cones[0]
                    if next_cone.color == 0:  # Yellow
                        dx, dy = unit_left
                        dy_offset = 1.0
                    else:  # Blue
                        dx, dy = unit_right
                        dy_offset = -1.0
                    next_offset_x = next_cone.x + dx * 1.0
                    next_offset_y = next_cone.y + dy * 1.0 + dy_offset
                    path.append((next_offset_x, next_offset_y))
                    current_x, current_y = next_offset_x, next_offset_y
                    used.add(id(next_cone))
                    all_front_cones.pop(0)

        # Extend path straight along last direction
        if path:
            last_x, last_y = path[-1]
            prev_x, prev_y = path[-2] if len(path) > 1 else (cx, cy)
            dx = last_x - prev_x
            dy = last_y - prev_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx /= dist
                dy /= dist
            else:
                dx, dy = cos_y, sin_y

            total_length = 0.0
            for i in range(1, len(path)):
                total_length += math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)

            while total_length < min_length and total_length < max_length:
                add_dist = min(step_size, max_length - total_length)
                x = last_x + add_dist * dx
                y = last_y + add_dist * dy
                path.append((x, y))
                total_length += add_dist
                last_x, last_y = x, y

        return path[:int(max_length / step_size) + 1]