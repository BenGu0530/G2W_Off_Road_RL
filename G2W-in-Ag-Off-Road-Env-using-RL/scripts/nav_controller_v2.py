"""
Navigation Controller v2 — Pavement-Preferring Path Planner
=============================================================
Same as nav_controller.py (v4 three-tier control) but with a
PATH PLANNING CHANGE: when traveling between two Y positions, the
planner routes via the nearest pavement strip whenever possible,
instead of going directly through off-road.

This lets a terrain-aware policy (Stage B) exploit pavement for
higher speed / lower CoT.

Drop-in replacement for nav_controller.py — same API.
"""

import torch
import time
from field_map import (
    GROUPS, CROP_RANGES, CROP_Y, OFFROAD_Y, INNER_X, TOTAL_X,
    SAFE_CENTERS, PAVEMENT_CENTERS,
    find_group, is_in_crop, nearest_safe_y, get_zone_type,
    X_MIN, X_MAX, INNER_X_MIN, INNER_X_MAX,
)


def nearest_safe_y_toward(y_world: float, target_y: float) -> float:
    if target_y >= y_world:
        candidates = [c for c in SAFE_CENTERS if c >= y_world - 0.5]
        if not candidates:
            candidates = SAFE_CENTERS
    else:
        candidates = [c for c in SAFE_CENTERS if c <= y_world + 0.5]
        if not candidates:
            candidates = SAFE_CENTERS
    return min(candidates, key=lambda c: abs(c - y_world))


def find_pavement_between(y_start: float, y_end: float, tolerance: float = 2.0):
    """Find a pavement strip between y_start and y_end.

    Returns the Y of the best pavement strip, or None if none found.
    'Best' = closest to the midpoint of the path.
    tolerance = how far outside the [y_start, y_end] range to consider.
    """
    y_lo = min(y_start, y_end) - tolerance
    y_hi = max(y_start, y_end) + tolerance
    candidates = [p for p in PAVEMENT_CENTERS if y_lo <= p <= y_hi]
    if not candidates:
        return None
    midpoint = (y_start + y_end) / 2.0
    return min(candidates, key=lambda p: abs(p - midpoint))


class NavController:

    def __init__(self, device="cuda:0"):
        self.device = device
        self.waypoints = []
        self.wp_idx = 0
        self.goal_pos = None
        self.goal_reached = False

        # Waypoint thresholds
        self.wp_threshold = 2.5
        self.goal_threshold = 0.5   # tighter (was 3.0) — must really arrive

        # Velocity params — match training distribution
        self.v_max = 1.2
        self.v_min = 0.3
        self.w_max = 0.8
        self.steer_gain = 2.0

        # Heading error thresholds
        self.err_pivot_thresh = 1.57  # 90°
        self.err_turn_thresh = 0.52   # 30°

        # EMA smoothing
        self.cmd_filt = None
        self.alpha = 0.3

        # Pivot timeout
        self.pivot_start_time = None
        self.pivot_timeout_s = 4.0

        self.debug_counter = 0

    def set_goal(self, goal_pos: torch.Tensor, robot_pos: torch.Tensor):
        self.goal_pos = goal_pos
        self.goal_reached = False
        self.cmd_filt = None
        self.pivot_start_time = None

        if robot_pos.dim() == 1:
            robot_pos = robot_pos.unsqueeze(0)

        self.waypoints = self._plan_path(robot_pos, goal_pos)
        self.wp_idx = 0

        print(f"  [NAV] Path: {len(self.waypoints)} waypoints")
        for i, wp in enumerate(self.waypoints):
            print(f"    WP{i+1}: ({wp[0,0]:.1f}, {wp[0,1]:.1f})")

    def update(self, robot_pos: torch.Tensor, robot_quat: torch.Tensor):
        N = robot_pos.shape[0]

        if self.goal_reached or len(self.waypoints) == 0:
            z = torch.zeros(N, device=self.device)
            return z, z, z, torch.zeros(N, device=self.device)

        pos_2d = robot_pos[:, :2]
        current_wp = self.waypoints[self.wp_idx]

        dist_to_wp = torch.norm(current_wp[0] - pos_2d[0]).item()
        is_last = self.wp_idx >= len(self.waypoints) - 1
        threshold = self.goal_threshold if is_last else self.wp_threshold

        past_plane = False
        if not is_last:
            next_wp = self.waypoints[self.wp_idx + 1]
            wp_to_next = next_wp[0] - current_wp[0]
            robot_to_wp = pos_2d[0] - current_wp[0]
            past_plane = torch.dot(robot_to_wp, wp_to_next).item() > 0

        if dist_to_wp < threshold or past_plane:
            if is_last:
                self.goal_reached = True
                z = torch.zeros(N, device=self.device)
                return z, z, z, torch.tensor([dist_to_wp] * N, device=self.device)
            else:
                self.wp_idx += 1
                current_wp = self.waypoints[self.wp_idx]
                self.pivot_start_time = None
                print(f"  [NAV] -> WP{self.wp_idx+1}/{len(self.waypoints)}: "
                      f"({current_wp[0,0]:.1f}, {current_wp[0,1]:.1f})")

        target = current_wp[0].clone()

        robot_y = pos_2d[0, 1].item()
        wp_y = current_wp[0, 1].item()
        if is_in_crop(robot_y):
            safe_y = nearest_safe_y_toward(robot_y, wp_y)
            target[1] = torch.tensor(safe_y, device=self.device, dtype=target.dtype)

        robot_x = pos_2d[0, 0].item()
        boundary_active = False
        if robot_x > X_MAX - 0.3:
            target[0] = torch.tensor(robot_x - 1.0, device=self.device, dtype=target.dtype)
            boundary_active = True
        elif robot_x < X_MIN + 0.3:
            target[0] = torch.tensor(robot_x + 1.0, device=self.device, dtype=target.dtype)
            boundary_active = True

        to_target = target - pos_2d[0]
        dist = torch.norm(to_target).item()

        w, x, y, z_q = robot_quat[0, 0], robot_quat[0, 1], robot_quat[0, 2], robot_quat[0, 3]
        robot_yaw = torch.atan2(
            2.0 * (w * z_q + x * y),
            1.0 - 2.0 * (y ** 2 + z_q ** 2)
        )
        target_yaw = torch.atan2(to_target[1], to_target[0])
        err = torch.atan2(
            torch.sin(target_yaw - robot_yaw),
            torch.cos(target_yaw - robot_yaw)
        )
        err_abs = abs(err.item())

        mode = "fwd"
        if err_abs > self.err_pivot_thresh:
            mode = "PIV"
            vx_val = self.v_min
            wz_val = self.w_max if err.item() > 0 else -self.w_max
            self.cmd_filt = torch.tensor([vx_val, 0.0, wz_val], device=self.device)

            if self.pivot_start_time is None:
                self.pivot_start_time = time.time()
            elif time.time() - self.pivot_start_time > self.pivot_timeout_s:
                print(f"  [NAV] !! Pivot timeout, skipping WP{self.wp_idx+1}")
                if not is_last:
                    self.wp_idx += 1
                    self.pivot_start_time = None

        elif err_abs > self.err_turn_thresh:
            mode = "trn"
            vx_val = self.v_min
            wz_val = float(torch.clamp(
                self.steer_gain * err, min=-self.w_max, max=self.w_max
            ).item())
            self.pivot_start_time = None

            new_cmd = torch.tensor([vx_val, 0.0, wz_val], device=self.device)
            if self.cmd_filt is None:
                self.cmd_filt = new_cmd.clone()
            else:
                self.cmd_filt = (1.0 - self.alpha) * self.cmd_filt + self.alpha * new_cmd

        else:
            mode = "fwd"
            align = torch.clamp(torch.cos(err), min=0.0)
            dist_scale = min(1.0, dist / 2.0)
            vx_desired = self.v_max * align.item() * dist_scale
            vx_val = max(self.v_min, vx_desired)
            wz_val = float(torch.clamp(
                self.steer_gain * err, min=-self.w_max, max=self.w_max
            ).item())
            self.pivot_start_time = None

            new_cmd = torch.tensor([vx_val, 0.0, wz_val], device=self.device)
            if self.cmd_filt is None:
                self.cmd_filt = new_cmd.clone()
            else:
                self.cmd_filt = (1.0 - self.alpha) * self.cmd_filt + self.alpha * new_cmd

        vx = self.cmd_filt[0:1].expand(N)
        vy = self.cmd_filt[1:2].expand(N)
        wz = self.cmd_filt[2:3].expand(N)

        self.debug_counter += 1
        if self.debug_counter % 100 == 0:
            bnd_str = " BND" if boundary_active else ""
            in_crop = " CRP" if is_in_crop(robot_y) else ""
            print(f"  [DBG:{mode}]{bnd_str}{in_crop} "
                  f"yaw={torch.rad2deg(robot_yaw).item():+6.1f}° "
                  f"err={torch.rad2deg(err).item():+6.1f}° "
                  f"cmd=({vx[0].item():+.2f},{vy[0].item():+.2f},{wz[0].item():+.2f}) "
                  f"pos=({pos_2d[0,0].item():+.1f},{pos_2d[0,1].item():+.1f}) "
                  f"tgt=({target[0].item():+.1f},{target[1].item():+.1f}) "
                  f"wp={self.wp_idx+1}/{len(self.waypoints)} wpd={dist_to_wp:.1f}")

        return vx, vy, wz, torch.tensor([dist_to_wp] * N, device=self.device)

    def is_goal_reached(self) -> bool:
        return self.goal_reached

    def get_current_wp(self):
        if self.wp_idx < len(self.waypoints):
            return self.waypoints[self.wp_idx]
        return None

    # ── Path Planning (WITH PAVEMENT PREFERENCE) ────────────────────────

    def _plan_path(self, start_pos, goal_pos):
        sx, sy = start_pos[0, 0].item(), start_pos[0, 1].item()
        gx, gy = goal_pos[0, 0].item(), goal_pos[0, 1].item()

        # Approach point: offroad strip adjacent to crop
        approach_below = gy - (CROP_Y / 2.0 + OFFROAD_Y / 2.0)
        approach_above = gy + (CROP_Y / 2.0 + OFFROAD_Y / 2.0)
        if abs(sy - approach_below) < abs(sy - approach_above):
            approach_y = approach_below
        else:
            approach_y = approach_above

        start_group = find_group(sy)
        goal_group = find_group(approach_y)

        if start_group == goal_group:
            raw_waypoints = self._direct_path_via_pavement(sx, sy, gx, approach_y)
        else:
            raw_waypoints = self._edge_route_via_pavement(
                sx, sy, gx, approach_y, start_group, goal_group
            )

        waypoints = []
        for (wx, wy) in raw_waypoints:
            waypoints.append(torch.tensor([[wx, wy]], device=self.device, dtype=torch.float32))

        if len(waypoints) > 1:
            filtered = [waypoints[0]]
            for wp in waypoints[1:]:
                if torch.norm(wp - filtered[-1]).item() > 1.0:
                    filtered.append(wp)
            waypoints = filtered if filtered else [waypoints[-1]]

        return waypoints

    def _direct_path_via_pavement(self, sx, sy, gx, gy):
        """Within same group, prefer routing via pavement if available.

        Strategy:
        - If pavement exists between sy and gy, detour via it:
            1. Go from start to pavement at sx
            2. Travel along pavement to gx
            3. Exit pavement to approach at gy
        - Otherwise, fall back to direct diagonal.
        """
        pavement_y = find_pavement_between(sy, gy, tolerance=1.5)

        if pavement_y is not None and abs(pavement_y - sy) > 0.5 and abs(pavement_y - gy) > 0.5:
            # Pavement detour
            points = []
            # Step 1: short hop to pavement at current X
            points.append((sx, pavement_y))
            # Step 2: travel along pavement, interpolated every ~2m
            n_steps = max(int(abs(gx - sx) / 2.0), 1)
            for i in range(1, n_steps + 1):
                t = i / n_steps
                px = sx + t * (gx - sx)
                points.append((px, pavement_y))
            # Step 3: exit pavement to approach Y
            points.append((gx, gy))
            return points

        # Fallback: diagonal direct path with intermediate points
        points = []
        dist = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
        n_points = max(int(dist / 2.0), 2)
        for i in range(1, n_points + 1):
            t = i / n_points
            px = sx + t * (gx - sx)
            py = sy + t * (gy - sy)
            if is_in_crop(py):
                py = nearest_safe_y_toward(py, gy)
            points.append((px, py))
        return points

    def _edge_route_via_pavement(self, sx, sy, gx, approach_y,
                                  start_group, goal_group):
        """Cross between groups via field edge, then re-enter via pavement
        if a pavement strip is available near approach_y.
        """
        points = []

        # Edge X: in outer offroad strip, safe distance from boundary
        if sx < 0:
            edge_x = -11.5
        else:
            edge_x = 11.5

        # Step 1: go to edge in current corridor
        safe_start_y = nearest_safe_y(sy)
        points.append((edge_x, safe_start_y))

        # Step 2: walk along edge toward approach Y
        dy = approach_y - safe_start_y
        n_steps = max(int(abs(dy) / 2.0), 1)
        for i in range(1, n_steps + 1):
            t = i / n_steps
            iy = safe_start_y + t * dy
            points.append((edge_x, iy))

        # Step 3: come back from edge to goal X.
        # PAVEMENT PREFERENCE: if there's a pavement strip near approach_y,
        # re-enter via pavement instead of directly going to approach_y.
        pavement_y = find_pavement_between(approach_y, approach_y, tolerance=2.5)

        if pavement_y is not None and abs(pavement_y - approach_y) > 0.5:
            # Go from edge to (gx, pavement_y) via pavement, then to approach_y
            # Step 3a: enter pavement at edge_x
            points.append((edge_x, pavement_y))
            # Step 3b: travel along pavement to gx
            dx = gx - edge_x
            n_steps_x = max(int(abs(dx) / 2.0), 1)
            for i in range(1, n_steps_x + 1):
                t = i / n_steps_x
                ix = edge_x + t * dx
                points.append((ix, pavement_y))
            # Step 3c: exit pavement to approach_y
            points.append((gx, approach_y))
        else:
            # No helpful pavement — direct route
            dx = gx - edge_x
            n_steps_x = max(int(abs(dx) / 2.0), 1)
            for i in range(1, n_steps_x + 1):
                t = i / n_steps_x
                ix = edge_x + t * dx
                points.append((ix, approach_y))

        return points