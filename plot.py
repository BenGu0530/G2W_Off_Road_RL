"""
Poster Figures Generator for Go2-W Agricultural Demo
======================================================
Generates 4 publication-quality PNG figures from demo comparison CSVs:

  fig1_terrain_comparison.png   — Block 1: speed + CoT by terrain (pavement vs offroad)
  fig2_shallow_vs_rough.png     — Block 3: how terrain difficulty shifts policy behavior
  fig3_trajectory_topdown.png   — Top-down bird's-eye view of robot path over the field
  fig4_power_timeline.png       — Power draw over time, color-coded by terrain

Usage:
    pip install pandas matplotlib numpy
    python make_poster_figures.py

Input files (must be in same directory):
    comparison_data_walk_shallow.csv   — terrain noise 0.01-0.10
    comparison_data_walk_rough.csv     — terrain noise 0.05-0.18

Output: 4 PNG files at 300 DPI, ready to drop into PowerPoint / Illustrator.

Tweak:
    - POSTER_FONT_SIZE    : change base font size if poster is very large
    - COLOR_PAVEMENT etc. : change palette to match your poster theme
    - SHALLOW_CSV, ROUGH_CSV : rename if your CSVs are named differently
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — tweak these
# ═══════════════════════════════════════════════════════════════════════════

SHALLOW_CSV = "comparison_data_walk_shallow.csv"
ROUGH_CSV   = "comparison_data_walk_rough.csv"

POSTER_FONT_SIZE = 14
FIG_DPI = 300

# Color palette
COLOR_PAVEMENT = "#4A90E2"   # blue
COLOR_OFFROAD  = "#C97B4D"   # terracotta / dirt
COLOR_CROP     = "#E57373"   # soft red (matches Isaac Sim crop blocks)
COLOR_SHALLOW  = "#7CB342"   # green (easy)
COLOR_ROUGH    = "#D32F2F"   # dark red (hard)
COLOR_TRAJ     = "#1A237E"   # navy for trajectory
COLOR_GOAL     = "#FFD600"   # yellow for goal markers

# Field layout (must match field_map.py / ag_env_cfg.py)
TOTAL_X = 24.0
TOTAL_Y = 42.0
BORDER_X = 2.0
INNER_X = 20.0
BORDER_Y = 2.0
CROP_Y = 2.0
OFFROAD_Y = 2.0
PAVEMENT_Y = 2.0

# Goals (approximate — match demo_compare.generate_goals with seed 42)
CROP_RANGES_Y = [(-17, -15), (-9, -7), (-1, 1), (7, 9), (15, 17)]
CROP_CENTERS_Y = [(s + e) / 2.0 for s, e in CROP_RANGES_Y]

# ═══════════════════════════════════════════════════════════════════════════
# Matplotlib global style
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size":        POSTER_FONT_SIZE,
    "font.family":      "sans-serif",
    "axes.linewidth":   1.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelweight":  "bold",
    "axes.titleweight":  "bold",
    "figure.dpi":        100,
    "savefig.dpi":       FIG_DPI,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.2,
})


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load both CSVs and filter out 'crop' terrain rows (we don't analyze those).

    The 'crop' rows appear when the robot accidentally enters a crop zone —
    they're rare and misleading for per-terrain stats, so we drop them.
    """
    s = pd.read_csv(SHALLOW_CSV)
    r = pd.read_csv(ROUGH_CSV)
    # Drop crop rows for clean 2-terrain analysis
    s = s[s["terrain"].isin(["pavement", "offroad"])].reset_index(drop=True)
    r = r[r["terrain"].isin(["pavement", "offroad"])].reset_index(drop=True)
    return s, r


def compute_per_terrain_cot(df):
    """Compute Cost of Transport separately for each terrain type.

    CoT = Energy / (m * g * distance)
    We re-compute per terrain from power * dt / distance_increment, because the
    stored cot_cumulative is global.
    """
    ROBOT_MASS = 19.5
    GRAVITY    = 9.81
    DT         = 0.02  # matches env.step_dt

    results = {}
    # Compute distance increment per row
    df = df.copy()
    df["dist_step"] = df["distance_cumulative"].diff().fillna(0.0).clip(lower=0)

    for terrain in ["pavement", "offroad"]:
        sub = df[df["terrain"] == terrain]
        total_energy = (sub["power"] * DT).sum()
        total_dist = sub["dist_step"].sum()
        if total_dist > 0.1:
            cot = total_energy / (ROBOT_MASS * GRAVITY * total_dist)
        else:
            cot = np.nan
        results[terrain] = {
            "cot":       cot,
            "mean_speed": sub["speed"].mean(),
            "mean_power": sub["power"].mean(),
            "time_s":    len(sub) * DT,
            "distance":  total_dist,
            "n":         len(sub),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Block 1: Pavement vs Off-road (single run, shallow)
# ═══════════════════════════════════════════════════════════════════════════

def fig1_terrain_comparison(s_df):
    """Bar plots comparing pavement and offroad within the shallow run."""
    stats = compute_per_terrain_cot(s_df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    terrains = ["pavement", "offroad"]
    labels   = ["Pavement", "Off-road"]
    colors   = [COLOR_PAVEMENT, COLOR_OFFROAD]

    # --- Speed ---
    speeds = [stats[t]["mean_speed"] for t in terrains]
    axes[0].bar(labels, speeds, color=colors, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Mean speed (m/s)")
    axes[0].set_title("(a) Locomotion Speed", pad=15)
    axes[0].set_ylim(0, max(speeds) * 1.25)
    for i, v in enumerate(speeds):
        axes[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")

    # --- CoT ---
    cots = [stats[t]["cot"] for t in terrains]
    axes[1].bar(labels, cots, color=colors, edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("Cost of Transport")
    axes[1].set_title("(b) Energy Efficiency", pad=15)
    axes[1].set_ylim(0, max(cots) * 1.25)
    for i, v in enumerate(cots):
        axes[1].text(i, v + max(cots) * 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    # --- Time spent ---
    times = [stats[t]["time_s"] for t in terrains]
    total = sum(times)
    pcts = [t / total * 100 for t in times]
    axes[2].bar(labels, pcts, color=colors, edgecolor="black", linewidth=1.5)
    axes[2].set_ylabel("Time spent (%)")
    axes[2].set_title("(c) Traversal Distribution", pad=15)
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(pcts):
        axes[2].text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")

    fig.suptitle("Go2-W locomotion adapts to terrain type",
                 fontsize=POSTER_FONT_SIZE + 3, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("fig1_terrain_comparison.png")
    plt.close()
    print("[OK] fig1_terrain_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Block 3: Shallow vs Rough terrain (policy behavior shifts)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_shallow_vs_rough(s_df, r_df):
    """Compare overall metrics between shallow and rough terrain runs."""
    s_stats = compute_per_terrain_cot(s_df)
    r_stats = compute_per_terrain_cot(r_df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(2)   # pavement, offroad
    width = 0.35
    terrains = ["pavement", "offroad"]
    labels   = ["Pavement", "Off-road"]

    # --- Speed grouped bar ---
    s_speeds = [s_stats[t]["mean_speed"] for t in terrains]
    r_speeds = [r_stats[t]["mean_speed"] for t in terrains]
    axes[0].bar(x - width/2, s_speeds, width, label="Shallow (0.01-0.10m)",
                color=COLOR_SHALLOW, edgecolor="black", linewidth=1.2)
    axes[0].bar(x + width/2, r_speeds, width, label="Rough (0.05-0.18m)",
                color=COLOR_ROUGH, edgecolor="black", linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean speed (m/s)")
    axes[0].set_title("(a) Speed degradation", pad=15)
    # Place legend below the title, outside bars
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.9,
                   fontsize=POSTER_FONT_SIZE - 3)
    axes[0].set_ylim(0, max(max(s_speeds), max(r_speeds)) * 1.25)
    for i, (sv, rv) in enumerate(zip(s_speeds, r_speeds)):
        axes[0].text(i - width/2, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=POSTER_FONT_SIZE - 2)
        axes[0].text(i + width/2, rv + 0.02, f"{rv:.2f}", ha="center", fontsize=POSTER_FONT_SIZE - 2)

    # --- CoT grouped bar ---
    s_cots = [s_stats[t]["cot"] for t in terrains]
    r_cots = [r_stats[t]["cot"] for t in terrains]
    axes[1].bar(x - width/2, s_cots, width, label="Shallow",
                color=COLOR_SHALLOW, edgecolor="black", linewidth=1.2)
    axes[1].bar(x + width/2, r_cots, width, label="Rough",
                color=COLOR_ROUGH, edgecolor="black", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Cost of Transport")
    axes[1].set_title("(b) Energy cost rises", pad=15)
    axes[1].legend(loc="upper left", frameon=False, fontsize=POSTER_FONT_SIZE - 2)
    for i, (sv, rv) in enumerate(zip(s_cots, r_cots)):
        axes[1].text(i - width/2, sv + max(r_cots) * 0.02, f"{sv:.2f}",
                     ha="center", fontsize=POSTER_FONT_SIZE - 2)
        axes[1].text(i + width/2, rv + max(r_cots) * 0.02, f"{rv:.2f}",
                     ha="center", fontsize=POSTER_FONT_SIZE - 2)

    # --- Mean power grouped bar ---
    s_pwrs = [s_stats[t]["mean_power"] for t in terrains]
    r_pwrs = [r_stats[t]["mean_power"] for t in terrains]
    axes[2].bar(x - width/2, s_pwrs, width, label="Shallow",
                color=COLOR_SHALLOW, edgecolor="black", linewidth=1.2)
    axes[2].bar(x + width/2, r_pwrs, width, label="Rough",
                color=COLOR_ROUGH, edgecolor="black", linewidth=1.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Mean mechanical power (W)")
    axes[2].set_title("(c) Policy effort scales with difficulty", pad=15)
    axes[2].legend(loc="upper left", frameon=False, fontsize=POSTER_FONT_SIZE - 2)
    for i, (sv, rv) in enumerate(zip(s_pwrs, r_pwrs)):
        axes[2].text(i - width/2, sv + max(r_pwrs) * 0.02, f"{sv:.0f}",
                     ha="center", fontsize=POSTER_FONT_SIZE - 2)
        axes[2].text(i + width/2, rv + max(r_pwrs) * 0.02, f"{rv:.0f}",
                     ha="center", fontsize=POSTER_FONT_SIZE - 2)

    fig.suptitle("Rough terrain forces the policy to engage legs more actively",
                 fontsize=POSTER_FONT_SIZE + 3, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("fig2_shallow_vs_rough.png")
    plt.close()
    print("[OK] fig2_shallow_vs_rough.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Top-down trajectory over the field
# ═══════════════════════════════════════════════════════════════════════════

def derive_goal_positions(df):
    """Infer goal positions from the CSV.

    For each goal_idx, find the row where goal_dist is minimum — that's the
    robot's closest approach to that goal. Goal Y is at the crop row center;
    goal X is inferred from geometry:
        goal_dist^2 = (goal_x - robot_x)^2 + (goal_y - robot_y)^2
    We pick whichever sign of dx gives a goal_x closer to the robot's X
    (approach is roughly perpendicular to the crop row).
    """
    crop_centers_y = [cy for cy in CROP_CENTERS_Y]
    goals = []
    for gi in sorted(df["goal_idx"].unique()):
        sub = df[df["goal_idx"] == gi]
        if len(sub) == 0:
            continue
        min_row = sub.loc[sub["goal_dist"].idxmin()]
        cy = crop_centers_y[int(gi)]
        robot_x = min_row["x"]
        robot_y = min_row["y"]
        d = min_row["goal_dist"]
        dx2 = d ** 2 - (robot_y - cy) ** 2
        if dx2 < 0:
            # numerical edge case — fall back to robot's X
            goal_x = robot_x
        else:
            dx = np.sqrt(dx2)
            # Pick the option closer to robot X (tiny offset)
            goal_x = robot_x + dx if dx < 1.0 else robot_x
        goals.append((goal_x, cy))
    return goals


def draw_field(ax):
    """Draw the field layout as colored background rectangles.

    Matches ag_terrain.py: crop barriers only exist in the INNER_X region
    (|x| < INNER_X/2 = 10). The left/right BORDER_X strips are off-road
    on every row, even where the crop row runs through the center.
    """
    x_min = -TOTAL_X / 2.0   # -12
    x_max = +TOTAL_X / 2.0   # +12
    ix_min = -INNER_X / 2.0  # -10
    ix_max = +INNER_X / 2.0  # +10

    color_map = {
        "offroad":  COLOR_OFFROAD,
        "pavement": COLOR_PAVEMENT,
        "crop":     COLOR_CROP,
    }
    alpha_map = {"offroad": 0.25, "pavement": 0.35, "crop": 0.55}

    def add_rect(x0, x1, y0, y1, kind):
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color_map[kind], alpha=alpha_map[kind],
            edgecolor="none", zorder=1,
        )
        ax.add_patch(rect)

    # Walk through Y bands from bottom to top, replicating ZONES in ag_terrain.py
    y = -TOTAL_Y / 2.0
    zones = []
    zones.append(("offroad", y, y + BORDER_Y)); y += BORDER_Y
    for i in range(5):
        zones.append(("offroad", y, y + OFFROAD_Y)); y += OFFROAD_Y
        zones.append(("crop",    y, y + CROP_Y));    y += CROP_Y
        zones.append(("offroad", y, y + OFFROAD_Y)); y += OFFROAD_Y
        if i < 4:
            zones.append(("pavement", y, y + PAVEMENT_Y)); y += PAVEMENT_Y
    zones.append(("offroad", y, y + BORDER_Y))

    for ztype, ys, ye in zones:
        if ztype == "crop":
            # Inner region is crop; left + right border strips are offroad
            add_rect(x_min,  ix_min, ys, ye, "offroad")   # left border
            add_rect(ix_min, ix_max, ys, ye, "crop")      # inner crop
            add_rect(ix_max, x_max,  ys, ye, "offroad")   # right border
        elif ztype == "pavement":
            # Pavement is also inner-only; borders are offroad
            add_rect(x_min,  ix_min, ys, ye, "offroad")
            add_rect(ix_min, ix_max, ys, ye, "pavement")
            add_rect(ix_max, x_max,  ys, ye, "offroad")
        else:
            add_rect(x_min, x_max, ys, ye, ztype)


def fig3_trajectory(s_df):
    """Top-down view of the robot's path over the field, using the shallow run."""
    fig, ax = plt.subplots(figsize=(7, 11))  # tall & narrow, matches field aspect

    draw_field(ax)

    # Plot trajectory colored by speed
    x = s_df["x"].values
    y = s_df["y"].values
    spd = s_df["speed"].values

    # Segment line colored by speed
    from matplotlib.collections import LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis",
                        norm=plt.Normalize(0, spd.max()),
                        linewidth=2.5, zorder=3)
    lc.set_array(spd[:-1])
    ax.add_collection(lc)

    # Mark spawn
    ax.plot(x[0], y[0], marker="o", markersize=14, color="white",
            markeredgecolor="black", markeredgewidth=2, zorder=5,
            label="Spawn")

    # Mark goal positions (derived from CSV, not hardcoded)
    goal_positions = derive_goal_positions(s_df)

    for i, (gx, gy) in enumerate(goal_positions):
        ax.plot(gx, gy, marker="*", markersize=22, color=COLOR_GOAL,
                markeredgecolor="black", markeredgewidth=1.5, zorder=6)
        ax.annotate(f"G{i+1}", (gx, gy), xytext=(8, 0),
                    textcoords="offset points", fontsize=POSTER_FONT_SIZE,
                    fontweight="bold", zorder=6)

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Speed (m/s)", rotation=270, labelpad=22)

    # Legend patches
    legend_patches = [
        patches.Patch(facecolor=COLOR_PAVEMENT, alpha=0.35, label="Pavement"),
        patches.Patch(facecolor=COLOR_OFFROAD,  alpha=0.25, label="Off-road"),
        patches.Patch(facecolor=COLOR_CROP,     alpha=0.55, label="Crop row"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              bbox_to_anchor=(1.15, 1.0), frameon=False,
              fontsize=POSTER_FONT_SIZE - 2)

    ax.set_xlim(-TOTAL_X / 2 - 1, TOTAL_X / 2 + 1)
    ax.set_ylim(-TOTAL_Y / 2 - 1, TOTAL_Y / 2 + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Robot trajectory through the field\n(5 crop row inspections)",
                 pad=15, fontsize=POSTER_FONT_SIZE + 2)

    plt.tight_layout()
    plt.savefig("fig3_trajectory_topdown.png")
    plt.close()
    print("[OK] fig3_trajectory_topdown.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Power timeline with terrain annotation (rough run, more dramatic)
# ═══════════════════════════════════════════════════════════════════════════

def fig4_power_timeline(r_df):
    """Power draw over time, with terrain bands shaded behind the trace.

    Uses the rough run because it shows bigger swings between terrains.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Smoothed power (50-step window) for readability
    power_smooth = r_df["power"].rolling(window=50, center=True, min_periods=1).mean()
    t = r_df["step"].values * 0.02  # steps -> seconds (assuming dt=0.02)

    # Shade terrain bands behind
    terrain = r_df["terrain"].values
    current = terrain[0]
    start_idx = 0
    for i in range(1, len(terrain)):
        if terrain[i] != current:
            color = {"pavement": COLOR_PAVEMENT,
                     "offroad":  COLOR_OFFROAD,
                     "crop":     COLOR_CROP}.get(current, "gray")
            ax.axvspan(t[start_idx], t[i], color=color, alpha=0.15, zorder=1)
            current = terrain[i]
            start_idx = i
    # Final band
    color = {"pavement": COLOR_PAVEMENT,
             "offroad":  COLOR_OFFROAD,
             "crop":     COLOR_CROP}.get(current, "gray")
    ax.axvspan(t[start_idx], t[-1], color=color, alpha=0.15, zorder=1)

    # Power trace
    ax.plot(t, power_smooth, color="black", linewidth=1.5, zorder=3)

    # Legend for terrain shading
    legend_patches = [
        patches.Patch(facecolor=COLOR_PAVEMENT, alpha=0.25, label="Pavement"),
        patches.Patch(facecolor=COLOR_OFFROAD,  alpha=0.25, label="Off-road"),
        patches.Patch(facecolor=COLOR_CROP,     alpha=0.25, label="Crop (brief incursion)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=False,
              fontsize=POSTER_FONT_SIZE - 2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mechanical power (W)")
    ax.set_title("Instantaneous power draw on rough terrain\n"
                 "(50-step smoothed)",
                 pad=15, fontsize=POSTER_FONT_SIZE + 2)
    ax.set_xlim(0, t.max())
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("fig4_power_timeline.png")
    plt.close()
    print("[OK] fig4_power_timeline.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading CSVs...")
    s_df, r_df = load_data()
    print(f"  Shallow: {len(s_df)} rows")
    print(f"  Rough:   {len(r_df)} rows")
    print()

    fig1_terrain_comparison(s_df)
    fig2_shallow_vs_rough(s_df, r_df)
    fig3_trajectory(s_df)
    fig4_power_timeline(r_df)

    print()
    print("All figures saved at 300 DPI.")
    print("Drop them straight into PowerPoint or Illustrator.")