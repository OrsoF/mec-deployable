"""README composition; the allocation is an illustration, not a policy result."""
from __future__ import annotations

import numpy as np

from network_ppo.constraints import assess
from network_ppo.paths import PathCatalog
from network_ppo.problem import Solution

NAVY, BLUE, TEAL = "#142d4e", "#2563a6", "#087f82"
GRAY, PALE, LINE = "#52657b", "#f2f6fa", "#c5d1de"


def example_allocation(problem):
    """Build a manual example and verify every demand, including hidden routes."""
    assert [n.id for n in problem.nodes] == [
        "gateway_north", "gateway_south", "edge_north", "edge_south",
        "core_switch", "cloud_dc",
    ]
    assert [s.id for s in problem.services] == ["video_analytics", "machine_telemetry"]
    assert [d.id for d in problem.demands] == [
        "camera_north", "camera_south", "sensors_north", "sensors_south",
    ]
    catalog = PathCatalog(problem, 3)
    chosen_paths = [(0, 2), (1, 3, 4, 2), (0, 2, 4, 3), (1, 3)]
    placement = np.array([2, 3])
    routes = np.array([
        catalog.candidates(d.origin, int(placement[d.service])).index(path)
        for d, path in zip(problem.demands, chosen_paths)
    ])
    solution = Solution(placement, routes, np.full(len(problem.demands), 3))
    report = assess(problem, solution, catalog)
    assert report.feasible, report.violations
    assert list(report.paths) == chosen_paths
    return solution, report


def render_overview(SVG, problem, mobile=False):
    solution, report = example_allocation(problem)
    width, height = (560, 1150) if mobile else (1040, 674)
    s = SVG(width, height, "PPO placement, routing and bandwidth allocation",
            "Offline training produces a saved PPO policy. Inference uses a sequence "
            "of validated edits and exports the best feasible state. A manual, "
            "validated example places video analytics on the north edge and telemetry "
            "on the south edge. Two of four demand routes are highlighted. "
            "This illustration is not a PPO evaluation or a measured policy result.")
    s.text(24, 36, "PPO FOR EDGE NETWORKS", 16, TEAL, 700)
    s.text(24, 74, "Placement, routes, bandwidth", 28 if mobile else 34, weight=700)

    # Training sits outside the request path and feeds the saved policy only.
    tx, ty, tw = (24, 98, 512) if mobile else (364, 100, 312)
    s.rect(tx, ty, tw, 62, "#eaf1fb")
    s.text(tx + 16, ty + 25, "OFFLINE TRAINING", 14, BLUE, 700)
    s.text(tx + 16, ty + 48, "Network scenarios → saved policy", 17, BLUE)
    if mobile:
        stages = [
            (24, 198, 512, "01  Network state", "Resources + traffic via CLI / HTTP"),
            (24, 302, 512, "02  Saved PPO policy", "Sequence of edits + feasibility checks"),
            (24, 406, 512, "03  Decision JSON", "Best feasible state visited"),
        ]
        # A separate blue rail reaches the policy, bypassing the state input.
        s.line(520, 160, 546, 160, BLUE, 2)
        s.line(546, 160, 546, 341, BLUE, 2)
        s.line(546, 341, 538, 341, BLUE, 2, True)
    else:
        stages = [
            (24, 196, 312, "01  Network state", "Resources + traffic · CLI / HTTP"),
            (364, 196, 312, "02  Saved PPO policy", "Validated edits, repeated"),
            (704, 196, 312, "03  Decision JSON", "Best feasible state visited"),
        ]
        s.line(520, 162, 520, 194, BLUE, 2, True)
    for i, (x, y, w, title, detail) in enumerate(stages):
        s.rect(x, y, w, 78, TEAL if i == 1 else PALE)
        s.text(x + 16, y + 30, title, 21, "white" if i == 1 else NAVY, 700)
        s.text(x + 16, y + 57, detail, 17, "white" if i == 1 else GRAY)
        if i < 2:
            if mobile:
                s.line(280, y + 78, 280, y + 102, GRAY, 2, True)
            else:
                s.line(x + w + 4, y + 39, x + w + 24, y + 39, GRAY, 2, True)

    top = 518 if mobile else 310
    panel_width = 512 if mobile else 640
    panel_height = 332 if mobile else 296
    s.rect(24, top, panel_width, panel_height, PALE)
    s.text(44, top + 30, "ILLUSTRATIVE ALLOCATION", 15, GRAY, 700)
    s.text(44, top + 55, "Two of four demand routes highlighted", 16, GRAY)
    if mobile:
        coords = [(120, top + 115), (120, top + 231),
                  (360, top + 115), (360, top + 231),
                  (478, top + 173), (478, top + 300)]
        gw_width, compute_width, transit_width = 144, 150, 80
    else:
        coords = [(120, top + 110), (120, top + 222),
                  (350, top + 110), (350, top + 222),
                  (562, top + 110), (562, top + 222)]
        gw_width, compute_width, transit_width = 144, 152, 132

    # Draw the whole source topology in gray; overlay only two named demands.
    for link in problem.links:
        s.line(*coords[link.source], *coords[link.target], LINE, 2, dashed=True)
    highlighted = [(0, BLUE), (3, TEAL)]
    for demand_index, color in highlighted:
        path = report.paths[demand_index]
        assert len(path) == 2
        a, b = (coords[i] for i in path)
        s.line(a[0] + gw_width / 2, a[1], b[0] - compute_width / 2 - 6,
               b[1], color, 4, True)
        s.pill((a[0] + b[0]) / 2, a[1] - 16,
               f"{solution.quantities[demand_index]:g} Mbps", color, 16)

    for i, (x, y) in enumerate(coords):
        if i in (2, 3):
            color = BLUE if i == 2 else TEAL
            w = compute_width
            s.rect(x - w / 2, y - 38, w, 76, "#eaf1fb" if i == 2 else "#e0f1f0", color)
            s.text(x, y - 14, "Edge north" if i == 2 else "Edge south", 18, NAVY, 700, "middle")
            s.text(x, y + 10, "Video analytics" if i == 2 else "Telemetry", 16, color, 600, "middle")
            s.text(x, y + 31, f"{report.node_load[i]:g} / {problem.nodes[i].cpu:g} vCPU", 15, GRAY, anchor="middle")
        else:
            w = gw_width if i < 2 else transit_width
            s.rect(x - w / 2, y - 26, w, 52, "white", LINE, 10)
            names = {0: "North gateway", 1: "South gateway", 4: "Core", 5: "Cloud"}
            s.text(x, y - 3, names[i], 17, NAVY, 600, "middle")
            detail = "Cameras + sensors" if i < 2 else "Transit" if i == 4 else f"{problem.nodes[i].cpu:g} vCPU"
            s.text(x, y + 16, detail, 13 if mobile and i < 2 else 14, GRAY, anchor="middle")

    ox, oy, ow = (24, 872, 512) if mobile else (688, 310, 328)
    s.rect(ox, oy, ow, 232 if mobile else 296, "#f2f6fa")
    s.text(ox + 20, oy + 30, "EXAMPLE OUTPUT", 15, GRAY, 700)
    s.text(ox + 20, oy + 67, "Video analytics → Edge north", 18, BLUE, 600)
    s.text(ox + 20, oy + 99, "Telemetry → Edge south", 18, TEAL, 600)
    assert len(set(solution.quantities)) == 1
    s.text(ox + 20, oy + 143,
           f"{len(problem.demands)} demands · {solution.quantities[0]:g} Mbps each", 18, NAVY, 700)
    s.text(ox + 20, oy + 174, "CPU, links, bounds, latency: valid", 16, GRAY)
    s.text(ox + 20, oy + 205, "Manually specified and checked.", 15, GRAY)
    if not mobile:
        s.text(ox + 20, oy + 237, "Feasible within the network model.", 16, GRAY)
    s.text(24, height - 31, "Illustration only · not a PPO evaluation" if mobile else
           "Illustration only · not a PPO evaluation. Gray links retain the full topology; other demand routes are omitted.",
           15, GRAY)
    s.save("ppo-placement-mobile.svg" if mobile else "ppo-placement.svg")
