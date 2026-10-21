"""Regenerate README figures from the shipped instance and recorded experiment.

Run from any directory after installing the project and Matplotlib. SVG diagrams
use only the standard library; experimental charts use Matplotlib's Agg backend.
No training, benchmark rerun, network access, or checkpoint loading is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import textwrap
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_ppo.constraints import assess
from network_ppo.heuristic import solve_baseline
from network_ppo.instance_io import load_instance
from network_ppo.paths import PathCatalog

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets"
INSTANCE = ROOT / "instances/eval/bottleneck_unseen.json"
CSV = ROOT / "docs/temporal_benchmark_two_cases.csv"
SUMMARY = ROOT / "docs/temporal_benchmark_two_cases_summary.json"
NAVY, BLUE, TEAL = "#142d4e", "#2563a6", "#087f82"
GRAY, PALE, LINE, AMBER = "#52657b", "#f2f6fa", "#c5d1de", "#a55b0b"


class SVG:
    def __init__(self, width, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            "<defs>"
            + "".join(
                f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{color}"/></marker>'
                for name, color in [
                    ("teal", TEAL),
                    ("blue", BLUE),
                    ("amber", AMBER),
                    ("gray", GRAY),
                ]
            )
            + "</defs>",
            f'<rect width="{width}" height="{height}" rx="16" fill="white"/>',
            '<g font-family="DejaVu Sans, Arial, sans-serif">',
        ]

    def text(self, x, y, value, size=20, color=NAVY, weight=400, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'
        )

    def rect(self, x, y, w, h, fill=PALE, stroke="none", radius=12):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
        )

    def line(
        self,
        x1,
        y1,
        x2,
        y2,
        color=LINE,
        width=3,
        arrows=False,
        both=False,
        dashed=False,
    ):
        name = (
            "teal"
            if color == TEAL
            else "blue"
            if color == BLUE
            else "amber"
            if color == AMBER
            else "gray"
        )
        markers = f' marker-end="url(#{name})"' if arrows else ""
        if both:
            markers += f' marker-start="url(#{name})"'
        self.parts.append(
            f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{markers}'
            + (' stroke-dasharray="6 6"' if dashed else "")
            + "/>"
        )

    def pill(self, x, y, value, color=GRAY, size=18):
        w = len(value) * size * 0.59 + 16
        self.rect(x - w / 2, y - 19, w, 27, "white", radius=5)
        self.text(x, y, value, size, color, anchor="middle")

    def save(self, name):
        (ASSETS / name).write_text(
            "\n".join(self.parts + ["</g></svg>", ""]), encoding="utf-8", newline="\n"
        )


def hero(problem, catalog, solutions, mobile=False):
    width = 560 if mobile else 960
    panel_h = 468 if mobile else 395
    height = 152 + 2 * panel_h + 80
    svg = SVG(
        width,
        height,
        "Network orchestration under constraints",
        "Actual held-out topology and independently validated greedy and lookahead decisions. Arrows show aggregate directed traffic, labels show load/capacity in Mbps. Service placements stay fixed.",
    )
    svg.text(24, 38, "NETWORK ORCHESTRATION", 17, TEAL, 700)
    svg.text(24, 76, "A detour unlocks capacity", 28 if mobile else 34, weight=700)
    svg.text(
        24,
        106,
        "Same topology. Same placement. Feasible routes.",
        18 if mobile else 21,
        GRAY,
    )
    # The hand-designed layout is keyed to real node IDs; fail on another topology.
    ids = [n.id for n in problem.nodes]
    assert ids == [
        "gateway_north",
        "gateway_south",
        "edge_north",
        "edge_south",
        "core_switch",
        "cloud_dc",
    ]
    assert [(e.source, e.target) for e in problem.links] == [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5),
    ]
    assert len({d.max_latency_ms for d in problem.demands}) == 1
    for index, (method, solution) in enumerate(solutions.items()):
        report = assess(problem, solution, catalog)
        assert report.feasible
        top = 130 + index * panel_h
        svg.rect(16, top, width - 32, panel_h - 16, PALE)
        svg.text(
            32,
            top + 33,
            "01 / Fixed-route greedy" if index == 0 else "02 / Routing lookahead",
            21,
            weight=700,
        )
        svg.text(
            32 if mobile else width - 32,
            top + 66 if mobile else top + 34,
            f"{report.total_mbps:g} / {sum(d.max_mbps for d in problem.demands)} Mbps",
            27,
            AMBER if index == 0 else TEAL,
            700,
            "start" if mobile else "end",
        )
        y0 = top + (142 if mobile else 113)
        y1 = y0 + (176 if mobile else 151)
        mid = (y0 + y1) / 2
        coords = [
            (80 if mobile else 110, y0),
            (80 if mobile else 110, y1),
            (290 if mobile else 365, y0),
            (290 if mobile else 365, y1),
            (470 if mobile else 625, mid),
            (470 if mobile else 842, mid + 116 if mobile else mid),
        ]
        labels = [
            (0, (coords[0][0] + coords[2][0]) / 2, y0 - 13),
            (1, 171 if mobile else 194, mid + 42),
            (2, 196 if mobile else 265, mid - 28),
            (3, (coords[1][0] + coords[3][0]) / 2, y1 - 13),
            (4, 402 if mobile else 498, y0 + 24),
            (5, 402 if mobile else 498, y1 - 8),
            (6, 516 if mobile else 736, mid + 64 if mobile else mid - 13),
        ]
        captions = []
        for li, lx, ly in labels:
            link = problem.links[li]
            a, b = coords[link.source], coords[link.target]
            forward = float(
                report.edge_load[catalog.arc_index[(link.source, link.target)]]
            )
            reverse = float(
                report.edge_load[catalog.arc_index[(link.target, link.source)]]
            )
            active = forward > 0 or reverse > 0
            color = (
                AMBER
                if index == 0 and active and max(forward, reverse) == link.capacity_mbps
                else TEAL
                if active
                else LINE
            )
            # Trim to the box boundary in either direction, with a visible arrow.
            dx, dy = b[0] - a[0], b[1] - a[1]
            norm = math.hypot(dx, dy)
            trim_a = min(norm * 0.3, 68 if abs(dx) > abs(dy) else 44)
            trim_b = min(norm * 0.3, 74 if abs(dx) > abs(dy) else 44)
            start = (a[0] + dx / norm * trim_a, a[1] + dy / norm * trim_a)
            end = (b[0] - dx / norm * trim_b, b[1] - dy / norm * trim_b)
            if reverse and not forward:
                start, end = end, start
            svg.line(
                *start,
                *end,
                color,
                4 if active else 2,
                active,
                forward > 0 and reverse > 0,
                not active,
            )
            value = (
                f"{forward:g}/{link.capacity_mbps:g}"
                if not reverse
                else f"{forward:g},{reverse:g}/{link.capacity_mbps:g}"
            )
            captions.append((lx, ly, value, AMBER if color == AMBER else GRAY))
        for lx, ly, value, color in captions:
            svg.pill(lx, ly, value, color, 17 if mobile else 19)
        names = [
            "North gateway",
            "South gateway",
            "Edge north",
            "Edge south",
            "Core switch",
            "Cloud DC",
        ]
        for ni, (x, y) in enumerate(coords):
            services = [
                s
                for si, s in enumerate(problem.services)
                if int(solution.placement[si]) == ni
            ]
            bw, bh = (150, 84) if services else (150 if ni < 2 else 116, 64)
            svg.rect(
                x - bw / 2,
                y - bh / 2,
                bw,
                bh,
                "#e0f1f0" if services else "white",
                TEAL if services else LINE,
                10,
            )
            svg.text(
                x, y - 10, names[ni], 17 if mobile else 19, weight=700, anchor="middle"
            )
            svg.text(
                x,
                y + 13,
                f"{report.node_load[ni]:g}/{problem.nodes[ni].cpu:g} vCPU",
                17,
                GRAY,
                anchor="middle",
            )
            if services:
                alias = {
                    "video_analytics": "Video analytics",
                    "machine_telemetry": "Telemetry",
                }
                svg.text(
                    x, y + 33, alias[services[0].id], 16, TEAL, 600, anchor="middle"
                )
        qs = dict(zip([d.id for d in problem.demands], solution.quantities.tolist()))
        bottom = top + panel_h - 54
        svg.text(
            32,
            bottom,
            f"Cameras N/S: {qs['camera_north']}/{qs['camera_south']}  ·  Sensors N/S: {qs['sensors_north']}/{qs['sensors_south']} Mbps",
            17 if mobile else 19,
            GRAY,
        )
        limit = min(d.max_latency_ms for d in problem.demands)
        svg.text(
            32,
            bottom + 26,
            f"Max route latency {max(report.demand_latency):g} ms / {limit:g} ms budget",
            17 if mobile else 19,
            GRAY,
        )
    y = height - 64
    svg.line(24, y, 54, y, TEAL, 4, True)
    svg.text(64, y + 6, "Traffic", 17, GRAY)
    svg.line(164, y, 194, y, AMBER, 4, True)
    svg.text(204, y + 6, "Bottleneck", 17, GRAY)
    svg.line(345, y, 375, y, LINE, 2, dashed=True)
    svg.text(385, y + 6, "Unused link", 17, GRAY)
    svg.text(
        24,
        height - 24,
        "Labels: load/capacity (Mbps); a,b = two directions."
        if mobile
        else "Labels: load/capacity (Mbps); a,b = forward,reverse loads. Independent capacities per direction.",
        16,
        GRAY,
    )
    svg.save("orchestration-mobile.svg" if mobile else "orchestration.svg")


def ppo_overview(problem, mobile=False):
    from readme_hero import render_overview

    render_overview(SVG, problem, mobile)


def architecture(mobile=False):
    w = 560 if mobile else 960
    h = 1130 if mobile else 950
    s = SVG(
        w,
        h,
        "Decision pipeline and training boundary",
        "CLI and HTTP inputs enter static or temporal decision engines. Engines reuse Problem, Solution, PathCatalog and assess. Optional PPO checkpoints are produced offline. Feasibility is checked before JSON export. Temporal clients persist next_previous; the demo applies no network actions.",
    )
    s.text(24, 38, "SOFTWARE ARCHITECTURE", 17, TEAL, 700)
    s.text(24, 76, "One shared constraint model", 28 if mobile else 34, weight=700)

    def box(x, y, width, height, title, lines, color=NAVY, fill=PALE):
        s.rect(x, y, width, height, fill, LINE)
        s.text(x + 18, y + 31, title, 22, color, 700)
        for i, line in enumerate(lines):
            s.text(x + 18, y + 60 + 27 * i, line, 18, GRAY)

    box(
        24,
        100,
        w - 48,
        114,
        "01 / Network input",
        [
            "CLI + HTTP JSON → instance_io.py / dynamics.py",
            "Static instance or current observations + public forecasts",
        ]
        if not mobile
        else [
            "CLI / HTTP JSON · instance_io / dynamics",
            "Instance OR observations + forecasts",
        ],
    )
    ey = 240
    ew = (w - 72) / 2 if not mobile else w - 48
    box(
        24,
        ey,
        ew,
        170,
        "02 / Static engine",
        [
            "inference.py · DecisionEngine",
            "Greedy / lookahead / multistart",
            "Optional PPO · feasible edits",
            "auto: best feasible throughput",
        ],
    )
    ty = ey if not mobile else ey + 186
    tx = 48 + ew if not mobile else 24
    box(
        tx,
        ty,
        ew,
        170,
        "02 / Temporal engine",
        [
            "temporal_inference.py",
            "Keep / greedy / myopic / beam MPC",
            "Optional PPO · one action/period",
            "auto: best immediate stage reward",
        ],
    )
    if not mobile:
        for cx in (24 + ew / 2, tx + ew / 2):
            s.line(cx, 214, cx, 238, GRAY, 2, True)
    else:
        s.line(w - 12, 157, w - 12, ty + 85, GRAY, 2)
        for cy in (ey + 85, ty + 85):
            s.line(w - 12, cy, w - 26, cy, GRAY, 2, True)
        s.line(w - 24, 157, w - 12, 157, GRAY, 2)
    shared_y = ty + 190
    box(
        24,
        shared_y,
        w - 48,
        95,
        "Shared domain + feasibility primitives",
        ["problem.py · paths.py · constraints.assess"],
        TEAL,
        "#e0f1f0",
    )
    s.text(42, shared_y + 83, "Optional exact.py: tiny-instance MILP (SciPy)", 16, GRAY)
    train_y = shared_y + 112
    box(
        24,
        train_y,
        w - 48,
        108,
        "Offline training → trusted checkpoints",
        [
            "train.py / train_temporal.py → environments → SB3 PPO",
            "model.zip + vecnormalize.pkl + metadata.json",
        ]
        if not mobile
        else [
            "train / train_temporal → Gymnasium → SB3",
            "Policy + frozen normalization + metadata",
        ],
        BLUE,
    )
    # The training rail feeds policy runtimes; it is outside the decision flow.
    s.line(24, train_y + 30, 12, train_y + 30, BLUE, 2)
    s.line(12, train_y + 30, 12, ey + 85, BLUE, 2)
    s.line(12, ey + 85, 24, ey + 85, BLUE, 2, True)
    if mobile:
        s.line(12, ty + 85, 24, ty + 85, BLUE, 2, True)
    else:
        s.line(12, ey + 85, 12, 228, BLUE, 2)
        s.line(12, 228, w / 2, 228, BLUE, 2)
        s.line(w / 2, 228, w / 2, ty + 85, BLUE, 2)
        s.line(w / 2, ty + 85, tx, ty + 85, BLUE, 2, True)
    vy = train_y + 126
    s.line(w - 12, ty + 85, w - 12, vy + 56, GRAY, 2)
    if not mobile:
        s.line(w - 24, ty + 85, w - 12, ty + 85, GRAY, 2)
        s.line(24 + ew / 2, ey + 170, 24 + ew / 2, ey + 178, GRAY, 2)
        s.line(24 + ew / 2, ey + 178, w - 12, ey + 178, GRAY, 2)
    s.line(w - 12, vy + 56, w - 26, vy + 56, GRAY, 2, True)
    box(
        24,
        vy,
        w - 48,
        112,
        "03 / Validate before export",
        [
            "CPU · directed bandwidth · integer bounds · loaded latency",
            "Static auto: explicit baseline fallback on PPO failure",
        ]
        if not mobile
        else [
            "CPU · bandwidth · bounds · latency",
            "Static auto: explicit baseline fallback",
        ],
        TEAL,
        "#e0f1f0",
    )
    oy = vy + 138
    s.line(w / 2, vy + 112, w / 2, oy - 2, GRAY, 2, True)
    box(
        24,
        oy,
        w - 48,
        110,
        "04 / Decision JSON",
        [
            "solution_io.py · placements / routes / Mbps / provenance",
            "Temporal: next_previous → client persists switching state",
        ]
        if not mobile
        else [
            "solution_io · allocation + provenance",
            "Temporal client persists next_previous",
        ],
    )
    s.text(
        24,
        h - 19,
        "Temporal invalid actions: feasible fallback. No actuation."
        if mobile
        else "Temporal invalid proposals: feasible fallback. No network actuation.",
        16,
        GRAY,
    )
    s.save("architecture-mobile.svg" if mobile else "architecture.svg")


def temporal_data():
    with CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    methods = list(summary["summary"])
    scenarios = {Path(f).stem for f in summary["protocol"]["test_files"]}
    assert len(scenarios) == 2
    for method in methods:
        group = [r for r in rows if r["strategy"] == method]
        assert len(group) == 2 and {r["scenario"] for r in group} == scenarios
        expected = summary["summary"][method]
        for key, column, fn in [
            ("mean_total_reward", "total_reward", mean),
            ("std_total_reward", "total_reward", pstdev),
            ("mean_runtime_total_ms", "runtime_total_ms", mean),
        ]:
            assert math.isclose(
                fn(float(r[column]) for r in group), expected[key], abs_tol=1e-8
            )
    return rows, summary, methods


def experiment(rows, summary, methods, mobile=False):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": NAVY,
            "axes.labelcolor": GRAY,
            "xtick.color": GRAY,
            "ytick.color": NAVY,
            "svg.fonttype": "none",
            "svg.hashsalt": "mec-deployable-figures",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2 if mobile else 1,
        1 if mobile else 2,
        figsize=(5.2, 9.5) if mobile else (10.4, 5.5),
    )
    fig.subplots_adjust(
        left=0.26 if mobile else 0.13,
        right=0.91,
        top=0.80 if mobile else 0.75,
        bottom=0.28 if mobile else 0.30,
        hspace=0.65,
        wspace=0.6,
    )
    fig.text(0.045, 0.95, "TEMPORAL BENCHMARK", color=TEAL, fontsize=12, weight="bold")
    fig.text(
        0.045,
        0.895,
        "More search, same reward"
        if mobile
        else "Bounded MPC matches myopic reward at higher cost",
        fontsize=19 if mobile else 20,
        weight="bold",
    )
    fig.text(
        0.045,
        0.845 if mobile else 0.82,
        "Two held-out synthetic cases · 24 periods each",
        fontsize=12,
        color=GRAY,
    )
    names = {
        "keep": "Keep",
        "greedy": "Greedy",
        "myopic": "Myopic",
        "mpc4": "MPC (H=4)",
        "mpc8": "MPC (H=8)",
    }
    colors = [LINE, BLUE, TEAL, BLUE, NAVY]
    for ax, column, scale, title, label in zip(
        axes,
        ["total_reward", "runtime_total_ms"],
        [1, 1000],
        ["Cumulative reward", "Total inference time"],
        ["Undiscounted reward (toy units)", "Seconds / 24-period rollout"],
    ):
        for i, method in enumerate(methods):
            vals = [float(r[column]) / scale for r in rows if r["strategy"] == method]
            avg = mean(vals)
            ax.barh(i, avg, height=0.56, color=colors[i], alpha=0.92)
            ax.scatter(
                vals,
                [i - 0.12, i + 0.12],
                marker="o",
                s=23,
                facecolor="white",
                edgecolor=NAVY,
                linewidth=0.9,
                zorder=3,
                clip_on=False,
            )
            ax.annotate(
                f"{avg:.1f}" if scale == 1 else f"{avg:.3f}",
                (avg, i),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=11,
                color=NAVY,
            )
        ax.set_yticks(range(len(methods)), [names[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 610 if scale == 1 else 6.4)
        ax.set_xlabel(label, labelpad=10, fontsize=11)
        ax.set_title(title, loc="left", fontsize=14, pad=14, weight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=PALE)
        ax.set_axisbelow(True)
    depth = mean(
        float(r["mean_horizon_reached"])
        for r in rows
        if r["strategy"] in ("mpc4", "mpc8")
    )
    cutoffs = sorted(
        {int(r["planning_cutoffs"]) for r in rows if r["strategy"] in ("mpc4", "mpc8")}
    )
    assert len(cutoffs) == 1
    notes = [
        "Bars: mean of two cases; circles: individual cases.",
        f"Requested H=4/8 → mean completed depth {depth:g}; {cutoffs[0]}/24 cutoffs per case.",
        f"Expansion budget {summary['protocol']['max_expansions']} · no wall-time cutoff · no PPO checkpoints.",
        "Recorded runtimes are machine-specific; no matched compute budget.",
    ]
    if mobile:
        notes = [line for note in notes for line in textwrap.wrap(note, width=52)]
    for i, line in enumerate(notes):
        fig.text(
            0.045,
            0.135 - i * 0.031 if not mobile else 0.205 - i * 0.026,
            line,
            fontsize=11 if mobile else 10.5,
            color=GRAY,
        )
    name = "temporal-comparison-mobile" if mobile else "temporal-comparison"
    fig.savefig(
        ASSETS / f"{name}.svg",
        metadata={
            "Date": None,
            "Title": "Bounded MPC: reward and computational cost",
            "Description": "Recorded two-case comparison; no MPC reward advantage and no PPO performance claim.",
        },
    )
    if not mobile:
        fig.savefig(
            ASSETS / f"{name}.png",
            dpi=180,
            metadata={"Software": "docs/figures/generate.py"},
        )
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    problem = load_instance(INSTANCE)
    catalog = PathCatalog(problem, 3)
    solutions = {
        m: solve_baseline(problem, catalog, m) for m in ("greedy", "lookahead")
    }
    rows, summary, methods = temporal_data()
    for mobile in (False, True):
        ppo_overview(problem, mobile)
        hero(problem, catalog, solutions, mobile)
        architecture(mobile)
        experiment(rows, summary, methods, mobile)
    sources = [
        INSTANCE,
        CSV,
        SUMMARY,
        Path(__file__),
        Path(__file__).with_name("readme_hero.py"),
    ]
    from readme_hero import example_allocation

    example, report = example_allocation(problem)
    provenance = {
        "schema_version": 1,
        "generator": "docs/figures/generate.py",
        "paths": 3,
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "readme_illustration": {
            "method": "manually specified; not a PPO evaluation",
            "placement": example.placement.tolist(),
            "routes": example.routes.tolist(),
            "quantities_mbps": example.quantities.tolist(),
            "feasible": report.feasible,
            "highlighted_demands": ["camera_north", "sensors_south"],
        },
        "static_decisions": {
            m: {
                "placement": s.placement.tolist(),
                "routes": s.routes.tolist(),
                "quantities_mbps": s.quantities.tolist(),
                "feasible": assess(problem, s, catalog).feasible,
            }
            for m, s in solutions.items()
        },
        "temporal_protocol": summary["protocol"],
        "temporal_summary": summary["summary"],
        "versions": {
            name: version(name) for name in ("matplotlib", "numpy", "networkx")
        },
        "notes": [
            "Architecture is a manually arranged source-code map.",
            "Temporal chart reuses recorded runtimes; regeneration does not rerun benchmarks.",
            "Legacy plot_provenance.json is retained unchanged.",
        ],
    }
    (ASSETS / "readme-figure-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        "Generated PPO overview and archived technical figures, mobile variants and provenance."
    )


if __name__ == "__main__":
    main()
