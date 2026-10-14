# PPO overview and retained technical figures

Run from the repository root after installing the project. Matplotlib is the only extra figure dependency; NumPy and NetworkX are already core dependencies.

```bash
python -m pip install -e '.[test]' 'matplotlib>=3.7,<4'
python docs/figures/generate.py
```

The generator writes self-contained SVGs to `docs/assets/`, a PNG of the experimental comparison, narrow-layout SVG variants for the README's `<picture>` elements, and [`readme-figure-provenance.json`](../assets/readme-figure-provenance.json). It uses fixed layouts, a shared navy/blue/teal palette, an opaque white canvas for contrast in either GitHub theme, deterministic SVG IDs, and no timestamps. Byte-identical output requires the same libraries and font environment; the provenance records Matplotlib, NumPy and NetworkX versions. No external fonts, scripts, image references, or AI artwork are used.

Validation on 14 October 2026: all 52 existing tests passed (two pre-existing Gymnasium observation-bound warnings). Static CLI and benchmark outputs reproduced 18/28 and 28/28 Mbps. The full two-case temporal rerun reproduced rewards, served traffic, switching counts, search depths and cutoffs; historical runtimes were retained. All eight generated assets were byte-identical after a second regeneration. Forty-two local references and both provenance manifests were checked. SVGs were rasterized and visually inspected at 800 px and 390 px; a local Markdown browser preview also verified image loading, mobile source selection and absence of page overflow at 1100 px and 390 px, using light and dark page backgrounds. This was a local preview, not a published GitHub rendering check. Docker was not run.

## Sources and interpretation

| Asset | Source and verification |
| --- | --- |
| `ppo-overview.svg` | The current README visual: actual topology, available CPU, service requirements and demand ceilings from `instances/eval/bottleneck_unseen.json`, followed by a conceptual PPO → validation → JSON pipeline. It shows no allocation outcome or comparison. A compact mobile layout is generated from the same inputs. |
| `orchestration.svg` | Loads `instances/eval/bottleneck_unseen.json`, runs the actual `greedy` and `lookahead` baselines with three candidate paths, and checks both with `assess`. All loads, bandwidths, CPU use and latencies are computed. |
| `architecture.svg` | Manually arranged source map of input parsers, static/temporal inference engines, shared types/paths/checker, offline training artifacts and JSON export. The outer rails connect inputs, engines and validation; training feeds policy runtimes separately. |
| `temporal-comparison.svg` / `.png` | Reads the committed two-case CSV and verifies every plotted method's case membership, sample count, mean reward, population standard deviation and mean runtime against its JSON summary. Bars show means; circles show both cases, not confidence intervals. |

The orchestration comparison, detailed architecture and temporal experimental figure are retained documentation assets; the simplified README uses only the PPO overview. Earlier results and original visuals are preserved without making comparisons the focus of the deployment project.

For the hero, shortened labels map to `gateway_north`, `gateway_south`, `edge_north`, `edge_south`, `core_switch`, and `cloud_dc`. Video analytics and telemetry are the actual `video_analytics` and `machine_telemetry` services. Arrows aggregate demands on directed arcs. A single label `a/c` gives forward load/capacity; `a,b/c` gives forward and reverse loads against **independent** capacity `c` in each direction. The unused cloud link is retained. Routes and demand quantities are also captured in provenance. The layout deliberately fails on a different node ordering rather than silently drawing the wrong topology.

The hero demonstrates rerouting, not service migration. In the improved decision, south camera traffic follows south gateway → south edge → core → north edge; north sensor traffic follows north gateway → north edge → core → south edge. Both cross-links become unused, local links carry 14/14 Mbps, and each edge/core direction carries 7/20 Mbps. The 28 Mbps demand ceiling certifies this instance's throughput optimum.

The historical temporal data are [CSV](../temporal_benchmark_two_cases.csv) and [summary](../temporal_benchmark_two_cases_summary.json). The original legacy provenance is actually at `docs/assets/plot_provenance.json`, not `docs/plot_provenance.json`. All original data files are preserved.

## Repeat the experiment

The two cases are the first two sorted test files: `temporal_test_bursty_901.json` and `temporal_test_daily_900.json`. The recorded command, documented in [verification](../verification.md), uses the benchmark's default beam width of two:

```bash
python scripts/benchmark_temporal.py --scenarios instances/temporal/test --max-cases 2 --beam-width 2 --max-candidates 5 --max-expansions 30 --max-planning-ms 0 --horizons 1 4 8 --output outputs/temporal_figure_check
```

This writes a new experiment to ignored `outputs/`; it does not overwrite the historical data. Compare reward, served traffic, route changes, cutoffs and completed depth to the committed CSV. Wall times vary by machine. Requested horizon 4 or 8 is not completed search depth: both recorded methods averaged 2.88, with 21/24 cutoffs in each case. No PPO checkpoint was used. The recorded summary's generic “time-bounded” string is imprecise for this run: `max_planning_ms=null` disables the time cutoff; `max_expansions=30` is the operative budget. This discrepancy is disclosed without altering the source result.

To attempt untruncated planning, set `--max-expansions 0 --max-planning-ms 0`; finite candidate caps and beam width still make it an approximation. This can require substantially more CPU time. For static results:

```bash
python benchmark.py --instances instances/eval --output outputs/static_figure_check
```

## Retained legacy visuals

The original [network](../assets/network.png), [greedy decision](../assets/decision_network.png), and [efficiency comparison](../assets/efficiency.png) remain useful documentation of the **legacy factory scenarios**. Their [provenance](../assets/plot_provenance.json) records instance hashes, feasible-seed and greedy results, and library versions; the [notebook](../../notebook/network_and_efficiency.ipynb) can regenerate them. They show no PPO results and are not evidence about the new congested instance.

The original [integration sketch](../assets/integration.svg) is also retained as historical context: its external orchestration adapter is explicitly marked “to develop.” The new architecture figure maps the implemented CLI/HTTP decision adapters without suggesting network actuation exists. No original figures or attribution were removed.
