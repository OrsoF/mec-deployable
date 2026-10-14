# Repository guidance

## Project purpose

`mec-deployable` presents PPO-based service placement, routing and bandwidth
allocation, with feasibility validation and CLI/HTTP deployment interfaces.
Its public presentation should emphasize how a trained policy is integrated and
used to generate network decisions. Comparative solver benchmarking belongs in
the related `mec-placement-benchmarks` project, not in this README's pitch.

## README and communication

- Keep the README short, compact and understandable to a first-time visitor.
  Explain the network problem before implementation details. Keep the README in
  English unless the user requests another language; discuss work with this user
  in French.
- Use concrete, literal language. Do not use marketing slogans, catchy formulas
  or vague promises. In particular, do not reintroduce "Learn offline. Decide on
  demand." Describe the operation or result directly.
- Make PPO central. Explain offline training and reuse of the saved policy for
  inference, without implying that the current static engine generates a complete
  decision in one neural-network forward pass.
- Keep only the essential installation, training, evaluation and Docker commands
  in the README. Put optional arguments, HTTP examples, architecture details and
  experimental protocols in linked documentation.
- Use one clear main visual. Do not add solver comparison tables, before/after
  solver comparisons, MPC comparisons or PPO-versus-baseline claims to the README.
  Preserve existing detailed experiments and historical assets unless the user
  explicitly asks to remove them.
- Place `Related Projects` near the end, with these links and descriptions:
  - `mec-placement-benchmarks`: https://github.com/OrsoF/mec-placement-benchmarks
    — Benchmarking optimization and learning methods for constrained edge-network
    orchestration.
  - `mdpforge`: https://github.com/OrsoF/mdpforge — A Python framework for finite MDP
    modeling, solving, and benchmarking.
  Follow them with: "More about my research, publications, and software projects:
  [orsof.github.io](https://orsof.github.io/)." Keep personal-profile links out of
  the header.

## Evidence and scope

- Measure speed before claiming a latency. Report the actual timed scope, model
  loading policy, number of samples, hardware and checkpoint. Prefer the complete
  validated decision path over an isolated policy forward pass.
- Distinguish inference timing from placement quality. A smoke checkpoint does
  not establish the quality or typical latency of a well-trained policy. Keep
  claims traceable to committed results and a reproducible measurement script.
- Keep limitations concise in the README and detailed in documentation: synthetic
  scenarios, fixed topology/identities, feasibility within the implemented model,
  trusted checkpoints, HTTP security limitations and unverified deployment steps.
- Documentation changes must not alter algorithms, objectives, numerical results
  or public APIs. Do not push automatically.

## Visuals and verification

- Store final visuals in `docs/assets/`. Prefer SVG for diagrams and source-derived
  numerical labels. Maintain their generator under `docs/figures/`.
- Edit and regenerate the actual desktop and mobile image files referenced by
  the README; changing prose or a script alone is insufficient.
- Inspect the rendered images. Verify that the old wording is absent from both
  variants and that labels fit at desktop and mobile widths. If an old image
  persists in a preview, verify the source and use a new asset reference when
  needed; show the current rendered image rather than merely saying it changed.
- Check internal links and deterministic figure generation. Use meaningful
  existing tests for affected execution paths; do not add tests that merely
  mirror a reversible wording change.
