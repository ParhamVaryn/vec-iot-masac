# Screenshot Guide - VEC-IoT MASAC Report

Place all screenshots in `figures/` beside `main.tex`. Use the exact names below. The LaTeX file compiles without them and shows placeholders until you add the images.

| # | Image name | Source | What to capture |
|---|---|---|---|
| 1 | `screenshot_01.png` | `src/xml_loader.py`, lines 14-35 | `VehicleRecord` and `TaskRecord` |
| 2 | `screenshot_02.png` | `src/xml_loader.py`, lines 37-77 | XML timestep iterators |
| 3 | `screenshot_03.png` | `src/masac_env.py`, lines 12-25 | `EnvironmentConfig` |
| 4 | `screenshot_04.png` | `src/masac_env.py`, lines 41-121 | edge/UAV initialization, candidates, queues |
| 5 | `screenshot_05.png` | `src/masac_env.py`, lines 147-163 | `_serve_queues()` |
| 6 | `screenshot_06.png` | `src/uav_control.py`, lines 5-9 | `compute_traffic_centers()` |
| 7 | `screenshot_07.png` | `src/uav_control.py`, lines 12-50 | center assignment and UAV movement |
| 8 | `screenshot_08.png` | `src/masac_env.py`, lines 164-202 | `advance()` |
| 9 | `screenshot_09.png` | `src/system_model.py`, lines 325-358 | `build_candidates()` |
| 10 | `screenshot_10.png` | `src/system_model.py`, lines 360-402 | `valid_action_mask()` |
| 11 | `screenshot_11.png` | `src/system_model.py`, lines 161-181 and 246-278 | path/channel + `link_rate()`; if needed collapse unimportant lines in editor |
| 12 | `screenshot_12.png` | `src/system_model.py`, lines 283-320 | bits, cycles, compute time, compute energy |
| 13 | `screenshot_13.png` | `src/masac_env.py`, lines 223-260 | `_global_features()` |
| 14 | `screenshot_14.png` | `src/masac_env.py`, lines 262-304 | `observation()` |
| 15 | `screenshot_15.png` | `src/masac_env.py`, lines 321-382 | `step_task()` and reward |
| 16 | `screenshot_16.png` | `src/masac.py`, lines 20-46 | MASACConfig + MLP |
| 17 | `screenshot_17.png` | `src/masac.py`, lines 49-81 | ReplayBuffer |
| 18 | `screenshot_18.png` | `src/masac.py`, lines 84-112 | actor, Q1/Q2, TQ1/TQ2, optimizers |
| 19 | `screenshot_19.png` | `src/masac.py`, lines 114-127 | `masked_logits()` + `act()` |
| 20 | `screenshot_20.png` | `src/masac.py`, lines 129-145 | `_dist()` and current Q evaluation |
| 21 | `screenshot_21.png` | `src/masac.py`, lines 147-161 | target critics, soft value, Bellman target |
| 22 | `screenshot_22.png` | `src/masac.py`, lines 163-170 | critic losses + optimizer steps |
| 23 | `screenshot_23.png` | `src/masac.py`, lines 172-186 | actor objective + backprop |
| 24 | `screenshot_24.png` | `src/masac.py`, lines 188-196 | target soft update |
| 25 | `screenshot_25.png` | `train_masac.py`, lines 108-182 | complete task-level training loop |
| 26 | `screenshot_26.png` | `evaluate_masac.py`, lines 155-223 | matched scenarios + deterministic actor action |
| 27 | `screenshot_27.png` | `evaluate_masac.py`, lines 238-317 | evaluation summaries + CSV output |
| 28 | `screenshot_28.png` | final Colab notebook | cells reading evaluation CSV files |
| 29 | `screenshot_29.png` | final Colab notebook | matched UAV vs No-UAV latency/energy result |
| 30 | `screenshot_30.png` | final Colab notebook | packet loss/bitrate/load/success/UAV/decision plots |

## Screenshot style
- Use the final local files, not old/legacy versions.
- Keep the editor tab or filename visible when practical.
- Use a readable zoom (usually 110-140%).
- Do not include unrelated sidebars or terminal output unless it is part of the evidence.
- One screenshot should support one report idea; avoid combining many unrelated functions.
- For result screenshots, use only outputs from the final retrained model after the latest code/formula changes.
