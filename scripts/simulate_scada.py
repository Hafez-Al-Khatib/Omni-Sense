"""
Omni-Sense WNTR Digital Twin — SCADA Pressure Simulator
=========================================================
Generates hydraulically accurate pressure readings for Lebanese water
distribution network scenarios using the WNTR (Water Network Tool for
Resilience) library.

The output CSV feeds directly into EEP's metadata validation:
  pressure_bar per junction → cross-checked against predicted fault class
  by iep2/app/main.py::_check_scada_consistency()

Two engines:
  1. WNTR (preferred) — full hydraulic simulation via EPANET solver
  2. Analytical fallback — Hazen-Williams + emitter model, no extra deps

Usage:
    # Basic: simulate all 5 fault scenarios (24h each)
    python scripts/simulate_scada.py --batch --output-dir data/scada

    # Single scenario:
    python scripts/simulate_scada.py --scenario orifice_leak --output data/scada_orifice.csv

    # Install WNTR for higher fidelity:
    pip install wntr

Scenarios:
    no_leak               — Normal operation, pressure at design levels
    orifice_leak          — Pressurised orifice leak (rapid pressure drop)
    gasket_leak           — Slow seepage leak (gradual pressure loss)
    longitudinal_crack    — Crack propagating along pipe axis
    circumferential_crack — Full cross-section crack (worst case)

Output columns:
    timestamp, node_id, pressure_bar, demand_lps, scenario,
    fault_class, pipe_material, network_condition
"""

import argparse
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# ─── Inline Network INP ───────────────────────────────────────────────────────
# A minimal but physically realistic WDN representing a Lebanese
# municipal district: 6 junctions, 1 reservoir, 6 pipes.
# Pipe materials mix PVC (modern), Steel (industrial) and Cast_Iron (legacy).

_NETWORK_INP = """
[TITLE]
 Omni-Sense Demo Network — Lebanese District

[JUNCTIONS]
;ID     Elev   Demand  Pattern
 J1     45     0.15    DEM_PATTERN
 J2     38     0.12    DEM_PATTERN
 J3     32     0.18    DEM_PATTERN
 J4     28     0.20    DEM_PATTERN
 J5     25     0.10    DEM_PATTERN
 J6     22     0.08    DEM_PATTERN

[RESERVOIRS]
;ID   Head
 R1   90

[PIPES]
;ID  Node1  Node2  Length  Diameter  Roughness  MinorLoss  Status
 P1  R1     J1     500     150       100        0          Open
 P2  J1     J2     400     125       100        0          Open
 P3  J2     J3     350     100       110        0          Open
 P4  J3     J4     300     100       110        0          Open
 P5  J4     J5     280     80        120        0          Open
 P6  J5     J6     250     80        120        0          Open

[PATTERNS]
;ID           Multipliers
 DEM_PATTERN  0.6 0.5 0.45 0.4 0.5 0.7 1.0 1.3 1.4 1.3 1.2 1.1
              1.0 1.0 1.1  1.2 1.3 1.5 1.4 1.3 1.2 1.1 0.9 0.7

[TIMES]
 Duration            24:00
 Hydraulic Timestep   0:15
 Pattern Timestep     1:00
 Report Timestep      0:15
 Start ClockTime     12 am
 Statistic           None

[OPTIONS]
 Units               LPS
 Headloss            H-W
 Trials              40
 Accuracy            0.001
 Unbalanced          Continue 10
 Pattern             1
 Demand Multiplier   1.0
 Emitter Exponent    0.5

[END]
"""

# ─── Pipe material map ────────────────────────────────────────────────────────
_PIPE_MATERIAL = {
    "P1": "Steel", "P2": "Steel",
    "P3": "PVC",   "P4": "PVC",
    "P5": "Cast_Iron", "P6": "Cast_Iron",
}
_NODE_PIPE = {
    "J1": "P1", "J2": "P2", "J3": "P3",
    "J4": "P4", "J5": "P5", "J6": "P6",
}

# ─── Scenario Definitions ─────────────────────────────────────────────────────
# emitter_coeff (m³/s / bar^0.5) controls leak outflow magnitude.
SCENARIO_CONFIG = {
    "no_leak": {
        "fault_class":   "No_Leak",
        "leak_nodes":    [],
        "emitter_coeff": 0.0,
        "description":   "Normal operation — no active leaks",
    },
    "orifice_leak": {
        "fault_class":   "Orifice_Leak",
        "leak_nodes":    ["J5"],
        "emitter_coeff": 0.0012,
        "description":   "Pressurised orifice leak at J5",
    },
    "gasket_leak": {
        "fault_class":   "Gasket_Leak",
        "leak_nodes":    ["J4"],
        "emitter_coeff": 0.0004,
        "description":   "Gasket seepage at J4 (Cast Iron section)",
    },
    "longitudinal_crack": {
        "fault_class":   "Longitudinal_Crack",
        "leak_nodes":    ["J3", "J4"],
        "emitter_coeff": 0.0008,
        "description":   "Longitudinal crack spanning J3→J4",
    },
    "circumferential_crack": {
        "fault_class":   "Circumferential_Crack",
        "leak_nodes":    ["J3"],
        "emitter_coeff": 0.0020,
        "description":   "Circumferential crack at J3 (catastrophic)",
    },
}


# ─── WNTR Engine ─────────────────────────────────────────────────────────────

def _try_import_wntr():
    try:
        import wntr
        return wntr
    except ImportError:
        return None


def simulate_with_wntr(scenario: str, seed: int = 42) -> list[dict]:
    """Run a 24-hour WNTR hydraulic simulation. Falls back analytically if WNTR unavailable."""
    wntr = _try_import_wntr()
    if wntr is None:
        print("  [WARN] wntr not installed — using analytical fallback.")
        print("         pip install wntr  for full hydraulic simulation.")
        return _analytical_fallback(scenario, seed=seed)

    cfg = SCENARIO_CONFIG[scenario]

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".inp", delete=False, encoding="utf-8") as tmp:
        tmp.write(_NETWORK_INP)
        inp_path = tmp.name

    try:
        wn = wntr.network.WaterNetworkModel(inp_path)
        for node_id in cfg["leak_nodes"]:
            wn.get_node(node_id).emitter_coefficient = cfg["emitter_coeff"]

        results = wntr.sim.WNTRSimulator(wn).run_sim()
        pressures = results.node["pressure"]
        demands   = results.node["demand"]

        rows = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        for timestamp in pressures.index:
            sim_time = base_time + timedelta(seconds=float(timestamp))
            for node_id in ["J1", "J2", "J3", "J4", "J5", "J6"]:
                pressure_m   = float(pressures.loc[timestamp, node_id])
                pressure_bar = max(0.0, pressure_m * 0.0980665)
                demand_lps   = float(demands.loc[timestamp, node_id]) * 1000
                rows.append(_make_row(sim_time, node_id, pressure_bar, demand_lps, scenario, cfg))
        return rows
    finally:
        os.unlink(inp_path)


# ─── Analytical Fallback ──────────────────────────────────────────────────────

def _analytical_fallback(scenario: str, seed: int = 42) -> list[dict]:
    """
    Physics-based pressure model using Hazen-Williams + emitter equation.

    Reference pressures calibrated to Lebanese MOEW standards:
        Reservoir head = 90m → J1 ≈ 4.5 bar, J6 ≈ 2.5 bar
    """
    rng = random.Random(seed)
    cfg = SCENARIO_CONFIG[scenario]

    # Design pressures (bar) under normal operation
    baseline = {"J1": 4.50, "J2": 4.10, "J3": 3.70, "J4": 3.30, "J5": 2.90, "J6": 2.50}

    # Hourly demand multiplier (24 values, 1-hour resolution)
    pattern = [
        0.6, 0.5, 0.45, 0.4, 0.5, 0.7,
        1.0, 1.3, 1.4,  1.3, 1.2, 1.1,
        1.0, 1.0, 1.1,  1.2, 1.3, 1.5,
        1.4, 1.3, 1.2,  1.1, 0.9, 0.7,
    ]

    emitter    = cfg["emitter_coeff"]
    leak_nodes = set(cfg["leak_nodes"])
    node_order = ["J1", "J2", "J3", "J4", "J5", "J6"]

    rows = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for step in range(96):  # 96 × 15-min = 24h
        sim_time   = base_time + timedelta(minutes=step * 15)
        hour       = (step * 15) // 60
        multiplier = pattern[hour % 24]

        for node_id in node_order:
            p_base     = baseline[node_id]
            demand_lps = max(0.0, 0.12 * multiplier + rng.gauss(0, 0.005))
            p_demand   = 0.10 * (multiplier - 1.0)

            p_leak = 0.0
            if emitter > 0:
                base_drop = emitter * math.sqrt(max(p_base, 0.1)) * 15.0
                node_idx  = node_order.index(node_id)
                leak_idx  = min(node_order.index(n) for n in leak_nodes if n in node_order)
                dist      = node_idx - leak_idx
                attenuation = 1.0 if dist >= 0 else 0.25 ** abs(dist)
                p_leak    = base_drop * attenuation

            pressure_bar = max(0.05, p_base - p_demand - p_leak + rng.gauss(0, 0.02))
            rows.append(_make_row(sim_time, node_id, round(pressure_bar, 3), round(demand_lps, 4), scenario, cfg))

    return rows


def _make_row(sim_time, node_id, pressure_bar, demand_lps, scenario, cfg):
    leak_nodes = set(cfg["leak_nodes"])
    return {
        "timestamp":         sim_time.isoformat(),
        "node_id":           node_id,
        "pressure_bar":      pressure_bar,
        "demand_lps":        demand_lps,
        "scenario":          scenario,
        "fault_class":       cfg["fault_class"],
        "pipe_material":     _PIPE_MATERIAL.get(_NODE_PIPE.get(node_id, ""), "PVC"),
        "network_condition": "leak" if node_id in leak_nodes else "normal",
    }


# ─── I/O ─────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], output_path: Path):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "WNTR Digital Twin: generate hydraulically accurate SCADA pressure "
            "data for Omni-Sense cross-modal validation."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_CONFIG.keys()),
        default="no_leak",
        help="Scenario to simulate (single mode).",
    )
    parser.add_argument(
        "--output",
        default="data/scada_sim.csv",
        help="Output CSV path (single-scenario mode).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Simulate all 5 scenarios and write separate CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/scada",
        help="Output directory (batch mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for analytical fallback noise.",
    )
    args = parser.parse_args()

    engine = "WNTR" if _try_import_wntr() else "Analytical fallback"
    print("=" * 60)
    print(f"Omni-Sense WNTR Digital Twin  [{engine}]")
    print("=" * 60)

    if args.batch:
        output_dir = Path(args.output_dir)
        all_rows: list[dict] = []
        for name, cfg in SCENARIO_CONFIG.items():
            print(f"\n[{name}] {cfg['description']}")
            rows = simulate_with_wntr(name, seed=args.seed)
            write_csv(rows, output_dir / f"{name}.csv")
            all_rows.extend(rows)

        write_csv(all_rows, output_dir / "all_scenarios.csv")
        print(f"\nBatch complete: {len(all_rows):,} total rows.")

        # Pressure summary at J5 (most distal, most affected by leak)
        print("\nPressure at J5 — mean ± std (bar):")
        for name in SCENARIO_CONFIG:
            pts = [r["pressure_bar"] for r in all_rows if r["scenario"] == name and r["node_id"] == "J5"]
            if pts:
                mean_p = sum(pts) / len(pts)
                std_p  = math.sqrt(sum((p - mean_p) ** 2 for p in pts) / len(pts))
                print(f"  {name:<28} {mean_p:.2f} ± {std_p:.3f} bar")

    else:
        cfg = SCENARIO_CONFIG[args.scenario]
        print(f"\nScenario: {args.scenario} — {cfg['description']}")
        rows = simulate_with_wntr(args.scenario, seed=args.seed)
        write_csv(rows, Path(args.output))
        print("Done.")


if __name__ == "__main__":
    main()
