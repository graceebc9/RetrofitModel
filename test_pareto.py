"""
test_multichoice_knapsack.py
=============================
Test harness for validating the multi-choice knapsack solver on a small
sample of data. Runs three tests:

  1. Multi-choice vs pre-select: does the new method beat the old one?
  2. Equity constraint: does raising the floor actually bind, and does
     abatement drop monotonically?
  3. Package diversity: is the solver actually using the multi-choice
     freedom, or does it mostly pick each building's min-£/tCO2 package?

Usage:
    python test_multichoice_knapsack.py

Edit CONFIG block below to point at your data.
"""

import os
import sys
import glob
import gc
import numpy as np
import pandas as pd

sys.path.append('/Users/gracecolverd/RetrofitModel')

from src.personas import load_personas
from src.ParetoKnapsack import (
    multichoice_knapsack,
    preselect_best_cpt,
    DEFAULT_HIGH_EQUITY_PERSONAS,
)


# ============================================================================
# CONFIG — edit paths and sample size here
# ============================================================================

INPUT_FILES_PATH = (
 '/Volumes/T9/2025_10_RetrofitModel/12_v2_greedy/1_all_interventions/risk_sigma_1.0/processed_all_scenarios/*'
)

INPUT_FILES_PATH = (
'/home/gb669/rds/hpc-work/energy_map/RetrofitModel/4_optimized_priorities/risk_sigma_1.0/processed_all_scenarios/*'
)

 

LOFT_PROB = 0.65
SAMPLE_SIZE = 2000       # buildings
RANDOM_SEED = 42

# Budgets to test. Scaled for SAMPLE_SIZE buildings.
# Full data has 678k buildings, ~£10-15k avg retrofit cost.
# For 1000 buildings, full-retrofit budget ≈ £10-15M.
TEST_BUDGETS = [500_000, 2_000_000, 10_000_000]

# Equity floors for Test 2
EQUITY_FLOORS = [0, 25, 50, 75, 100]

COST_COL = 'mean_total_capex'
CARBON_COL = 'mean_total_co2_saved'
UPN_COL = 'upn'
PERSONA_COL = 'meta_socio_persona'


# ============================================================================
# DATA LOADING (mirrors main script)
# ============================================================================

def load_data_simple(files):
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_and_clean(files_path, loft_prob):
    input_files = glob.glob(files_path)
    files_to_use = [x for x in input_files if f'loft_{loft_prob}' in x]
    print(f"Found {len(files_to_use)} files with loft_prob={loft_prob}")
    if not files_to_use:
        raise RuntimeError(f"No files match loft_{loft_prob} in {files_path}")

    print("Loading files...")
    res_df = load_data_simple(files_to_use)
    print(f"  Raw: {len(res_df):,} rows, {res_df['upn'].nunique():,} UPNs")

    # Drop UPN-postcode collisions (upstream quirk)
    bad_upns = res_df.groupby('upn')['postcode'].nunique()
    bad_upns = bad_upns[bad_upns > 1].index
    if len(bad_upns) > 0:
        res_df = res_df[~res_df['upn'].isin(bad_upns)].reset_index(drop=True)
        print(f"  Dropped {len(bad_upns)} UPN-postcode collisions")

    print("Loading personas...")
    personas = load_personas().drop_duplicates()

    df = res_df.merge(personas, on='postcode', how='inner')
    df = df[df['premise_type'] != 'Domestic_outbuilding']
    df = df[~df['premise_type'].isna()]
    df = df.reset_index(drop=True)
    print(f"  After merge + filter: {len(df):,} rows, "
          f"{df['upn'].nunique():,} buildings")

    gc.collect()
    return df


# ============================================================================
# STRATIFIED SAMPLING
# ============================================================================

def sample_for_testing(df_all, n_buildings=1000, seed=42):
    """
    Stratified sample on (persona, package_count) so the sample
    preserves:
      - Persona mix (for meaningful equity constraint tests)
      - Menu diversity (so multi-choice has real choices)
      - Package-count distribution
    """
    rng = np.random.default_rng(seed)

    # One row per building with stratification keys
    building_summary = (
        df_all.groupby('upn')
        .agg(
            persona=(PERSONA_COL, 'first'),
            n_packages=('intervention', 'count'),
        )
    )
    total = len(building_summary)
    print(f"\nSampling {n_buildings} buildings from {total:,} "
          f"(stratified on persona × package_count)")

    strata = building_summary.groupby(['persona', 'n_packages']).size()

    sampled_upns = []
    for (persona, n_pkg), count in strata.items():
        target = max(1, int(round(n_buildings * count / total)))
        stratum_upns = building_summary[
            (building_summary['persona'] == persona) &
            (building_summary['n_packages'] == n_pkg)
        ].index.to_numpy()
        k = min(target, len(stratum_upns))
        if k > 0:
            pick = rng.choice(stratum_upns, size=k, replace=False)
            sampled_upns.extend(pick.tolist())

    sampled_upns = list(set(sampled_upns))
    sample = df_all[df_all['upn'].isin(sampled_upns)].copy()
    sample = sample.reset_index(drop=True)

    print(f"  Actual sample: {sample['upn'].nunique()} buildings, "
          f"{len(sample)} package rows")
    persona_mix = (sample.drop_duplicates('upn')[PERSONA_COL]
                   .value_counts().to_dict())
    print(f"  Persona mix: {persona_mix}")
    pkg_dist = sample.groupby('upn').size().value_counts().sort_index()
    print(f"  Packages per building: {pkg_dist.to_dict()}")
    return sample


# ============================================================================
# TEST 1: Multi-choice vs pre-select
# ============================================================================

def test_multichoice_vs_preselect(sample_df, budget):
    print(f"\n{'=' * 70}")
    print(f"TEST 1: Multi-choice vs pre-select | Budget £{budget:,.0f}")
    print(f"{'=' * 70}")

    # Old method: pre-select best £/tCO2 per building, then knapsack
    df_preselected = preselect_best_cpt(
        sample_df, upn_col=UPN_COL,
        cost_col=COST_COL, carbon_col=CARBON_COL,
    )
    pre_out, pre_stats = multichoice_knapsack(
        df_preselected, budget=budget, equity_floor_pct=0,
        upn_col=UPN_COL, persona_col=PERSONA_COL,
        cost_col=COST_COL, carbon_col=CARBON_COL,
    )

    # New method: full multi-choice
    mc_out, mc_stats = multichoice_knapsack(
        sample_df, budget=budget, equity_floor_pct=0,
        upn_col=UPN_COL, persona_col=PERSONA_COL,
        cost_col=COST_COL, carbon_col=CARBON_COL,
    )

    pre_ab = pre_stats['total_abatement'] or 0
    mc_ab = mc_stats['total_abatement'] or 0
    print(f"\nPre-select:   {pre_ab:>10.1f} tCO2, "
          f"{pre_stats['n_retrofitted']:>5} bldgs, "
          f"£{pre_stats['total_cost']:>12,.0f}, "
          f"{pre_stats.get('solve_time_s', 0):.1f}s")
    print(f"Multi-choice: {mc_ab:>10.1f} tCO2, "
          f"{mc_stats['n_retrofitted']:>5} bldgs, "
          f"£{mc_stats['total_cost']:>12,.0f}, "
          f"{mc_stats.get('solve_time_s', 0):.1f}s")

    if pre_ab > 0:
        delta = mc_ab - pre_ab
        pct = 100 * delta / pre_ab
        print(f"Improvement:  {delta:>+10.1f} tCO2 ({pct:+.2f}%)")

    # INVARIANT: multi-choice ≥ pre-select (strict superset of feasible region)
    if mc_ab < pre_ab - 0.01:
        print(f"\n  ✗ FAIL: multi-choice < pre-select "
              f"({mc_ab:.2f} < {pre_ab:.2f}). "
              f"The multi-choice feasible region contains pre-select's, "
              f"so this should never happen.")
        return False

    # Diagnostic: which packages overlap, which differ
    if 'intervention' in mc_out.columns and 'intervention' in pre_out.columns:
        pre_map = dict(zip(pre_out[UPN_COL], pre_out['intervention']))
        mc_map = dict(zip(mc_out[UPN_COL], mc_out['intervention']))
        common = set(pre_map) & set(mc_map)
        different = sum(1 for u in common if pre_map[u] != mc_map[u])
        print(f"\nBuilding overlap:")
        print(f"  In both solutions:       {len(common)}")
        print(f"    Same package:          {len(common) - different}")
        print(f"    Different package:     {different}")
        print(f"  Only in pre-select:      {len(set(pre_map) - set(mc_map))}")
        print(f"  Only in multi-choice:    {len(set(mc_map) - set(pre_map))}")

    print(f"  ✓ PASS")
    return True


# ============================================================================
# TEST 2: Equity constraint binds
# ============================================================================

def test_equity_constraint(sample_df, budget, floors=None):
    print(f"\n{'=' * 70}")
    print(f"TEST 2: Equity constraint | Budget £{budget:,.0f}")
    print(f"{'=' * 70}")
    if floors is None:
        floors = EQUITY_FLOORS

    results = []
    for eps in floors:
        _, stats = multichoice_knapsack(
            sample_df, budget=budget, equity_floor_pct=eps,
            upn_col=UPN_COL, persona_col=PERSONA_COL,
            cost_col=COST_COL, carbon_col=CARBON_COL,
        )
        if stats['status'] not in ('Optimal', 'Not Solved'):
            print(f"  Infeasible at {eps}% — stopping.")
            break
        results.append({
            'floor_pct': eps,
            'actual_high_eq_pct': stats['high_eq_spend_pct'],
            'abatement_tCO2': stats['total_abatement'],
            'cpex_per_ton': stats['cpex_per_ton'],
            'n_retrofitted': stats['n_retrofitted'],
        })

    res = pd.DataFrame(results)
    print("\nResults:")
    print(res.to_string(index=False))

    if res.empty:
        print("  ✗ FAIL: no feasible solutions at any floor")
        return False

    passed = True

    # INVARIANT 1: realised high_eq_pct >= floor (tolerance 0.01)
    violations = res[res['actual_high_eq_pct'] < res['floor_pct'] - 0.01]
    if len(violations) > 0:
        print(f"\n  ✗ FAIL: Equity constraint violated at:")
        print(violations.to_string(index=False))
        passed = False
    else:
        print(f"\n  ✓ Equity constraint respected at all floors")

    # INVARIANT 2: abatement monotone decreasing (tolerance 0.01)
    diffs = res['abatement_tCO2'].diff().iloc[1:]
    if (diffs > 0.01).any():
        print(f"  ✗ FAIL: Abatement INCREASES as floor rises. "
              f"Tightening a constraint cannot improve the objective.")
        print(f"     Diffs: {diffs.tolist()}")
        passed = False
    else:
        print(f"  ✓ Abatement is monotone non-increasing as floor rises")

    # Also check: does raising the floor actually change the solution?
    if res['actual_high_eq_pct'].nunique() == 1:
        print(f"  ⚠️  Warning: high_eq_pct is constant across all floors "
              f"({res['actual_high_eq_pct'].iloc[0]:.1f}%). "
              f"Either the constraint never bound, or the sample lacks "
              f"equity diversity.")

    return passed


# ============================================================================
# TEST 3: Package diversity
# ============================================================================

def test_package_diversity(sample_df, budget):
    print(f"\n{'=' * 70}")
    print(f"TEST 3: Package diversity | Budget £{budget:,.0f}")
    print(f"{'=' * 70}")

    # Compute £/tCO2 rank within each building
    df = sample_df.copy()
    df['_cpt'] = df[COST_COL] / df[CARBON_COL].clip(lower=0.01)
    df['_rank_in_building'] = (
        df.groupby(UPN_COL)['_cpt']
        .rank(method='first', ascending=True)
        .astype(int)
    )

    # Solve
    selected, stats = multichoice_knapsack(
        sample_df, budget=budget, equity_floor_pct=0,
        upn_col=UPN_COL, persona_col=PERSONA_COL,
        cost_col=COST_COL, carbon_col=CARBON_COL,
    )

    if selected.empty:
        print("  No buildings selected — skipping diversity test")
        return True

    # Join rank onto selected
    merged = selected.merge(
        df[[UPN_COL, 'intervention', '_rank_in_building']],
        on=[UPN_COL, 'intervention'], how='left',
    )

    rank_dist = merged['_rank_in_building'].value_counts().sort_index()
    print(f"\nRank (1=cheapest £/tCO2) of selected package within its building:")
    for rank, n in rank_dist.items():
        pct = 100 * n / len(merged)
        print(f"  Rank {rank}: {n:>5} buildings ({pct:>5.1f}%)")

    non_rank1 = (merged['_rank_in_building'] > 1).sum()
    print(f"\nBuildings NOT picking their min-£/tCO2 package: "
          f"{non_rank1} / {len(merged)} "
          f"({100 * non_rank1 / len(merged):.1f}%)")

    if non_rank1 == 0:
        print(f"  ⚠️  Solver picked rank-1 for every building. "
              f"Multi-choice formulation is equivalent to pre-select here. "
              f"Could be expected (tight budget, rank-1 dominates) or "
              f"could mean the solver isn't exploring — worth checking "
              f"at higher budgets or with equity constraints active.")
    else:
        print(f"  ✓ Multi-choice formulation is exercising diverse packages")

    # Also: intervention mix
    if 'intervention' in selected.columns:
        print(f"\nIntervention mix in selection:")
        intv_counts = selected['intervention'].value_counts()
        for intv, n in intv_counts.items():
            pct = 100 * n / len(selected)
            print(f"  {intv}: {n} ({pct:.1f}%)")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MULTI-CHOICE KNAPSACK TEST HARNESS")
    print("=" * 70)
    print(f"Input:       {INPUT_FILES_PATH}")
    print(f"Loft prob:   {LOFT_PROB}")
    print(f"Sample size: {SAMPLE_SIZE} buildings")
    print(f"Budgets:     {TEST_BUDGETS}")
    print(f"Equity:      {EQUITY_FLOORS}")

    # Load + sample
    df_full = load_and_clean(INPUT_FILES_PATH, LOFT_PROB)
    sample_df = sample_for_testing(df_full, SAMPLE_SIZE, RANDOM_SEED)
    del df_full
    gc.collect()


    # Compute £/tCO2 by intervention type to understand the economics
    df = sample_df.copy()
    df['cpt'] = df[COST_COL] / df[CARBON_COL].clip(lower=0.01)
    print(df.groupby('intervention')['cpt'].describe()[['mean', '50%', '25%', '75%']])
    print("\nRank distribution by intervention (within-building rank):")
    df['rank'] = df.groupby(UPN_COL)['cpt'].rank(method='first').astype(int)
    print(pd.crosstab(df['intervention'], df['rank']))

    # Run tests at each budget
    results = []
    for budget in TEST_BUDGETS:
        t1 = test_multichoice_vs_preselect(sample_df, budget)
        t2 = test_equity_constraint(sample_df, budget)
        t3 = test_package_diversity(sample_df, budget)
        results.append({
            'budget': budget,
            'test1_multichoice_ge_preselect': t1,
            'test2_equity_monotone': t2,
            'test3_diversity_ran': t3,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = pd.DataFrame(results)
    summary['budget_M'] = summary['budget'] / 1e6
    print(summary[['budget_M',
                   'test1_multichoice_ge_preselect',
                   'test2_equity_monotone',
                   'test3_diversity_ran']].to_string(index=False))

    all_passed = (summary['test1_multichoice_ge_preselect'].all()
                  and summary['test2_equity_monotone'].all())
    print()
    if all_passed:
        print("✓ ALL CORRECTNESS INVARIANTS HELD")
    else:
        print("✗ ONE OR MORE TESTS FAILED — investigate before scaling up")
    print("=" * 70)


if __name__ == "__main__":
    main()