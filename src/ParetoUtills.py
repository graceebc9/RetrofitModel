# =============================================================================
# NEW: Expected packages-per-building range. Used for validation, not filtering.
# If your pipeline always emits exactly 6, set both to 6. Keeping a range is
# safer because some buildings may legitimately have fewer viable packages
# (e.g. already-insulated buildings drop loft options).
# =============================================================================
MIN_PACKAGES_PER_BUILDING = 1
MAX_PACKAGES_PER_BUILDING = 6

 

def build_building_level_view(df_packages, upn_col='upn'):
    """
    Collapse a multi-package dataframe to one row per building, preserving
    building-level attributes (persona, postcode, premise_type, IMD decile,
    etc.) for use as a baseline in distribution plots.

    Package-level columns (cost, carbon, intervention type) are dropped
    because they are not meaningful at the building level — a building
    has N candidate packages, not one cost.
    """
    # Columns that vary within a UPN are package-level; drop them.
    # Columns constant within a UPN are building-level; keep them.
    package_level_cols = set()
    # Sample a handful of UPNs to infer which columns vary per building.
    sample_upns = df_packages[upn_col].drop_duplicates().head(50)
    sample = df_packages[df_packages[upn_col].isin(sample_upns)]
    for col in sample.columns:
        if col == upn_col:
            continue
        # If any building has >1 unique value in this column, it's package-level.
        varies = sample.groupby(upn_col)[col].nunique(dropna=False).max()
        if varies > 1:
            package_level_cols.add(col)

    building_cols = [c for c in df_packages.columns if c not in package_level_cols]
    building_df = df_packages[building_cols].drop_duplicates(subset=upn_col, keep='first')

    print(f"  Building-level view: {len(building_df)} buildings, "
          f"{len(building_cols)} building-level columns "
          f"(dropped {len(package_level_cols)} package-level cols)")
    return building_df


def validate_multipackage_input(
    res_df,
    personas,
    upn_col='upn',
    min_packages=MIN_PACKAGES_PER_BUILDING,
    max_packages=MAX_PACKAGES_PER_BUILDING,
):
    """
    Validate a multi-package retrofit dataframe before merging with personas.

    Unlike the previous single-package validator, this:
      - EXPECTS duplicate UPNs (one per package).
      - Checks packages-per-building is within expected range.
      - Never silently drops rows — raises on anomalies so issues are visible.
      - Returns the dataframe unchanged on success (use explicit dedupe
        elsewhere if you need it).
    """
    print("\n=== PRE-MERGE VALIDATION (multi-package) ===")
    print(f"res_df rows (packages):      {len(res_df):,}")
    print(f"res_df unique UPNs:          {res_df[upn_col].nunique():,}")
    print(f"personas rows:               {len(personas):,}")

    # 1. Fully-duplicate rows. These are genuine duplicates — two identical
    #    package rows for the same building — and should be rare. Drop them
    #    but report the count so you notice if the number is unexpectedly high.
    full_dupes = res_df.duplicated().sum()
    if full_dupes > 0:
        pct = 100 * full_dupes / len(res_df)
        print(f"Fully-duplicate rows:        {full_dupes:,} ({pct:.2f}%)")
        if pct > 5:
            raise ValueError(
                f"Too many fully-duplicate rows ({pct:.1f}%). "
                f"Expected <5%. Check upstream pipeline."
            )
        res_df = res_df.drop_duplicates().reset_index(drop=True)
        print(f"  Dropped → {len(res_df):,} rows remain")
    else:
        print("Fully-duplicate rows:        0")

    # 2. Packages-per-building distribution. The key new check.
    pkg_counts = res_df.groupby(upn_col).size()
    print(f"Packages per building:       "
          f"min={pkg_counts.min()}, max={pkg_counts.max()}, "
          f"median={pkg_counts.median():.0f}, mean={pkg_counts.mean():.2f}")

    over = (pkg_counts > max_packages).sum()
    under = (pkg_counts < min_packages).sum()
    if over > 0:
        raise ValueError(
            f"{over} building(s) have >{max_packages} packages. "
            f"Max found: {pkg_counts.max()}. Widen MAX_PACKAGES_PER_BUILDING "
            f"or check for an exploded merge upstream."
        )
    if under > 0:
        # Treat as warning, not error — a building with 0 packages shouldn't
        # exist, but 1–4 packages for a building is plausible.
        print(f"  ⚠️  {under} buildings have <{min_packages} packages "
              f"(min={pkg_counts.min()}). Not blocking.")

    # 3. Personas dedupe (unchanged from before — postcode should be unique).
    personas_dupes = personas['postcode'].duplicated().sum()
    print(f"Duplicate postcodes in personas: {personas_dupes}")
    if personas_dupes > 0:
        raise ValueError(
            f"personas has {personas_dupes} duplicate postcodes. "
            f"Dedupe upstream — a many-to-many merge here would explode "
            f"the package table."
        )

    # 4. Postcode overlap.
    common_postcodes = set(res_df['postcode']) & set(personas['postcode'])
    res_postcodes = res_df['postcode'].nunique()
    print(f"res_df postcodes:            {res_postcodes:,}")
    print(f"Postcodes in common:         {len(common_postcodes):,} "
          f"({100*len(common_postcodes)/res_postcodes:.1f}% of res_df)")
    if len(common_postcodes) / res_postcodes < 0.5:
        print("  ⚠️  <50% postcode overlap — check postcode formatting "
              "(spaces, case) in both tables.")

    return res_df


def validate_post_merge(df, upn_col='upn', max_packages=MAX_PACKAGES_PER_BUILDING):
    """
    Validate the merged (packages × personas) dataframe.
    The key check: persona merge must not have inflated rows-per-building.
    """
    print(f"\n=== POST-MERGE VALIDATION ===")
    print(f"Merged rows:                 {len(df):,}")
    print(f"Unique UPNs:                 {df[upn_col].nunique():,}")

    pkg_counts = df.groupby(upn_col).size()
    print(f"Packages per building:       "
          f"min={pkg_counts.min()}, max={pkg_counts.max()}, "
          f"mean={pkg_counts.mean():.2f}")

    if pkg_counts.max() > max_packages:
        # This would indicate the persona merge duplicated rows — which means
        # personas had duplicate postcodes that slipped through.
        offenders = pkg_counts[pkg_counts > max_packages].head(5)
        raise ValueError(
            f"Merge inflated packages-per-building above {max_packages}. "
            f"Worst offenders:\n{offenders}\n"
            f"This means the persona merge was many-to-many. "
            f"Dedupe personas on postcode before merging."
        )

    # Persona coverage
    persona_col = 'meta_socio_persona'
    if persona_col in df.columns:
        n_missing = df[persona_col].isna().sum()
        if n_missing > 0:
            pct = 100 * n_missing / len(df)
            print(f"  ⚠️  {n_missing} rows ({pct:.1f}%) missing persona")


# =============================================================================
# DROP-IN REPLACEMENT for the validation block in main()
# =============================================================================
# Replaces lines ~343-379 in your current script. Everything from
#   "print("\nLoading input data...")"
# through
#   "print(f"After filtering: {len(df)} rows")"
# becomes:

"""
print("\nLoading input data...")
res_df = load_data_simple(files_to_use)
print(f'res_df shape (with any upstream dupes): {res_df.shape}')
print(f'num upns: {res_df.upn.nunique()}')

print("\nLoading personas...")
personas = load_personas()
personas = personas.drop_duplicates()

# Validate multi-package input — will raise on anomalies.
res_df = validate_multipackage_input(
    res_df, personas,
    upn_col='upn',
    min_packages=MIN_PACKAGES_PER_BUILDING,
    max_packages=MAX_PACKAGES_PER_BUILDING,
)

df = res_df.merge(personas, on='postcode', how='inner')
validate_post_merge(df, upn_col='upn', max_packages=MAX_PACKAGES_PER_BUILDING)

df = df[df['premise_type'] != 'Domestic_outbuilding']
df = df[~df['premise_type'].isna()]
gc.collect()
print(f"After premise filtering: {len(df):,} rows "
      f"({df['upn'].nunique():,} buildings)")
"""


# =============================================================================
# BASELINE PLOT FIX
# =============================================================================
# In run_pareto(), replace the plot_greedy_distribution_analysis call.
#
# OLD:
#     plot_greedy_distribution_analysis(
#         baseline_df=df_all_packages,   # 5-6 rows per building → wrong
#         selected_df=selected_df,
#         ...
#     )
#
# NEW: build the building-level view once at the top of run_pareto, then
# pass that as baseline_df. This preserves the plot's original meaning
# ("distribution of all buildings by decile and persona").

"""
def run_pareto(df_all_packages, budget, equity_floors, ...):
    os.makedirs(output_dir, exist_ok=True)

    # Build building-level view once for use as plot baseline.
    # This represents the full universe of candidate buildings — one row
    # each — independent of which package is picked.
    df_buildings = build_building_level_view(df_all_packages, upn_col=upn_col)

    ...

    for eps in equity_floors:
        selected_df, stats = multichoice_knapsack(...)
        all_stats.append(stats)

        if not selected_df.empty:
            eq_label = f"{eps:.0f}"
            selected_path = os.path.join(output_dir, f'selected_projects_eq{eq_label}.csv')
            selected_df.to_csv(selected_path, index=False)
            try:
                plot_greedy_distribution_analysis(
                    baseline_df=df_buildings,     # ← CHANGED
                    selected_df=selected_df,
                    scenario_name=f'pareto_eq{eq_label}_loft{loft_prob}',
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"  Plot failed for eq={eps}: {e}")
"""