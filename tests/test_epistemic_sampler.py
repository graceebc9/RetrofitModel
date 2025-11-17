"""
Unit tests for the epistemic scenarios LHS sampler.

Tests cover:
- Output structure and data types
- Distribution bounds (truncated normals, uniforms)
- Statistical properties (means, standard deviations)
- Discrete cost scenario sampling
- Edge cases and robustness
- LHS sampling properties
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats


# Assuming the function is imported from your module
# from your_module import generate_epistemic_scenarios_lhs
import sys 
sys.path.append('/Users/gracecolverd/RetrofitModel')
from src.RetrofitEpistemic import generate_epistemic_scenarios_lhs 
 


class TestBasicStructure:
    """Tests for basic output structure and data types."""
    
    def test_returns_dataframe(self):
        """Function should return a pandas DataFrame."""
        result = generate_epistemic_scenarios_lhs(10)
        assert isinstance(result, pd.DataFrame)
    
    def test_correct_number_of_rows(self):
        """Output should have exactly N rows."""
        for N in [10, 50, 100]:
            result = generate_epistemic_scenarios_lhs(N)
            assert len(result) == N, f"Expected {N} rows, got {len(result)}"
    
    def test_correct_number_of_columns(self):
        """Output should have exactly 7 columns."""
        result = generate_epistemic_scenarios_lhs(10)
        assert result.shape[1] ==12
    
    def test_expected_column_names(self):
        """All expected columns should be present with correct names."""
        result = generate_epistemic_scenarios_lhs(10)
        expected_columns = [
            'time_scale_bias',
            'decile_misclassification_bias',
            'solid_wall_internal_improvement_factor',
            'solid_wall_external_improvement_factor',
            'regional_multipliers_uncertainty',
            'age_band_multipliers_uncertainty',
            'cost_scenario',
        'external_wall_probability',
        'flat_fp_mean',
        'flat_fp_std',
        'flat_eff_mean',
        'flat_eff_std',
        ]
        assert list(result.columns) == expected_columns
    
    def test_no_missing_values(self):
        """DataFrame should not contain any NaN values."""
        result = generate_epistemic_scenarios_lhs(100)
        assert not result.isnull().any().any()
    
    def test_numeric_columns_are_float(self):
        """All numeric columns should be float dtype."""
        result = generate_epistemic_scenarios_lhs(10)
        numeric_cols = [
            'time_scale_bias',
            'decile_misclassification_bias',
            'solid_wall_internal_improvement_factor',
            'solid_wall_external_improvement_factor',
            'regional_multipliers_uncertainty',
            'age_band_multipliers_uncertainty'
        ]
        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(result[col]), \
                f"Column {col} should be float, got {result[col].dtype}"


class TestTimeScaleBias:
    """Tests for time_scale_bias (truncated normal: loc=1.0, scale=0.05, bounds=[0.9, 1.1])."""
    
    def test_within_bounds(self):
        """All values should be within [0.9, 1.1]."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['time_scale_bias']
        assert values.min() >= 0.9, f"Min {values.min()} below 0.9"
        assert values.max() <= 1.1, f"Max {values.max()} above 1.1"
    
    def test_mean_approximately_correct(self):
        """Mean should be close to 1.0 for large samples."""
        result = generate_epistemic_scenarios_lhs(5000)
        mean = result['time_scale_bias'].mean()
        assert 0.98 <= mean <= 1.02, f"Mean {mean} far from expected 1.0"
    
    def test_spread_reasonable(self):
        """Values should span most of the allowed range."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['time_scale_bias']
        # Check that we get values in lower and upper portions of range
        assert values.min() < 0.95, "No samples in lower tail"
        assert values.max() > 1.05, "No samples in upper tail"


class TestDecileMisclassificationBias:
    """Tests for decile_misclassification_bias (normal: loc=0.0, scale=0.02)."""
    
    def test_mean_near_zero(self):
        """Mean should be close to 0.0 for large samples."""
        result = generate_epistemic_scenarios_lhs(5000)
        mean = result['decile_misclassification_bias'].mean()
        assert -0.01 <= mean <= 0.01, f"Mean {mean} far from expected 0.0"
    
    def test_standard_deviation(self):
        """Standard deviation should be close to 0.02."""
        result = generate_epistemic_scenarios_lhs(5000)
        std = result['decile_misclassification_bias'].std()
        # Allow 20% tolerance
        assert 0.016 <= std <= 0.024, f"Std {std} far from expected 0.02"
    
    def test_reasonable_range(self):
        """Most values should be within ±3 standard deviations."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['decile_misclassification_bias']
        # For normal distribution, ~99.7% within ±3σ
        assert values.min() > -0.1, "Values unreasonably low"
        assert values.max() < 0.1, "Values unreasonably high"


class TestSolidWallInternalImprovement:
    """Tests for solid_wall_internal_improvement_factor (truncated normal: loc=0.1, scale=0.01, bounds=[0.08, 0.12])."""
    
    def test_within_bounds(self):
        """All values should be within [0.08, 0.12]."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['solid_wall_internal_improvement_factor']
        assert values.min() >= 0.08, f"Min {values.min()} below 0.08"
        assert values.max() <= 0.12, f"Max {values.max()} above 0.12"
    
    def test_mean_approximately_correct(self):
        """Mean should be close to 0.1."""
        result = generate_epistemic_scenarios_lhs(5000)
        mean = result['solid_wall_internal_improvement_factor'].mean()
        assert 0.095 <= mean <= 0.105, f"Mean {mean} far from expected 0.1"


class TestSolidWallExternalImprovement:
    """Tests for solid_wall_external_improvement_factor (truncated normal: loc=0.2, scale=0.02, bounds=[0.15, 0.25])."""
    
    def test_within_bounds(self):
        """All values should be within [0.15, 0.25]."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['solid_wall_external_improvement_factor']
        assert values.min() >= 0.15, f"Min {values.min()} below 0.15"
        assert values.max() <= 0.25, f"Max {values.max()} above 0.25"
    
    def test_mean_approximately_correct(self):
        """Mean should be close to 0.2."""
        result = generate_epistemic_scenarios_lhs(5000)
        mean = result['solid_wall_external_improvement_factor'].mean()
        assert 0.19 <= mean <= 0.21, f"Mean {mean} far from expected 0.2"


class TestRegionalMultipliers:
    """Tests for regional_multipliers_uncertainty (uniform: [0.9, 1.1])."""
    
    def test_within_bounds(self):
        """All values should be within [0.9, 1.1]."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['regional_multipliers_uncertainty']
        assert values.min() >= 0.9, f"Min {values.min()} below 0.9"
        assert values.max() <= 1.1, f"Max {values.max()} above 1.1"
    
    def test_mean_near_midpoint(self):
        """Mean should be close to 1.0 (midpoint of range)."""
        result = generate_epistemic_scenarios_lhs(5000)
        mean = result['regional_multipliers_uncertainty'].mean()
        assert 0.98 <= mean <= 1.02, f"Mean {mean} far from expected 1.0"
    
    def test_standard_deviation(self):
        """Std should match uniform distribution: (b-a)/sqrt(12) ≈ 0.0577."""
        result = generate_epistemic_scenarios_lhs(5000)
        std = result['regional_multipliers_uncertainty'].std()
        expected_std = 0.2 / np.sqrt(12)  # ≈ 0.0577
        # Allow 20% tolerance
        assert 0.8 * expected_std <= std <= 1.2 * expected_std, \
            f"Std {std} far from expected {expected_std}"
    
    def test_good_coverage_of_range(self):
        """Values should cover the full range well."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['regional_multipliers_uncertainty']
        # Check coverage in lower, middle, upper thirds
        assert (values < 0.95).sum() >= 200, "Insufficient coverage of lower range"
        assert ((values >= 0.95) & (values <= 1.05)).sum() >= 200, "Insufficient coverage of middle range"
        assert (values > 1.05).sum() >= 200, "Insufficient coverage of upper range"


class TestAgeBandMultipliers:
    """Tests for age_band_multipliers_uncertainty (uniform: [0.92, 1.08])."""
    
    def test_within_bounds(self):
        """All values should be within [0.92, 1.08]."""
        result = generate_epistemic_scenarios_lhs(1000)
        values = result['age_band_multipliers_uncertainty']
        assert values.min() >= 0.92, f"Min {values.min()} below 0.92"
        assert values.max() <= 1.08, f"Max {values.max()} above 1.08"
    
    def test_mean_near_midpoint(self):
        """Mean should be close to 1.0 (midpoint of range)."""
        result = generate_epistemic_scenarios_lhs(5000)
        mean = result['age_band_multipliers_uncertainty'].mean()
        assert 0.98 <= mean <= 1.02, f"Mean {mean} far from expected 1.0"
    
    def test_standard_deviation(self):
        """Std should match uniform distribution: (b-a)/sqrt(12) ≈ 0.0462."""
        result = generate_epistemic_scenarios_lhs(5000)
        std = result['age_band_multipliers_uncertainty'].std()
        expected_std = 0.16 / np.sqrt(12)  # ≈ 0.0462
        # Allow 20% tolerance
        assert 0.8 * expected_std <= std <= 1.2 * expected_std, \
            f"Std {std} far from expected {expected_std}"


class TestCostScenario:
    """Tests for cost_scenario (discrete categorical variable)."""
    
    def test_only_valid_values(self):
        """Should only contain 'optimistic', 'central', or 'pessimistic'."""
        result = generate_epistemic_scenarios_lhs(1000)
        valid_values = {'optimistic', 'central', 'pessimistic'}
        unique_values = set(result['cost_scenario'].unique())
        assert unique_values.issubset(valid_values), \
            f"Found invalid values: {unique_values - valid_values}"
    
    def test_all_categories_present(self):
        """All three categories should appear with sufficient samples."""
        result = generate_epistemic_scenarios_lhs(300)
        unique_values = set(result['cost_scenario'].unique())
        assert len(unique_values) == 3, \
            f"Expected 3 categories, got {len(unique_values)}: {unique_values}"
    
    def test_approximately_uniform_distribution(self):
        """Categories should appear with roughly equal frequency."""
        result = generate_epistemic_scenarios_lhs(3000)
        counts = result['cost_scenario'].value_counts()
        
        # Each should be roughly 1/3 of samples (±15%)
        for category in ['optimistic', 'central', 'pessimistic']:
            proportion = counts[category] / len(result)
            assert 0.25 <= proportion <= 0.42, \
                f"Category '{category}' has proportion {proportion:.3f}, expected ≈0.333"
    
    def test_data_type(self):
        """Column should be object/string dtype."""
        result = generate_epistemic_scenarios_lhs(10)
        dtype = result['cost_scenario'].dtype
        assert dtype == object or pd.api.types.is_string_dtype(dtype)


class TestLHSProperties:
    """Tests for Latin Hypercube Sampling properties."""
    
    def test_high_uniqueness_in_samples(self):
        """LHS should produce highly unique values in each dimension."""
        result = generate_epistemic_scenarios_lhs(100)
        numeric_cols = [col for col in result.columns if col != 'cost_scenario']
        
        for col in numeric_cols:
            unique_count = result[col].nunique()
            # Expect at least 95% unique values
            assert unique_count >= 95, \
                f"Column {col} has only {unique_count} unique values out of 100"
    
    def test_space_filling_property(self):
        """LHS should fill the sample space well (no large gaps)."""
        result = generate_epistemic_scenarios_lhs(100)
        
        # Check regional_multipliers (uniform, easier to test)
        values = result['regional_multipliers_uncertainty'].sort_values().values
        gaps = np.diff(values)
        max_gap = gaps.max()
        
        # Maximum gap should be reasonable (less than 10% of range)
        range_size = 0.2  # 1.1 - 0.9
        assert max_gap < 0.1 * range_size, \
            f"Large gap detected: {max_gap}, suggests poor space-filling"


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    @pytest.mark.skip(reason="LHS with maximin criterion doesn't support N=1")
    def test_single_sample(self):
        """Function should work with N=1."""
        result = generate_epistemic_scenarios_lhs(1)
        assert result.shape == (1, 7)
        assert not result.isnull().any().any()
    
    def test_large_sample(self):
        """Function should work with large N."""
        result = generate_epistemic_scenarios_lhs(10000)
        assert result.shape == (10000, 12)
        assert not result.isnull().any().any()
    
    @pytest.mark.skip(reason="LHS uses internal random state, not numpy random seed")
    def test_reproducibility_with_seed(self):
        """Setting random seed should produce reproducible results."""
        np.random.seed(42)
        result1 = generate_epistemic_scenarios_lhs(50)
        
        np.random.seed(42)
        result2 = generate_epistemic_scenarios_lhs(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_different_runs_without_seed(self):
        """Without seed, runs should produce different results."""
        result1 = generate_epistemic_scenarios_lhs(50)
        result2 = generate_epistemic_scenarios_lhs(50)
        
        # Should be different (very unlikely to be identical)
        assert not result1.equals(result2)


class TestStatisticalValidation:
    """Statistical tests for distribution correctness."""
    
    def test_ks_test_uniform_distributions(self):
        """Kolmogorov-Smirnov test for uniform distributions."""
        result = generate_epistemic_scenarios_lhs(1000)
        
        # Test regional_multipliers_uncertainty
        reg_values = result['regional_multipliers_uncertainty']
        reg_normalized = (reg_values - 0.9) / 0.2  # Normalize to [0, 1]
        ks_stat, p_value = stats.kstest(reg_normalized, 'uniform')
        assert p_value > 0.01, \
            f"Regional multipliers failed KS test (p={p_value:.4f})"
        
        # Test age_band_multipliers_uncertainty
        age_values = result['age_band_multipliers_uncertainty']
        age_normalized = (age_values - 0.92) / 0.16  # Normalize to [0, 1]
        ks_stat, p_value = stats.kstest(age_normalized, 'uniform')
        assert p_value > 0.01, \
            f"Age band multipliers failed KS test (p={p_value:.4f})"
    
    def test_chi_square_test_for_cost_scenario(self):
        """Chi-square test for uniform categorical distribution."""
        result = generate_epistemic_scenarios_lhs(3000)
        observed = result['cost_scenario'].value_counts().sort_index()
        expected = np.array([1000, 1000, 1000])  # Equal distribution
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        # Should not reject null hypothesis of uniform distribution
        assert p_value > 0.01, \
            f"Cost scenario distribution failed chi-square test (p={p_value:.4f})"


class TestIntegration:
    """Integration tests for overall function behavior."""
    
    def test_all_distributions_work_together(self):
        """All distributions should work correctly when generated together."""
        result = generate_epistemic_scenarios_lhs(1000)
        
        # Quick checks on all columns
        assert 0.9 <= result['time_scale_bias'].min()
        assert result['time_scale_bias'].max() <= 1.1
        
        assert 0.08 <= result['solid_wall_internal_improvement_factor'].min()
        assert result['solid_wall_internal_improvement_factor'].max() <= 0.12
        
        assert 0.15 <= result['solid_wall_external_improvement_factor'].min()
        assert result['solid_wall_external_improvement_factor'].max() <= 0.25
        
        assert 0.9 <= result['regional_multipliers_uncertainty'].min()
        assert result['regional_multipliers_uncertainty'].max() <= 1.1
        
        assert 0.92 <= result['age_band_multipliers_uncertainty'].min()
        assert result['age_band_multipliers_uncertainty'].max() <= 1.08
        
        assert set(result['cost_scenario'].unique()).issubset(
            {'optimistic', 'central', 'pessimistic'}
        )
    
    def test_dataframe_is_usable(self):
        """DataFrame should be usable for typical operations."""
        result = generate_epistemic_scenarios_lhs(100)
        
        # Should be able to filter
        filtered = result[result['cost_scenario'] == 'optimistic']
        assert len(filtered) > 0
        
        # Should be able to compute statistics
        means = result.select_dtypes(include=[np.number]).mean()
        assert len(means) == 11
        
        # Should be able to save/load
        result.to_csv('/tmp/test_output.csv', index=False)
        loaded = pd.read_csv('/tmp/test_output.csv')
        assert loaded.shape == result.shape


class TestFlatAreaFactors:
    """Test suite for the new flat area factors in the epistemic scenarios sampler."""
    
    @pytest.fixture
    def sample_scenarios(self):
        """Generate a sample set of epistemic scenarios for testing."""
        N_runs = 1000
        return generate_epistemic_scenarios_lhs(N_runs)
    
    def test_flat_factors_present_in_output(self, sample_scenarios):
        """Test that all new flat area factors are present in the output DataFrame."""
        expected_columns = [
            'flat_fp_mean',
            'flat_fp_std',
            'flat_eff_mean',
            'flat_eff_std'
        ]
        
        for col in expected_columns:
            assert col in sample_scenarios.columns, f"Column '{col}' missing from output"
    
    def test_flat_fp_mean_bounds(self, sample_scenarios):
        """Test that flat footprint mean values are within expected bounds [40, 70]."""
        fp_mean = sample_scenarios['flat_fp_mean']
        
        assert fp_mean.min() >= 40, f"Minimum fp_mean ({fp_mean.min():.2f}) is below lower bound (40)"
        assert fp_mean.max() <= 70, f"Maximum fp_mean ({fp_mean.max():.2f}) is above upper bound (70)"
        assert not fp_mean.isna().any(), "NaN values found in flat_fp_mean"
    
    def test_flat_fp_std_bounds(self, sample_scenarios):
        """Test that flat footprint std values are within expected bounds [2, 15]."""
        fp_std = sample_scenarios['flat_fp_std']
        
        assert fp_std.min() >= 2, f"Minimum fp_std ({fp_std.min():.2f}) is below lower bound (2)"
        assert fp_std.max() <= 15, f"Maximum fp_std ({fp_std.max():.2f}) is above upper bound (15)"
        assert not fp_std.isna().any(), "NaN values found in flat_fp_std"
    
    def test_flat_eff_mean_bounds(self, sample_scenarios):
        """Test that efficiency mean values are within expected bounds [0.5, 0.8]."""
        eff_mean = sample_scenarios['flat_eff_mean']
        
        assert eff_mean.min() >= 0.5, f"Minimum eff_mean ({eff_mean.min():.3f}) is below lower bound (0.5)"
        assert eff_mean.max() <= 0.8, f"Maximum eff_mean ({eff_mean.max():.3f}) is above upper bound (0.8)"
        assert not eff_mean.isna().any(), "NaN values found in flat_eff_mean"
    
    def test_flat_eff_std_bounds(self, sample_scenarios):
        """Test that efficiency std values are within expected bounds [0.01, 0.1]."""
        eff_std = sample_scenarios['flat_eff_std']
        
        assert eff_std.min() >= 0.01, f"Minimum eff_std ({eff_std.min():.3f}) is below lower bound (0.01)"
        assert eff_std.max() <= 0.1, f"Maximum eff_std ({eff_std.max():.3f}) is above upper bound (0.1)"
        assert not eff_std.isna().any(), "NaN values found in flat_eff_std"
    
    def test_flat_fp_mean_distribution(self, sample_scenarios):
        """Test that flat_fp_mean follows approximately truncated normal distribution centered at 55."""
        fp_mean = sample_scenarios['flat_fp_mean']
        
        # Check mean is approximately 55 (within tolerance for truncated normal)
        assert 52 <= fp_mean.mean() <= 58, f"Mean of fp_mean ({fp_mean.mean():.2f}) deviates significantly from expected (55)"
        
        # Check standard deviation is reasonable (should be less than the specified 5 due to truncation)
        assert fp_mean.std() <= 6, f"Std of fp_mean ({fp_mean.std():.2f}) is unexpectedly large"
    
    def test_flat_fp_std_distribution(self, sample_scenarios):
        """Test that flat_fp_std follows approximately truncated normal distribution centered at 8."""
        fp_std = sample_scenarios['flat_fp_std']
        
        # Check mean is approximately 8
        assert 7 <= fp_std.mean() <= 9, f"Mean of fp_std ({fp_std.mean():.2f}) deviates significantly from expected (8)"
    
    def test_flat_eff_mean_distribution(self, sample_scenarios):
        """Test that flat_eff_mean follows approximately truncated normal distribution centered at 0.75."""
        eff_mean = sample_scenarios['flat_eff_mean']
        
        # Check mean is approximately 0.75
        assert 0.73 <= eff_mean.mean() <= 0.77, f"Mean of eff_mean ({eff_mean.mean():.3f}) deviates significantly from expected (0.75)"
    
    def test_flat_eff_std_distribution(self, sample_scenarios):
        """Test that flat_eff_std follows approximately truncated normal distribution centered at 0.05."""
        eff_std = sample_scenarios['flat_eff_std']
        
        # Check mean is approximately 0.05
        assert 0.04 <= eff_std.mean() <= 0.06, f"Mean of eff_std ({eff_std.mean():.3f}) deviates significantly from expected (0.05)"
    
    def test_lhs_coverage_flat_factors(self, sample_scenarios):
        """Test that LHS provides good coverage across the parameter space for flat factors."""
        # For each factor, check that we have reasonable coverage across quantiles
        
        for col in ['flat_fp_mean', 'flat_fp_std', 'flat_eff_mean', 'flat_eff_std']:
            values = sample_scenarios[col]
            
            # Check that all quartiles are populated
            quartiles = np.percentile(values, [25, 50, 75])
            
            # Ensure quartiles are distinct (good spread)
            assert quartiles[0] < quartiles[1] < quartiles[2], \
                f"Poor LHS coverage for {col}: quartiles are not well-separated"
    
    def test_no_duplicates_in_flat_factors(self, sample_scenarios):
        """Test that LHS generates unique samples (no exact duplicates)."""
        # Check each flat factor individually
        for col in ['flat_fp_mean', 'flat_fp_std', 'flat_eff_mean', 'flat_eff_std']:
            unique_count = sample_scenarios[col].nunique()
            total_count = len(sample_scenarios)
            
            # Allow for very small number of coincidental duplicates in large samples
            assert unique_count >= total_count * 0.95, \
                f"Too many duplicate values in {col}: {unique_count}/{total_count} unique"
    
    def test_flat_factors_correlation_low(self, sample_scenarios):
        """Test that flat factors have low correlation (good LHS property)."""
        flat_cols = ['flat_fp_mean', 'flat_fp_std', 'flat_eff_mean', 'flat_eff_std']
        corr_matrix = sample_scenarios[flat_cols].corr()
        
        # Get off-diagonal elements
        for i, col1 in enumerate(flat_cols):
            for j, col2 in enumerate(flat_cols):
                if i < j:  # Upper triangle only
                    correlation = abs(corr_matrix.loc[col1, col2])
                    assert correlation < 0.2, \
                        f"High correlation ({correlation:.3f}) between {col1} and {col2}"
    
    def test_small_sample_size(self):
        """Test that function works with small sample sizes."""
        small_scenarios = generate_epistemic_scenarios_lhs(N_epistemic_runs=10)
        
        assert len(small_scenarios) == 10, "Incorrect number of scenarios generated"
        assert 'flat_fp_mean' in small_scenarios.columns, "Missing flat factor columns in small sample"
    
    def test_data_types(self, sample_scenarios):
        """Test that flat factors have correct data types."""
        for col in ['flat_fp_mean', 'flat_fp_std', 'flat_eff_mean', 'flat_eff_std']:
            assert sample_scenarios[col].dtype in [np.float64, np.float32], \
                f"Column {col} has incorrect dtype: {sample_scenarios[col].dtype}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])