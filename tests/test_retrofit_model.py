import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import sys
sys.path.append('/Users/gracecolverd/RetrofitModel') 
# Assuming your module structure
 
import pytest
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, call # Use unittest.mock or pytest-mock

 
import pytest
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, call # Use unittest.mock or pytest-mock

# --- Mock Dependencies ---
# To make the test file self-contained, we mock the imported modules.
# In a real project, you would just import them.
# We use MagicMock to simulate the classes and functions.

@dataclass
class MockBuildingCharacteristics:
    floor_count: int = 1
    gross_external_area: float = 100.0
    gross_internal_area: float = 80.0
    footprint_circumference: float = 40.0
    flat_count: int = 1
    building_footprint_area: float = 80.0
    avg_gas_percentile: int = 5
    typology: str = 'Small low terraces'

@dataclass
class MockRetrofitConfig:
    existing_intervention_probs: Dict[str, float] = \
        field(default_factory=lambda: {'roof_scaling_factor': 0.8})

class MockCostEstimator:
    def sample_cost_for_package(self, *args, **kwargs):
        pass # Will be mocked at the instance level

class MockRetrofitEnergy:
    solid_wall_internal_improvement_factor: Optional[float] = None
    solid_wall_external_improvement_factor: Optional[float] = None
    
    def __init__(self, *args, **kwargs):
        self.solid_wall_internal_improvement_factor = kwargs.get('solid_wall_internal_improvement_factor')
        self.solid_wall_external_improvement_factor = kwargs.get('solid_wall_external_improvement_factor')
    
    def sample_intervention_energy_savings_monte_carlo(self, *args, **kwargs):
        pass # Will be mocked at the instance level

def mock_calc_est_flats_building(*args, **kwargs):
    return 1.0 # Default mock return

mock_retrofit_packages = {
    'wall_only': {'interventions': ['WALL_INSULATION']},
    'loft_only': {'interventions': ['loft_percentile']},
    'joint_scenario': {'interventions': ['joint_loft_wall_add']},
}

# --- Import the Code Under Test ---
# This assumes the code from the prompt is in a file named `RetrofitModel2D.py`
# We patch its dependencies *before* importing it.

@pytest.fixture(scope='module', autouse=True)
def patch_dependencies(module_mocker):
    """
    Patches all external dependencies for the entire test module.
    This allows us to import RetrofitModel2D without the real dependencies.
    """
    module_mocker.patch('src.RetrofitModel2D.BuildingCharacteristics', MockBuildingCharacteristics)
    module_mocker.patch('src.RetrofitModel2D.RetrofitConfig', MockRetrofitConfig)
    module_mocker.patch('src.RetrofitModel2D.CostEstimator', MockCostEstimator)
    module_mocker.patch('src.RetrofitModel2D.RetrofitEnergy', MockRetrofitEnergy)
    module_mocker.patch('src.RetrofitModel2D.calc_est_flats_building', mock_calc_est_flats_building)
    module_mocker.patch.dict('src.RetrofitModel2D.retrofit_packages', mock_retrofit_packages, clear=True)

# Now, we can safely import the class
from src.RetrofitModel2D  import RetrofitModel2D, BuildingCharacteristics, RetrofitConfig, RetrofitEnergy

# --- Pytest Fixtures ---

@pytest.fixture
def base_epistemic_scenario():
    """A simple, default epistemic scenario."""
    return {
        'solid_wall_internal_improvement_factor': 0.1,
        'solid_wall_external_improvement_factor': 0.2,
        'time_scale_bias': 1.0,
        'decile_misclassification_bias': 0.0,
        'regional_multipliers_uncertainty': 1.0,
        'age_band_multipliers_uncertainty': 1.0,
        'cost_scenario': 1.0,
        'flat_fp_mean': 70.0,
        'flat_fp_std': 10.0,
        'flat_eff_mean': 0.8,
        'flat_eff_std': 0.1,
    }

@pytest.fixture
def mock_retrofit_config_instance():
    """Returns an instance of the mocked RetrofitConfig."""
    return MockRetrofitConfig(existing_intervention_probs={'roof_scaling_factor': 0.8})

@pytest.fixture
def model(mock_retrofit_config_instance, base_epistemic_scenario, mocker):
    """
    Provides a default RetrofitModel2D instance for tests.
    Its dependencies (energy_config, cost_estimator) are mocked.
    """
    model_instance = RetrofitModel2D(
        retrofit_config=mock_retrofit_config_instance,
        n_samples=10,
        epistemic_scenario=base_epistemic_scenario
    )
 
    model_instance.cost_estimator.sample_cost_for_package = MagicMock()
    model_instance.energy_config.sample_intervention_energy_savings_monte_carlo = MagicMock()
    
    return model_instance

@pytest.fixture
def sample_building_chars():
    """A sample BuildingCharacteristics object."""
    return MockBuildingCharacteristics(
        floor_count=2,
        gross_external_area=150,
        gross_internal_area=120,
        footprint_circumference=50,
        flat_count=1,
        building_footprint_area=75,
        avg_gas_percentile=3, # Uses decile_risk_scaling[3] = 1.0
        typology='Standard size semi detached'
    )

@pytest.fixture
def sample_row():
    """A sample DataFrame row (as a pd.Series) for row-level tests."""
    return pd.Series({
        'building_id': 123,
        'building_type': 'Standard size semi detached',
        'age_band': '1919-1944',
        'floor_count': 2,
        'footprint_area': 75.0,
        'gross_external_area': 150.0,
        'gross_internal_area': 120.0,
        'footprint_circumference': 50.0,
        'flat_count': 1,
        'avg_gas_percentile': 3,
        'inferred_wall_type': 'solid',
        'inferred_insulation_type': 'external_wall_insulation',
        'wall_insulated': False,
        'existing_loft_insulation': False,
        'existing_floor_insulation': False,
        'existing_window_upgrades': False,
        'total_gas_derived': 20000.0, # Baseline gas
        'total_elec_derived': 3000.0, # Baseline elec
    })

@pytest.fixture
def sample_df(sample_row):
    """A sample DataFrame for the main workflow test."""
    df = pd.DataFrame([sample_row, sample_row.copy()])
    df.loc[1, 'building_id'] = 456
    df.loc[1, 'inferred_insulation_type'] = 'internal_wall_insulation'
    df.loc[1, 'total_gas_derived'] = 15000.0
    return df

@pytest.fixture
def default_col_mapping():
    """Default column mapping."""
    return {
        'building_type': 'building_type',
        'age_band': 'age_band',
        'floor_count': 'floor_count',
        'footprint_area': 'footprint_area',
        'gross_external_area': 'gross_external_area',
        'gross_internal_area': 'gross_internal_area',
        'footprint_circumference': 'footprint_circumference',
        'flat_count': 'flat_count',
        'avg_gas_percentile': 'avg_gas_percentile',
        # --- FIX: ADD THESE MISSING KEYS ---
        'inferred_wall_type': 'inferred_wall_type',
        'inferred_insulation_type': 'inferred_insulation_type',
        'building_footprint_area': 'footprint_area',
    }


# --- Test Cases ---

class TestRetrofitModel2D_Initialization:

    def test_init_success(self, mock_retrofit_config_instance, base_epistemic_scenario):
        """Test successful initialization and creation of dependencies."""
        model = RetrofitModel2D(
            retrofit_config=mock_retrofit_config_instance,
            n_samples=100,
            epistemic_scenario=base_epistemic_scenario
        )
        assert model.n_samples == 100
        assert model.retrofit_config == mock_retrofit_config_instance
        # --- FIX: Change this from assert False ---
        assert isinstance(model.energy_config, MockRetrofitEnergy)
        assert isinstance(model.cost_estimator, MockCostEstimator)

    def test_init_applies_epistemic_factors_to_energy(self, mock_retrofit_config_instance):
        """Check that epistemic factors are passed to RetrofitEnergy on creation."""
        scenario = {'solid_wall_internal_improvement_factor': 0.5,
                    'solid_wall_external_improvement_factor': 0.7}
        
        model = RetrofitModel2D(
            retrofit_config=mock_retrofit_config_instance,
            n_samples=100,
            epistemic_scenario=scenario
        )
        
        assert model.energy_config.solid_wall_internal_improvement_factor == 0.5
        assert model.energy_config.solid_wall_external_improvement_factor == 0.7

    def test_init_updates_existing_energy_config(self, mock_retrofit_config_instance, mocker):
        """Check that epistemic factors are updated on a pre-supplied config."""
        scenario = {'solid_wall_internal_improvement_factor': 0.5,
                    'solid_wall_external_improvement_factor': 0.7}
        
        # Create a pre-existing, mocked config instance
        existing_energy_config = MockRetrofitEnergy(
            solid_wall_internal_improvement_factor=99.0, # Old value
            solid_wall_external_improvement_factor=99.0  # Old value
        )
        
        model = RetrofitModel2D(
            retrofit_config=mock_retrofit_config_instance,
            n_samples=100,
            epistemic_scenario=scenario,
            energy_config=existing_energy_config # Pass it in
        )
        
        assert model.energy_config == existing_energy_config
        assert model.energy_config.solid_wall_internal_improvement_factor == 0.5 # New value
        assert model.energy_config.solid_wall_external_improvement_factor == 0.7 # New value

    def test_init_invalid_samples(self, mock_retrofit_config_instance, base_epistemic_scenario):
        """Test that n_samples < 1 raises a ValueError."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            RetrofitModel2D(
                retrofit_config=mock_retrofit_config_instance,
                n_samples=0,
                epistemic_scenario=base_epistemic_scenario
            )

class TestRetrofitModel2D_Validators:

    def test_validate_region(self, model):
        assert model.validate_region('LN') == 'LN'
        with pytest.raises(ValueError, match="Invalid region 'XX'"):
            model.validate_region('XX')

    def test_validate_statistics(self, model):
        assert model._validate_statistics(None) == ['mean', 'p5', 'p50', 'p95', 'std']
        assert model._validate_statistics(['mean', 'p90']) == ['mean', 'p90']
        invalid = model._validate_statistics(['mean', 'foo', 'bar'])
        assert 'error' in invalid
        assert 'foo' in invalid['error']
        
    def test_get_scenario_interventions(self, model):
        # Note: 'loft_only' is from our mock_retrofit_packages
        interventions = model._get_scenario_interventions('loft_only')
        assert interventions == ['loft_percentile']
        
        error = model._get_scenario_interventions('non_existent')
        assert 'error' in error


class TestRetrofitModel2D_Utilities:

    @pytest.mark.parametrize("stat, expected", [
        ('mean', 2.0),
        ('median', 2.0),
        ('p50', 2.0),
        ('std', np.std([1, 2, 3], ddof=0)),
        ('p90', np.percentile([1, 2, 3], 90)),
    ])
    def test_calculate_single_statistic_basic(self, model, stat, expected):
        samples = np.array([1, 2, 3])
        assert model._calculate_single_statistic(samples, stat) == pytest.approx(expected)

    @pytest.mark.parametrize("stat, expected", [
        ('mean', 1.5),  # (1+2)/2
        ('median', 1.5),# (1+2)/2
        ('p50', 1.5),
        ('std', np.nanstd([1, 2, np.nan], ddof=0)), # 0.5
        ('p90', np.nanpercentile([1, 2, np.nan], 90)), # 1.9
    ])
    def test_calculate_single_statistic_with_nans(self, model, stat, expected):
        samples = np.array([1, 2, np.nan])
        assert model._calculate_single_statistic(samples, stat) == pytest.approx(expected)

    def test_calculate_single_statistic_all_nans(self, model):
        samples = np.array([np.nan, np.nan])
        assert np.isnan(model._calculate_single_statistic(samples, 'mean'))
        assert np.isnan(model._calculate_single_statistic(samples, 'std'))
        assert np.isnan(model._calculate_single_statistic(samples, 'p50'))

    def test_calculate_single_statistic_empty(self, model):
        samples = np.array([])
        with pytest.raises(ValueError, match="empty array"):
            model._calculate_single_statistic(samples, 'mean')
            
    def test_calculate_single_statistic_invalid(self, model):
        samples = np.array([1, 2])
        with pytest.raises(ValueError, match="Unknown statistic"):
            model._calculate_single_statistic(samples, 'foo')


class TestRetrofitModel2D_CoreLogic:

    def test_sample_intervention_cost_monte_carlo(self, model, sample_building_chars, mocker):
        """
        Tests that cost sampling correctly applies combined epistemic and nominal
        multipliers when calling the cost estimator.
        """
        # Arrange
        # Set up a complex scenario to test the multiplier math
        model.age_band_multipliers['1919-1944'] = 2.0  # Nominal age mult
        model.regional_multipliers['LN'] = 1.25       # Nominal region mult
        model.epistemic_scenario = {
            'age_band_multipliers_uncertainty': 0.9, # Epistemic age mult
            'regional_multipliers_uncertainty': 1.1, # Epistemic region mult
            'cost_scenario': 0.8                     # Epistemic cost scenario
        }
        
        expected_final_age_mult = 2.0 * 0.9   # 1.8
        expected_final_regional_mult = 1.25 * 1.1 # 1.375
        
        mock_return_samples = np.array([1000] * 10)
        model.cost_estimator.sample_cost_for_package.return_value = mock_return_samples

        # Act
        samples = model.sample_intervention_cost_monte_carlo(
            intervention=['loft_percentile'],
            cost_col_name='test_cost',
            building_chars=sample_building_chars,
            typology='Standard size semi detached',
            wall_insulation_type='solid',
            age_band='1919-1944',
            region='LN'
        )

        # Assert
        assert np.array_equal(samples, mock_return_samples)
        
        # Check the *call* to the mocked cost estimator
        model.cost_estimator.sample_cost_for_package.assert_called_once_with(
            intervention=['loft_percentile'],
            building_chars=sample_building_chars,
            typology='Standard size semi detached',
            wall_type='solid',
            age_band='1919-1944',
            region='LN',
            cost_col_name='test_cost',
            epist_scenario=0.8,
            regional_multiplier=pytest.approx(expected_final_regional_mult), # 1.375
            age_multiplier=pytest.approx(expected_final_age_mult),         # 1.8
            n_samples=model.n_samples
        )

    def test_calculate_joint_statistics_epistemic_energy(self, model, sample_building_chars, mocker):
        """
        Tests that energy savings percentages are correctly adjusted by
        epistemic factors (time_scale_bias, decile_misclassification_bias).
        """
        # Arrange
        model.n_samples = 2
        model.epistemic_scenario = {
            'time_scale_bias': 1.1,                 # 10% uplift
            'decile_misclassification_bias': -0.05, # 5% reduction
        }
        # This building has percentile 3, so decile_scale = 1.0
        # effective_beta_DEC = -0.05 * 1.0 = -0.05
        
        # Mock costs (simple)
        mocker.patch.object(model, 'sample_intervention_cost_monte_carlo', 
                            return_value=np.full(2, 1000))
        
        # Mock energy (simple percentage)
        # Raw savings = -20%
        raw_gas_perc = np.full(2, -0.2)
        model.energy_config.sample_intervention_energy_savings_monte_carlo.return_value = {
            'gas': raw_gas_perc
        }

        # Expected calculation:
        # final_perc = (raw_perc + effective_beta_DEC) * beta_TS
        # final_perc = (-0.20 + -0.05) * 1.1
        # final_perc = (-0.25) * 1.1 = -0.275
        expected_final_gas_perc = -0.275
        
        # total_gas_derived = 1000 kWh
        # expected_abs_kwh = -0.275 * 1000 = -275
        expected_final_abs_kwh = -275

        # Act
        results = model.calculate_joint_scenario_statistics(
            joint_intervention='loft_percentile',
            building_chars=sample_building_chars,
            wall_insulation_type='solid',
            typology='Standard size semi detached',
            age_band='1919-1944',
            region='LN',
            return_statistics=['mean'],
            scenario_name='test_scen',
            total_gas_derived=1000.0, # Easy math
            total_elec_derived=100.0
        )
        
        # Assert
        assert results['gas_saving_perc_test_scen_mean'] == pytest.approx(expected_final_gas_perc)
        assert results['gas_saving_abs_kwh_test_scen_mean'] == pytest.approx(expected_final_abs_kwh)
        # Cost-per-unit (ratio-of-means)
        # 1000 / -275
        assert results['cost_per_gas_kwh_test_scen_mean'] == pytest.approx(1000 / -275)

    def test_calculate_joint_statistics_hybrid_ratio_logic(self, model, sample_building_chars, mocker):
        """
        **This is the most critical test.**
        It validates the hybrid logic for calculating cost-per-unit statistics
        when some simulations "fail" (i.e., savings are >= 0).
        """
        # Arrange
        model.n_samples = 10
        stats = ['mean', 'std', 'p50', 'p95']
        
        # 1. Mock Costs
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        costs = (np.arange(10) + 1) * 100
        mocker.patch.object(model, 'sample_intervention_cost_monte_carlo', return_value=costs)
        
        # 2. Mock Energy (Absolute kWh)
        # 9 successful savings, 1 "failure" (positive saving/backfire)
        # We set total_gas_derived=1.0 so perc_samples == abs_samples
        abs_kwh = np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, 5.0])
        model.energy_config.sample_intervention_energy_savings_monte_carlo.return_value = {
            'gas': abs_kwh
        }
        
        # 3. Expected "Base" Statistics (simple averages)
        # Mean Cost: 550
        # Mean Abs kWh: (-90 + 5) / 10 = -8.5
        
        # 4. Expected "Ratio" Statistics (THE HYBRID LOGIC)
        # Individual ratios for the 9 successes:
        # [100/-10, 200/-10, ..., 900/-10] = [-10, -20, -30, -40, -50, -60, -70, -80, -90]
        # The 10th sample (1000 / 5.0) is a failure.
        
        success_ratios = np.array([-10, -20, -30, -40, -50, -60, -70, -80, -90])
        
        # --- Ratio-of-Means ---
        # expected_mean_ratio = mean(costs) / mean(abs_kwh)
        expected_mean_ratio = 550 / -8.5 
        
        # # --- Standard Deviation (uses np.nan array) ---
        # # Internal array: [-10, -20, ..., -90, np.nan]
        # expected_std_ratio = np.nanstd(success_ratios) # Approx 25.98
        
        # # --- Percentiles (uses np.inf array) ---
        # # Internal array: [-10, -20, ..., -90, np.inf]
        # # p50 (median) of this array:
        # expected_p50_ratio = np.nanmedian(success_ratios) # -50
        # # p95 of this array:
        # expected_p95_ratio = np.nanpercentile(np.append(success_ratios, np.inf), 95) # -15.0
        expected_std_ratio = np.nanstd(success_ratios)
        expected_p50_ratio = np.nanmedian(success_ratios) # -50
        expected_p95_ratio = np.nanpercentile(success_ratios, 95) # -14.0

        # 1 failure out of 10 samples
        expected_failure_rate = 0.1
        # Act
        results = model.calculate_joint_scenario_statistics(
            joint_intervention='loft_percentile',
            building_chars=sample_building_chars,
            wall_insulation_type='solid',
            typology='Standard size semi detached',
            age_band='1919-1944',
            region='LN',
            return_statistics=stats,
            scenario_name='hybrid_test',
            total_gas_derived=1.0, # Makes perc == abs
            total_elec_derived=0.0
        )
        
 
        
        # --- FIX: Update all assertions ---
        assert results['cost_per_gas_kwh_hybrid_test_mean'] == pytest.approx(expected_mean_ratio)
        assert results['cost_per_gas_kwh_hybrid_test_std'] == pytest.approx(expected_std_ratio)
        assert results['cost_per_gas_kwh_hybrid_test_p50'] == pytest.approx(expected_p50_ratio)
        assert results['cost_per_gas_kwh_hybrid_test_p95'] == pytest.approx(expected_p95_ratio)
        
        # --- NEW: Add assertion for the failure rate ---
        assert results['gas_saving_failure_rate_hybrid_test'] == pytest.approx(expected_failure_rate)
        
    def test_calculate_joint_statistics_all_failures(self, model, sample_building_chars, mocker):
        """Tests that ratios are np.nan if *all* savings are >= 0."""
        # Arrange
        model.n_samples = 2
        stats = ['mean', 'std', 'p50']
        mocker.patch.object(model, 'sample_intervention_cost_monte_carlo', 
                            return_value=np.full(2, 1000))
        # All savings are "failures" (zero)
        model.energy_config.sample_intervention_energy_savings_monte_carlo.return_value = {
            'gas': np.zeros(2)
        }
        
        # Act
        results = model.calculate_joint_scenario_statistics(
            joint_intervention='loft_percentile',
            building_chars=sample_building_chars,
            wall_insulation_type='solid',
            typology='Standard size semi detached',
            age_band='1919-1944',
            region='LN',
            return_statistics=stats,
            scenario_name='fail_test',
            total_gas_derived=1000.0,
            total_elec_derived=0.0
        )

        # Assert
        assert np.isnan(results['cost_per_gas_kwh_fail_test_mean'])
        assert np.isnan(results['cost_per_gas_kwh_fail_test_std'])
        
        # --- FIX: Median of all-nan array is nan ---
        assert np.isnan(results['cost_per_gas_kwh_fail_test_p50'])
        
        # --- NEW: Add assertion for 100% failure rate ---
        assert results['gas_saving_failure_rate_fail_test'] == 1.0


    def test_calculate_row_statistics(self, model, sample_row, default_col_mapping, mocker):
        """
        Tests the row-level orchestrator.
        Ensures it resolves 'WALL_INSULATION' correctly and calls the
        main statistics function with the right parameters.
        """
        # Arrange
        # This row has 'inferred_insulation_type': 'external_wall_insulation'
        # So 'WALL_INSULATION' should resolve to 'solid_wall_external_percentile'
        expected_intervention = 'solid_wall_external_percentile'
        scenario_interventions = ['WALL_INSULATION'] # From 'wall_only' scenario
        
        mock_stats_return = {'cost_wall_only_mean': 1234.5}
        mocker.patch.object(model, 'calculate_joint_scenario_statistics', 
                            return_value=mock_stats_return)

        # Act
        result_series = model.calculate_row_statistics(
            row=sample_row,
            col_mapping=default_col_mapping,
            scenario_interventions=scenario_interventions,
            scenario_name='wall_only',
            region='LN',
            return_statistics=['mean']
        )
        
        # Assert
        assert isinstance(result_series, pd.Series)
        assert result_series['cost_wall_only_mean'] == 1234.5
        
        # Check that the stats function was called with the *resolved* intervention
        args, kwargs = model.calculate_joint_scenario_statistics.call_args
        
        assert kwargs['joint_intervention'] == expected_intervention
        assert kwargs['wall_insulation_type'] == 'external_wall_insulation'
        assert kwargs['scenario_name'] == 'wall_only'
        assert kwargs['total_gas_derived'] == 20000.0
        assert kwargs['total_elec_derived'] == 3000.0
        assert isinstance(kwargs['building_chars'], MockBuildingCharacteristics)
        assert kwargs['building_chars'].avg_gas_percentile == 3


class TestRetrofitModel2D_FullWorkflow:

    def test_calculate_building_costs_df_updated_success(self, model, sample_df, default_col_mapping, mocker):
        """
        Tests the main public entry point.
        Mocks the row-level function and checks that the final DataFrame
        is assembled and renamed correctly.
        """
        # Arrange
        # We need to mock the *row-level* function's return value
        # It will be called twice (once for each row)
        row1_return = pd.Series({
            'cost_wall_only_mean': 1000,
            'gas_saving_abs_kwh_wall_only_mean': -100,
            'cost_per_gas_kwh_wall_only_mean': -10,
        })
        row2_return = pd.Series({
            'cost_wall_only_mean': 2000,
            'gas_saving_abs_kwh_wall_only_mean': -200,
            'cost_per_gas_kwh_wall_only_mean': -10,
        })
        
        mocker.patch.object(model, 'calculate_row_statistics', 
                            side_effect=[row1_return, row2_return])
        
        # --- FIX: Update this mock ---
        # It must return the column names that your other mocks create.
        # These names match the keys in your 'row1_return' Series.
        expected_cost_cols = ['cost_wall_only_mean', 'cost_per_gas_kwh_wall_only_mean']
        expected_energy_cols = ['gas_saving_abs_kwh_wall_only_mean']
        
        mocker.patch.object(model, '_get_cols_scenario_intervention', 
                            return_value=(expected_cost_cols, expected_energy_cols))
        
        # Expected column names after final renaming:
        # {scenario}_{col_name} -> 'wall_only_cost_wall_only_mean'
        expected_cost_col = 'wall_only_cost_wall_only_mean'
        expected_gas_col = 'wall_only_gas_saving_abs_kwh_wall_only_mean'
        expected_ratio_col = 'wall_only_cost_per_gas_kwh_wall_only_mean'

        # Act
        result_df = model.calculate_building_costs_df_updated(
            df=sample_df,
            region='LN',
            scenario='wall_only',
            col_mapping=default_col_mapping,
            return_statistics=['mean']
        )
        
        # Assert
        # Check that the row function was called twice
        assert model.calculate_row_statistics.call_count == 2
        
        # Check that the original columns are present
        assert 'building_id' in result_df.columns
        
        # Check that the new, *renamed* columns are present
        assert expected_cost_col in result_df.columns
        assert expected_gas_col in result_df.columns
        assert expected_ratio_col in result_df.columns
        
        # Check the values from the mocked returns
        assert result_df.loc[0, expected_cost_col] == 1000
        assert result_df.loc[1, expected_cost_col] == 2000
        assert result_df.loc[0, expected_gas_col] == -100
        assert result_df.loc[1, expected_gas_col] == -200
        assert result_df.loc[0, expected_ratio_col] == -10
        
    def test_calculate_building_costs_df_validation_error(self, model, sample_df, default_col_mapping):
        """Tests that top-level validation failures return an error dict."""
        
        # Test missing DataFrame
        result = model.calculate_building_costs_df_updated(
            df=None, region='LN', scenario='wall_only', col_mapping=default_col_mapping
        )
        assert 'error' in result and 'DataFrame is None' in result['error']
        
        # Test invalid scenario
        result = model.calculate_building_costs_df_updated(
            df=sample_df, region='LN', scenario='bad_scenario', col_mapping=default_col_mapping
        )
        assert 'error' in result and 'not found' in result['error']

        # Test missing columns
        bad_df = sample_df.drop(columns=['building_type'])
        result = model.calculate_building_costs_df_updated(
            df=bad_df, region='LN', scenario='wall_only', col_mapping=default_col_mapping
        )
        assert 'error' in result and 'building_type' in result['error'] 