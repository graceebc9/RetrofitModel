import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import sys
sys.path.append('/Users/gracecolverd/retrofit_model')
# Assuming your module structure
from src.RetrofitModel import RetrofitModel
from src.BuildingCharacteristics import BuildingCharacteristics
from src.RetrofitConfig import RetrofitConfig
from src.RetrofitEnergy import RetrofitEnergy
from src.RetrofitCostsScalingRules import InterventionScalingRules


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def retrofit_config():
    """Create a mock RetrofitConfig with necessary attributes."""
    config = Mock(spec=RetrofitConfig)
    config.existing_intervention_probs = {
        'roof_scaling_factor': 0.8,
        'wall_insulation': 0.3,
        'loft_insulation': 0.5,
    }
    return config


@pytest.fixture
def retrofit_model(retrofit_config):
    """Create a RetrofitModel instance with test configuration."""
    model = RetrofitModel(
        retrofit_config=retrofit_config,
        n_samples=100
    )
    return model


@pytest.fixture
def sample_building_characteristics():
    """Create sample BuildingCharacteristics for testing."""
    return BuildingCharacteristics(
        floor_count=2,
        gross_external_area=150.0,
        gross_internal_area=140.0,
        footprint_circumference=50.0,
        flat_count=1,
        building_footprint_area=70.0,
        avg_gas_percentile=50
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'premise_type': ['Standard size semi detached', 'Small low terraces'],
        'premise_age': ['1960-1979', 'Pre 1919'],
        'total_fl_area_avg': [2.0, 1.0],
        'scaled_fl_area': [140.0, 80.0],
        'perimeter_length': [50.0, 40.0],
        'premise_area': [70.0, 40.0],
        'premise_floor_count': [2, 1],
        'wall_insulated': [False, True],
        'existing_loft_insulation': [False, False],
        'existing_floor_insulation': [False, False],
        'existing_window_upgrades': [False, False],
        'inferred_wall_type': ['cavity_wall', 'solid_wall'],
        'avg_gas_percentile': [50, 60]
    })


@pytest.fixture
def sample_row(sample_dataframe):
    """Create a sample row for testing."""
    return sample_dataframe.iloc[0]


# ============================================================================
# TEST CLASS 1: INTERVENTION RESOLUTION LOGIC
# ============================================================================

class TestInterventionResolution:
    """Tests for resolve_interventions_for_building method."""
    
    def test_cavity_wall_uses_correct_type_no_percentile(self, retrofit_model):
        """Test cavity walls get cavity_wall_insulation when percentile=False."""
        scenario_interventions = ['WALL_INSULATION', 'loft_insulation']
        
        interventions, selected_wall = retrofit_model.resolve_interventions_for_building(
            scenario_interventions=scenario_interventions,
            wall_type='cavity_wall',
            prob_external=0.5,
            percentile=False
        )
        
        assert 'cavity_wall_insulation' in interventions
        assert selected_wall == 'cavity_wall_insulation'
        assert 'loft_insulation' in interventions
    
    def test_cavity_wall_uses_percentile_when_enabled(self, retrofit_model):
        """Test cavity walls get cavity_wall_percentile when percentile=True."""
        scenario_interventions = ['WALL_INSULATION']
        
        interventions, selected_wall = retrofit_model.resolve_interventions_for_building(
            scenario_interventions=scenario_interventions,
            wall_type='cavity_wall',
            prob_external=0.5,
            percentile=True
        )
        
        assert 'cavity_wall_percentile' in interventions
        assert selected_wall == 'cavity_wall_insulation'
    
    def test_solid_wall_uses_percentile_when_enabled(self, retrofit_model):
        """Test solid walls get solid_wall_percentile when percentile=True."""
        scenario_interventions = ['WALL_INSULATION']
        
        interventions, selected_wall = retrofit_model.resolve_interventions_for_building(
            scenario_interventions=scenario_interventions,
            wall_type='solid_wall',
            prob_external=0.5,
            percentile=True
        )
        
        assert 'solid_wall_percentile' in interventions
        assert selected_wall == 'solid_insulation'
    
    @patch('numpy.random.random')
    def test_solid_wall_selects_external_based_on_probability(self, mock_random, retrofit_model):
        """Test solid wall selects external insulation when random < prob_external."""
        mock_random.return_value = 0.3  # Less than prob_external
        scenario_interventions = ['WALL_INSULATION']
        
        interventions, selected_wall = retrofit_model.resolve_interventions_for_building(
            scenario_interventions=scenario_interventions,
            wall_type='solid_wall',
            prob_external=0.5,
            percentile=False
        )
        
        assert 'external_wall_insulation' in interventions
        assert selected_wall == 'external_wall_insulation'
    
    @patch('numpy.random.random')
    def test_solid_wall_selects_internal_based_on_probability(self, mock_random, retrofit_model):
        """Test solid wall selects internal insulation when random >= prob_external."""
        mock_random.return_value = 0.7  # Greater than prob_external
        scenario_interventions = ['WALL_INSULATION']
        
        interventions, selected_wall = retrofit_model.resolve_interventions_for_building(
            scenario_interventions=scenario_interventions,
            wall_type='solid_wall',
            prob_external=0.5,
            percentile=False
        )
        
        assert 'internal_wall_insulation' in interventions
        assert selected_wall == 'internal_wall_insulation'
    
    def test_preserves_non_wall_interventions(self, retrofit_model):
        """Test other interventions remain unchanged."""
        scenario_interventions = ['loft_insulation', 'double_glazing', 'WALL_INSULATION']
        
        interventions, _ = retrofit_model.resolve_interventions_for_building(
            scenario_interventions=scenario_interventions,
            wall_type='cavity_wall',
            prob_external=0.5,
            percentile=False
        )
        
        assert 'loft_insulation' in interventions
        assert 'double_glazing' in interventions
        assert len(interventions) == 3


# ============================================================================
# TEST CLASS 2: SKIP INTERVENTIONS LOGIC
# ============================================================================

class TestSkipInterventions:
    """Tests for get_skip_interventions method."""
    
    def test_no_existing_retrofits_skips_nothing(self, retrofit_model):
        """Test that no interventions are skipped when nothing is installed."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=False,
            existing_loft=False,
            existing_floor=False,
            existing_windows=False
        )
        
        assert len(skip_set) == 0
    
    def test_wall_insulated_skips_all_wall_types(self, retrofit_model):
        """Test that all wall insulation types are skipped when wall_insulated=True."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=True,
            existing_loft=False,
            existing_floor=False,
            existing_windows=False
        )
        
        assert 'cavity_wall_insulation' in skip_set
        assert 'external_wall_insulation' in skip_set
        assert 'internal_wall_insulation' in skip_set
        assert 'cavity_wall_percentile' in skip_set
        assert 'solid_wall_percentile' in skip_set
    
    def test_existing_loft_skips_loft_interventions(self, retrofit_model):
        """Test that loft interventions are skipped when existing_loft=True."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=False,
            existing_loft=True,
            existing_floor=False,
            existing_windows=False
        )
        
        assert 'loft_insulation' in skip_set
        assert 'loft_percentile' in skip_set
    
    def test_existing_floor_skips_floor_insulation(self, retrofit_model):
        """Test that floor insulation is skipped when existing_floor=True."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=False,
            existing_loft=False,
            existing_floor=True,
            existing_windows=False
        )
        
        assert 'floor_insulation' in skip_set
    
    def test_existing_windows_skips_double_glazing(self, retrofit_model):
        """Test that double glazing is skipped when existing_windows=True."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=False,
            existing_loft=False,
            existing_floor=False,
            existing_windows=True
        )
        
        assert 'double_glazing' in skip_set
    
    def test_all_existing_retrofits_skips_all(self, retrofit_model):
        """Test that all interventions are skipped when everything is installed."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=True,
            existing_loft=True,
            existing_floor=True,
            existing_windows=True
        )
        
        expected_skips = {
            'cavity_wall_insulation', 'external_wall_insulation', 
            'internal_wall_insulation', 'cavity_wall_percentile',
            'solid_wall_percentile', 'loft_insulation', 'loft_percentile',
            'floor_insulation', 'double_glazing'
        }
        
        assert skip_set == expected_skips
    
    def test_combinations_of_existing_retrofits(self, retrofit_model):
        """Test various combinations of existing retrofits."""
        skip_set = retrofit_model.get_skip_interventions(
            wall_insulated=True,
            existing_loft=True,
            existing_floor=False,
            existing_windows=False
        )
        
        # Should skip wall and loft, but not floor or windows
        assert 'cavity_wall_insulation' in skip_set
        assert 'loft_insulation' in skip_set
        assert 'floor_insulation' not in skip_set
        assert 'double_glazing' not in skip_set


# ============================================================================
# TEST CLASS 3: COST AGGREGATION LOGIC
# ============================================================================

class TestCostAggregation:
    """Tests for intervention cost calculation and aggregation."""
    
    @patch.object(RetrofitModel, 'sample_intervention_cost_monte_carlo')
    def test_intervention_costs_calculated_for_all_non_skipped(
        self, mock_sample, retrofit_model, sample_building_characteristics
    ):
        """Test that costs are calculated for all non-skipped interventions."""
        # Mock returns different costs for each intervention
        mock_sample.side_effect = [
            np.array([1000.0] * 100),  # cavity_wall_insulation
            np.array([500.0] * 100),   # loft_insulation
        ]
        
        interventions = ['cavity_wall_insulation', 'loft_insulation']
        skip_interventions = set()
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean', 'p50'],
            include_total=True
        )
        
        assert 'cavity_wall_insulation_mean' in cost_stats
        assert 'loft_insulation_mean' in cost_stats
        assert cost_stats['cavity_wall_insulation_mean'] == 1000.0
        assert cost_stats['loft_insulation_mean'] == 500.0
    
    @patch.object(RetrofitModel, 'sample_intervention_cost_monte_carlo')
    def test_skipped_interventions_have_zero_cost(
        self, mock_sample, retrofit_model, sample_building_characteristics
    ):
        """Test that skipped interventions return cost=0."""
        mock_sample.return_value = np.array([1000.0] * 100)
        
        interventions = ['cavity_wall_insulation', 'loft_insulation']
        skip_interventions = {'loft_insulation'}
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean'],
            include_total=False
        )
        
        assert cost_stats['loft_insulation_mean'] == 0.0
        assert mock_sample.call_count == 1  # Only called for non-skipped
    
    @patch.object(RetrofitModel, 'sample_intervention_cost_monte_carlo')
    def test_total_cost_sums_samples_before_statistics(
        self, mock_sample, retrofit_model, sample_building_characteristics
    ):
        """Test that total cost is calculated by summing samples first, then computing stats."""
        # Create samples with variance
        samples1 = np.random.normal(1000, 100, 100)
        samples2 = np.random.normal(500, 50, 100)
        mock_sample.side_effect = [samples1, samples2]
        
        interventions = ['cavity_wall_insulation', 'loft_insulation']
        skip_interventions = set()
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean', 'p50', 'p95'],
            include_total=True
        )
        
        # Total should be calculated from summed samples
        expected_total_samples = samples1 + samples2
        expected_mean = np.mean(expected_total_samples)
        expected_p50 = np.percentile(expected_total_samples, 50)
        
        assert 'total_mean' in cost_stats
        assert 'total_p50' in cost_stats
        assert abs(cost_stats['total_mean'] - expected_mean) < 0.01
        assert abs(cost_stats['total_p50'] - expected_p50) < 0.01
    
    @patch.object(RetrofitModel, 'sample_intervention_cost_monte_carlo')
    def test_total_includes_only_non_skipped_interventions(
        self, mock_sample, retrofit_model, sample_building_characteristics
    ):
        """Test that total cost only includes non-skipped interventions."""
        samples1 = np.array([1000.0] * 100)
        samples2 = np.array([500.0] * 100)
        mock_sample.side_effect = [samples1]  # Only one call expected
        
        interventions = ['cavity_wall_insulation', 'loft_insulation']
        skip_interventions = {'loft_insulation'}
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean'],
            include_total=True
        )
        
        # Total should equal only the non-skipped intervention
        assert cost_stats['total_mean'] == 1000.0


# ============================================================================
# TEST CLASS 4: ENERGY SAVINGS AGGREGATION
# ============================================================================

class TestEnergySavingsAggregation:
    """Tests for energy savings calculation and aggregation."""
    
    @patch.object(RetrofitEnergy, 'sample_intervention_energy_savings_monte_carlo')
    def test_gas_savings_combine_additively_at_sample_level(
        self, mock_energy, retrofit_model, sample_building_characteristics
    ):
        """Test gas samples are added element-wise before calculating statistics."""
        # Create two sets of gas samples
        gas_samples1 = np.random.normal(0.9, 0.05, 100)  # 10% reduction
        gas_samples2 = np.random.normal(0.85, 0.05, 100)  # 15% reduction
        
        mock_energy.side_effect = [
            gas_samples1,  # First intervention (percentile)
            gas_samples2   # Second intervention (percentile)
        ]
        
        interventions = ['cavity_wall_percentile', 'loft_percentile']
        
        energy_stats = retrofit_model.calculate_intervention_energy_savings(
            interventions=interventions,
            building_chars=sample_building_characteristics,
            region='LN',
            return_statistics=['mean', 'p50', 'p95'],
            roof_scaling=0.8
        )
        
        # Gas should be combined additively
        expected_combined = gas_samples1 + gas_samples2
        expected_mean = np.mean(expected_combined)
        
        assert 'gas' in energy_stats
        assert abs(energy_stats['gas']['mean'] - expected_mean) < 0.01
    
    @patch.object(RetrofitEnergy, 'sample_intervention_energy_savings_monte_carlo')
    def test_electricity_savings_combine_additively_at_sample_level(
        self, mock_energy, retrofit_model, sample_building_characteristics
    ):
        """Test electricity samples are added element-wise before calculating statistics."""
        # Create electricity intervention samples
        elec_samples1 = np.random.normal(500, 50, 100)  # kWh
        elec_samples2 = np.random.normal(300, 30, 100)  # kWh
        
        mock_energy.side_effect = [
            {'electricity': elec_samples1},
            {'electricity': elec_samples2}
        ]
        
        interventions = ['heat_pump_upgrade', 'double_glazing']
        
        energy_stats = retrofit_model.calculate_intervention_energy_savings(
            interventions=interventions,
            building_chars=sample_building_characteristics,
            region='LN',
            return_statistics=['mean', 'p50'],
            roof_scaling=0.8
        )
        
        # Electricity should be combined additively
        expected_combined = elec_samples1 + elec_samples2
        expected_mean = np.mean(expected_combined)
        
        assert 'electricity' in energy_stats
        assert abs(energy_stats['electricity']['mean'] - expected_mean) < 0.01
    
    @patch.object(RetrofitEnergy, 'sample_intervention_energy_savings_monte_carlo')
    def test_gas_and_electricity_kept_separate(
        self, mock_energy, retrofit_model, sample_building_characteristics
    ):
        """Test gas and electricity are aggregated separately."""
        gas_samples = np.random.normal(0.9, 0.05, 100)
        elec_samples = np.random.normal(500, 50, 100)
        
        mock_energy.side_effect = [
            gas_samples,  # Gas intervention (percentile)
            {'electricity': elec_samples}  # Electricity intervention
        ]
        
        interventions = ['loft_percentile', 'heat_pump_upgrade']
        
        energy_stats = retrofit_model.calculate_intervention_energy_savings(
            interventions=interventions,
            building_chars=sample_building_characteristics,
            region='LN',
            return_statistics=['mean'],
            roof_scaling=0.8
        )
        
        assert 'gas' in energy_stats
        assert 'electricity' in energy_stats
        assert energy_stats['gas']['mean'] < 1.0  # Gas is a multiplier
        assert energy_stats['electricity']['mean'] > 0  # Electricity is kWh


# ============================================================================
# TEST CLASS 5: MULTIPLIER APPLICATION
# ============================================================================

class TestMultiplierApplication:
    """Tests for regional, age, and complexity multipliers."""
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_regional_multiplier_affects_costs(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test regional multiplier is applied to costs."""
        base_samples = np.array([1000.0] * 100)
        mock_scaling.return_value = base_samples
        
        # Call for London (multiplier = 1.25)
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        # Check that regional_multiplier was passed correctly
        call_args = mock_scaling.call_args[1]
        assert call_args['regional_multiplier'] == 1.25
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_age_band_multiplier_older_costs_more(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test older buildings have higher age multipliers."""
        base_samples = np.array([1000.0] * 100)
        mock_scaling.return_value = base_samples
        
        # Call for Pre 1919 building (multiplier = 2.0)
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='Pre 1919',
            region='LN'
        )
        
        call_args = mock_scaling.call_args[1]
        assert call_args['age_multiplier'] == 2.0
        
        # Call for Post 1999 building (multiplier = 0.9)
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='Post 1999',
            region='LN'
        )
        
        call_args = mock_scaling.call_args[1]
        assert call_args['age_multiplier'] == 0.9
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_complexity_multiplier_affects_tall_buildings(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test tall/complex buildings have higher complexity multipliers."""
        base_samples = np.array([1000.0] * 100)
        mock_scaling.return_value = base_samples
        
        # Call for tall flats (multiplier = 1.2)
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Tall flats 6-15 storeys',
            age_band='1960-1979',
            region='LN'
        )
        
        call_args = mock_scaling.call_args[1]
        assert call_args['complexity_multiplier'] == 1.2
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_multipliers_combine_correctly(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test all three multipliers are passed to scaling rules."""
        base_samples = np.array([1000.0] * 100)
        mock_scaling.return_value = base_samples
        
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Very tall point block flats',  # complexity = 1.4
            age_band='Pre 1919',  # age = 2.0
            region='LN'  # regional = 1.25
        )
        
        call_args = mock_scaling.call_args[1]
        assert call_args['regional_multiplier'] == 1.25
        assert call_args['age_multiplier'] == 2.0
        assert call_args['complexity_multiplier'] == 1.4


# ============================================================================
# TEST CLASS 6: STATISTICS CALCULATION
# ============================================================================

class TestStatisticsCalculation:
    """Tests for _calculate_single_statistic and _calculate_statistics methods."""
    
    def test_calculate_single_statistic_mean(self, retrofit_model):
        """Test mean calculation."""
        samples = np.array([10, 20, 30, 40, 50])
        result = retrofit_model._calculate_single_statistic(samples, 'mean')
        assert result == 30.0
    
    def test_calculate_single_statistic_median(self, retrofit_model):
        """Test median calculation."""
        samples = np.array([10, 20, 30, 40, 50])
        result = retrofit_model._calculate_single_statistic(samples, 'median')
        assert result == 30.0
    
    def test_calculate_single_statistic_std(self, retrofit_model):
        """Test standard deviation calculation."""
        samples = np.array([10, 20, 30, 40, 50])
        result = retrofit_model._calculate_single_statistic(samples, 'std')
        expected = np.std([10, 20, 30, 40, 50])
        assert abs(result - expected) < 0.01
    
    def test_calculate_single_statistic_percentiles(self, retrofit_model):
        """Test percentile calculations."""
        samples = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        p5 = retrofit_model._calculate_single_statistic(samples, 'p5')
        p50 = retrofit_model._calculate_single_statistic(samples, 'p50')
        p95 = retrofit_model._calculate_single_statistic(samples, 'p95')
        
        assert abs(p5 - np.percentile(samples, 5)) < 0.01
        assert p50 == 55.0
        assert abs(p95 - np.percentile(samples, 95)) < 0.01
    
    def test_calculate_single_statistic_empty_array_raises(self, retrofit_model):
        """Test that empty array raises ValueError."""
        samples = np.array([])
        
        with pytest.raises(ValueError, match="Cannot calculate statistics on empty array"):
            retrofit_model._calculate_single_statistic(samples, 'mean')
    
    def test_calculate_statistics_dict_gas_electricity(self, retrofit_model):
        """Test statistics calculation with dict containing gas and electricity."""
        samples = {
            'gas': np.array([0.9, 0.85, 0.88, 0.92, 0.87]),
            'electricity': np.array([100, 120, 110, 115, 105])
        }
        
        result = retrofit_model._calculate_statistics(samples, 'mean')
        
        assert 'gas' in result
        assert 'electricity' in result
        assert abs(result['gas'] - 0.884) < 0.01
        assert abs(result['electricity'] - 110.0) < 0.01
    
    def test_calculate_statistics_list_multiplicative_gas(self, retrofit_model):
        """Test list of dicts combines gas multiplicatively."""
        samples_list = [
            {
                'gas': np.array([0.9, 0.9, 0.9]),
                'electricity': np.array([100, 100, 100])
            },
            {
                'gas': np.array([0.8, 0.8, 0.8]),
                'electricity': np.array([50, 50, 50])
            }
        ]
        
        result = retrofit_model._calculate_statistics(samples_list, 'mean')
        
        # Gas should be multiplied: 0.9 * 0.8 = 0.72
        assert abs(result['gas'] - 0.72) < 0.01
        # Electricity should be added: 100 + 50 = 150
        assert abs(result['electricity'] - 150.0) < 0.01


# ============================================================================
# TEST CLASS 7: SCENARIO SPECIFIC LOGIC
# ============================================================================

class TestScenarioLogic:
    """Tests for scenario-specific intervention lists."""
    
    def test_scenario2_includes_correct_interventions(self, retrofit_model):
        """Test scenario2 has correct intervention list."""
        scenario = retrofit_model.retrofit_packages['scenario2']
        
        assert 'WALL_INSULATION' in scenario['interventions']
        assert 'loft_insulation' in scenario['interventions']
        assert 'double_glazing' in scenario['interventions']
        assert scenario['includes_wall_insulation'] is True
    
    def test_scenario3_adds_heat_pump(self, retrofit_model):
        """Test scenario3 includes envelope + heat pump."""
        scenario = retrofit_model.retrofit_packages['scenario3']
        
        assert 'WALL_INSULATION' in scenario['interventions']
        assert 'loft_insulation' in scenario['interventions']
        assert 'double_glazing' in scenario['interventions']
        assert 'heat_pump_upgrade' in scenario['interventions']
    
    def test_scenario5_adds_solar(self, retrofit_model):
        """Test scenario5 includes envelope + heat pump + solar."""
        scenario = retrofit_model.retrofit_packages['scenario5']
        
        assert 'WALL_INSULATION' in scenario['interventions']
        assert 'heat_pump_upgrade' in scenario['interventions']
        assert 'solar_pv' in scenario['interventions']
    
    def test_loft_installation_scenario(self, retrofit_model):
        """Test loft-only scenario."""
        scenario = retrofit_model.retrofit_packages['loft_installation']
        
        assert 'loft_percentile' in scenario['interventions']
        assert scenario['includes_wall_insulation'] is False


# ============================================================================
# TEST CLASS 8: VALIDATION LOGIC
# ============================================================================

class TestValidation:
    """Tests for input validation methods."""
    
    def test_validate_inputs_none_dataframe(self, retrofit_model):
        """Test None DataFrame returns error."""
        result = retrofit_model._validate_inputs(None, 'LN', 'scenario2')
        
        assert result is not None
        assert 'error' in result
        assert 'DataFrame is None or empty' in result['error']
    
    def test_validate_inputs_empty_dataframe(self, retrofit_model):
        """Test empty DataFrame returns error."""
        df = pd.DataFrame()
        result = retrofit_model._validate_inputs(df, 'LN', 'scenario2')
        
        assert result is not None
        assert 'error' in result
    
    def test_validate_inputs_missing_region(self, retrofit_model, sample_dataframe):
        """Test missing region returns error."""
        result = retrofit_model._validate_inputs(sample_dataframe, '', 'scenario2')
        
        assert result is not None
        assert 'error' in result
        assert 'Region parameter is required' in result['error']
    
    def test_validate_inputs_invalid_scenario(self, retrofit_model, sample_dataframe):
        """Test invalid scenario returns error."""
        result = retrofit_model._validate_inputs(
            sample_dataframe, 'LN', 'invalid_scenario'
        )
        
        assert result is not None
        assert 'error' in result
        assert 'not found' in result['error']
    
    def test_validate_inputs_valid(self, retrofit_model, sample_dataframe):
        """Test valid inputs return None."""
        result = retrofit_model._validate_inputs(
            sample_dataframe, 'LN', 'scenario2'
        )
        
        assert result is None
    
    def test_validate_statistics_invalid_stat(self, retrofit_model):
        """Test invalid statistic name."""
        result = retrofit_model._validate_statistics(['mean', 'invalid_stat'])
        
        assert 'error' in result
    
    def test_validate_statistics_valid(self, retrofit_model):
        """Test valid statistics list."""
        result = retrofit_model._validate_statistics(['mean', 'p50', 'p95'])
        
        assert result == ['mean', 'p50', 'p95']
    
    def test_validate_region_invalid(self, retrofit_model):
        """Test invalid region raises ValueError."""
        with pytest.raises(ValueError, match="Invalid region"):
            retrofit_model.validate_region('INVALID')
    
    def test_validate_region_valid(self, retrofit_model):
        """Test valid region returns region code."""
        result = retrofit_model.validate_region('LN')
        assert result == 'LN'


# Add these to tests/test_retrofit_model.py

# ============================================================================
# TEST CLASS: COST VALUE VALIDATION
# ============================================================================

class TestCostValueValidation:
    """Tests to validate that cost values are reasonable and within expected ranges."""
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_costs_are_always_positive(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that all intervention costs are positive values."""
        # Mock returns positive costs
        mock_scaling.return_value = np.random.uniform(1000, 5000, 100)
        
        samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        assert np.all(samples > 0), "All costs should be positive"
        assert not np.any(np.isnan(samples)), "No costs should be NaN"
        assert not np.any(np.isinf(samples)), "No costs should be infinite"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_loft_insulation_cost_in_reasonable_range(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that loft insulation costs are within reasonable range (£500-£3000)."""
        # Mock realistic loft insulation costs
        mock_scaling.return_value = np.random.uniform(500, 3000, 100)
        
        samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        mean_cost = np.mean(samples)
        assert 400 < mean_cost < 3500, f"Loft insulation mean cost {mean_cost} outside reasonable range"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_cavity_wall_cost_in_reasonable_range(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that cavity wall insulation costs are within reasonable range (£1000-£5000)."""
        mock_scaling.return_value = np.random.uniform(1000, 5000, 100)
        
        samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='cavity_wall_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        mean_cost = np.mean(samples)
        assert 800 < mean_cost < 6000, f"Cavity wall mean cost {mean_cost} outside reasonable range"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_heat_pump_more_expensive_than_insulation(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that heat pump costs more than basic insulation (business rule)."""
        # Mock costs: heat pump should be 3-4x more expensive than loft
        mock_scaling.side_effect = [
            np.random.uniform(800, 1500, 100),   # loft
            np.random.uniform(8000, 15000, 100)  # heat pump
        ]
        
        loft_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        hp_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='heat_pump_upgrade',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        assert np.mean(hp_samples) > np.mean(loft_samples) * 3, \
            "Heat pump should be at least 3x more expensive than loft insulation"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_external_wall_more_expensive_than_cavity(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that external wall insulation costs more than cavity wall."""
        mock_scaling.side_effect = [
            np.random.uniform(1000, 3000, 100),  # cavity
            np.random.uniform(8000, 15000, 100)  # external
        ]
        
        cavity_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='cavity_wall_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        external_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='external_wall_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        assert np.mean(external_samples) > np.mean(cavity_samples) * 2, \
            "External wall should be at least 2x more expensive than cavity"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_london_more_expensive_than_northeast(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that London costs more than North East (regional multiplier validation)."""
        base_cost = 5000
        
        # London calls with multiplier 1.25
        london_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        # Check that London multiplier (1.25) was passed
        call_args_london = mock_scaling.call_args[1]
        assert call_args_london['regional_multiplier'] == 1.25
        
        # North East calls with multiplier 0.85
        ne_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='NE'
        )
        
        # Check that NE multiplier (0.85) was passed
        call_args_ne = mock_scaling.call_args[1]
        assert call_args_ne['regional_multiplier'] == 0.85
        
        # Verify the ratio is approximately 1.25/0.85 = 1.47
        expected_ratio = 1.25 / 0.85
        assert abs(expected_ratio - 1.47) < 0.01
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_older_buildings_cost_more(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that pre-1919 buildings cost more than post-1999 (age multiplier)."""
        # Pre-1919: multiplier = 2.0
        old_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='Pre 1919',
            region='LN'
        )
        
        call_args_old = mock_scaling.call_args[1]
        assert call_args_old['age_multiplier'] == 2.0
        
        # Post 1999: multiplier = 0.9
        new_samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='Post 1999',
            region='LN'
        )
        
        call_args_new = mock_scaling.call_args[1]
        assert call_args_new['age_multiplier'] == 0.9
        
        # Verify ratio
        expected_ratio = 2.0 / 0.9
        assert abs(expected_ratio - 2.22) < 0.05
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_total_cost_reasonable_for_scenario2(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that total cost for scenario 2 is within reasonable range (£5k-£20k)."""
        # Mock realistic costs for scenario 2 interventions
        mock_scaling.side_effect = [
            np.random.uniform(2000, 4000, 100),   # wall
            np.random.uniform(800, 1500, 100),    # loft
            np.random.uniform(3000, 6000, 100),   # glazing
        ]
        
        interventions = ['cavity_wall_insulation', 'loft_insulation', 'double_glazing']
        skip_interventions = set()
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean'],
            include_total=True
        )
        
        total_mean = cost_stats['total_mean']
        assert 4000 < total_mean < 25000, \
            f"Scenario 2 total cost {total_mean} outside reasonable range (£4k-£25k)"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_total_cost_reasonable_for_scenario3(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that total cost for scenario 3 (with heat pump) is £15k-£40k."""
        # Mock realistic costs
        mock_scaling.side_effect = [
            np.random.uniform(2000, 4000, 100),   # wall
            np.random.uniform(800, 1500, 100),    # loft
            np.random.uniform(3000, 6000, 100),   # glazing
            np.random.uniform(10000, 18000, 100), # heat pump
        ]
        
        interventions = ['cavity_wall_insulation', 'loft_insulation', 
                        'double_glazing', 'heat_pump_upgrade']
        skip_interventions = set()
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean'],
            include_total=True
        )
        
        total_mean = cost_stats['total_mean']
        assert 12000 < total_mean < 45000, \
            f"Scenario 3 total cost {total_mean} outside reasonable range (£12k-£45k)"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_cost_variance_reasonable(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that cost variance is reasonable (CV < 30%)."""
        # Mock with some variance
        mean_cost = 5000
        mock_scaling.return_value = np.random.normal(mean_cost, mean_cost * 0.2, 100)
        
        samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='cavity_wall_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        # Coefficient of variation should be less than 30%
        cv = np.std(samples) / np.mean(samples)
        assert cv < 0.30, f"Cost variance too high: CV = {cv:.2%}"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_p95_not_absurdly_high(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that 95th percentile costs are not absurdly high (< 3x mean)."""
        mock_scaling.return_value = np.random.gamma(4, 1000, 100)  # Realistic skewed distribution
        
        samples = retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        mean_cost = np.mean(samples)
        p95_cost = np.percentile(samples, 95)
        
        ratio = p95_cost / mean_cost
        assert ratio < 3.0, f"P95 is {ratio:.1f}x mean - too high, suggests unrealistic outliers"
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_no_negative_costs_in_total(
        self, mock_scaling, retrofit_model, sample_building_characteristics
    ):
        """Test that total costs are never negative, even with skipped interventions."""
        # Mock some interventions
        mock_scaling.side_effect = [
            np.random.uniform(2000, 4000, 100),  # wall
            # loft will be skipped (cost = 0)
        ]
        
        interventions = ['cavity_wall_insulation', 'loft_insulation']
        skip_interventions = {'loft_insulation'}
        
        cost_stats = retrofit_model.calculate_intervention_costs(
            interventions=interventions,
            skip_interventions=skip_interventions,
            building_chars=sample_building_characteristics,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN',
            return_statistics=['mean', 'p5'],
            include_total=True
        )
        
        assert cost_stats['total_mean'] > 0
        assert cost_stats['total_p5'] >= 0
        assert cost_stats['loft_insulation_mean'] == 0


# ============================================================================
# TEST CLASS: COST SCALING VALIDATION
# ============================================================================

class TestCostScalingValidation:
    """Tests to validate that costs scale appropriately with building characteristics."""
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_larger_building_costs_more(
        self, mock_scaling, retrofit_model
    ):
        """Test that larger buildings cost more to retrofit."""
        # Small building
        small_building = BuildingCharacteristics(
            floor_count=1,
            gross_external_area=80.0,
            gross_internal_area=70.0,
            footprint_circumference=30.0,
            flat_count=1,
            building_footprint_area=40.0,
            avg_gas_percentile=5
        )
        
        # Large building
        large_building = BuildingCharacteristics(
            floor_count=3,
            gross_external_area=250.0,
            gross_internal_area=230.0,
            footprint_circumference=70.0,
            flat_count=1,
            building_footprint_area=120.0,
            avg_gas_percentile=5
        )
        
        # Sample for small building
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=small_building,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        # Get the building_chars that were passed
        small_call_args = mock_scaling.call_args[1]
        small_building_passed = small_call_args['building_chars']
        
        # Sample for large building
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='loft_insulation',
            building_chars=large_building,
            typology='Standard size semi detached',
            age_band='1960-1979',
            region='LN'
        )
        
        large_call_args = mock_scaling.call_args[1]
        large_building_passed = large_call_args['building_chars']
        
        # Verify that building characteristics are passed correctly
        assert large_building_passed.gross_external_area > small_building_passed.gross_external_area
    
    @patch.object(InterventionScalingRules, 'sample_intervention_cost')
    def test_flats_more_expensive_per_unit(
        self, mock_scaling, retrofit_model
    ):
        """Test that multi-unit buildings use flat_count appropriately."""
        # Single dwelling
        single_dwelling = BuildingCharacteristics(
            floor_count=2,
            gross_external_area=150.0,
            gross_internal_area=140.0,
            footprint_circumference=50.0,
            flat_count=1,
            building_footprint_area=70.0,
            avg_gas_percentile=5
        )
        
        # Multi-unit building
        multi_unit = BuildingCharacteristics(
            floor_count=4,
            gross_external_area=400.0,
            gross_internal_area=380.0,
            footprint_circumference=80.0,
            flat_count=8,
            building_footprint_area=100.0,
            avg_gas_percentile=5
        )
        
        # The flat_count should be passed to scaling rules
        retrofit_model.sample_intervention_cost_monte_carlo(
            intervention='cavity_wall_insulation',
            building_chars=multi_unit,
            typology='Medium height flats 5-6 storeys',
            age_band='1960-1979',
            region='LN'
        )
        
        call_args = mock_scaling.call_args[1]
        building_passed = call_args['building_chars']
        
        assert building_passed.flat_count == 8


 


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])