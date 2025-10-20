import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import sys
sys.path.append('/Users/gracecolverd/retrofit_model')
# Assuming your module structure
# Assuming your module structure
from src.RetrofitEnergy import RetrofitEnergy
from src.BuildingCharacteristics import BuildingCharacteristics


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def retrofit_energy():
    """Create a RetrofitEnergy instance for testing."""
    return RetrofitEnergy()


@pytest.fixture
def sample_building_characteristics():
    """Create sample BuildingCharacteristics for testing."""
    # Create a mock that properly handles solar_roof_area_estimate
    building = Mock(spec=BuildingCharacteristics)
    building.floor_count = 2
    building.gross_external_area = 150.0
    building.gross_internal_area = 140.0
    building.footprint_circumference = 50.0
    building.flat_count = 1
    building.building_footprint_area = 70.0
    building.avg_gas_percentile = 5
    
    # Mock the solar_roof_area_estimate method to return a reasonable value
    building.solar_roof_area_estimate = Mock(return_value=20.0)
    
    return building


# ============================================================================
# TEST CLASS 1: SOLAR PV CALCULATIONS
# ============================================================================

class TestSolarPVCalculations:
    """Tests for solar PV Monte Carlo calculations."""
    
    def test_solar_pv_returns_correct_structure(self, retrofit_energy):
        """Test that solar PV calculation returns expected dict structure."""
        result = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='LN',
            scaled_roof_size=15.0,
            n_samples=100
        )
        
        assert isinstance(result, dict)
        assert 'annual_generation_kwh' in result
        assert 'adjusted_kwh_per_m' in result
        assert 'solar_regional_multiplier' in result
        assert 'matched_roof_size' in result
        assert len(result['annual_generation_kwh']) == 100
    
    def test_solar_pv_regional_multiplier_high_sun(self, retrofit_energy):
        """Test that high sun regions (SW, SE, EE) get 1.2x multiplier."""
        result = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='SW',
            scaled_roof_size=20.0,
            n_samples=100
        )
        
        assert result['solar_regional_multiplier'][0] == 1.2
    
    def test_solar_pv_regional_multiplier_low_sun(self, retrofit_energy):
        """Test that low sun regions (NE, NW, YH, WA) get 0.8x multiplier."""
        result = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='NE',
            scaled_roof_size=20.0,
            n_samples=100
        )
        
        assert result['solar_regional_multiplier'][0] == 0.8
    
    def test_solar_pv_regional_multiplier_mid_sun(self, retrofit_energy):
        """Test that mid sun regions (LN, EM, WM) get 1.0x multiplier."""
        result = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='LN',
            scaled_roof_size=20.0,
            n_samples=100
        )
        
        assert result['regional_multiplier'][0] == 1.0
    
    def test_solar_pv_roof_size_snapping(self, retrofit_energy):
        """Test that roof sizes snap to nearest available system size."""
        # Test various roof sizes
        test_cases = [
            (3.0, 5),    # Close to 5
            (7.5, 10),   # Close to 10
            (18.0, 15),  # Close to 15
            (23.0, 20),  # Close to 20
            (28.0, 30),  # Close to 30
        ]
        
        for input_size, expected_size in test_cases:
            result = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
                region='LN',
                scaled_roof_size=input_size,
                n_samples=10
            )
            assert result['matched_roof_size'][0] == expected_size
    
    def test_solar_pv_generation_scales_with_roof_size(self, retrofit_energy):
        """Test that generation scales with roof size."""
        small_roof = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='LN',
            scaled_roof_size=10.0,
            n_samples=1000
        )
        
        large_roof = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='LN',
            scaled_roof_size=30.0,
            n_samples=1000
        )
        
        # Larger roof should generate more on average
        assert np.mean(large_roof['annual_generation_kwh']) > np.mean(small_roof['annual_generation_kwh'])
    
    def test_solar_pv_samples_within_triangular_range(self, retrofit_energy):
        """Test that kwh_per_m samples are within min-max range."""
        result = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='LN',
            scaled_roof_size=20.0,
            n_samples=1000
        )
        
        # Should be between 100 and 300 kwh/m (from config)
        assert np.all(result['adjusted_kwh_per_m'] >= 100)
        assert np.all(result['adjusted_kwh_per_m'] <= 300)
    
    def test_solar_pv_regional_multiplier_affects_generation(self, retrofit_energy):
        """Test that regional multiplier affects generation appropriately."""
        # Compare high sun vs low sun regions with same roof
        high_sun = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='SW',  # 1.2x
            scaled_roof_size=20.0,
            n_samples=1000
        )
        
        low_sun = retrofit_energy.calculate_solar_pv_impact_monte_carlo(
            region='NE',  # 0.8x
            scaled_roof_size=20.0,
            n_samples=1000
        )
        
        # High sun region should generate ~1.5x more (1.2/0.8)
        ratio = np.mean(high_sun['annual_generation_kwh']) / np.mean(low_sun['annual_generation_kwh'])
        assert 1.4 < ratio < 1.6  # Allow some variance


# ============================================================================
# TEST CLASS 2: PERCENTILE SAVINGS
# ============================================================================

class TestPercentileSavings:
    """Tests for percentile-based energy savings."""
    
    def test_percentile_savings_returns_array(self, retrofit_energy):
        """Test that percentile savings returns numpy array."""
        result = retrofit_energy.sample_percentile_savings(
            intervention='loft_percentile',
            avg_gas_percentile=5,
            n_samples=100
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
    
    def test_percentile_savings_loft_higher_percentile_more_savings(self, retrofit_energy):
        """Test that higher gas percentiles have larger savings for loft."""
        # Low percentile (less consumption, less savings potential)
        low_savings = retrofit_energy.sample_percentile_savings(
            intervention='loft_percentile',
            avg_gas_percentile=1,
            n_samples=1000
        )
        
        # High percentile (more consumption, more savings potential)
        high_savings = retrofit_energy.sample_percentile_savings(
            intervention='loft_percentile',
            avg_gas_percentile=8,
            n_samples=1000
        )
        
        # Higher percentile should have larger absolute savings (more negative)
        assert abs(np.mean(high_savings)) > abs(np.mean(low_savings))
    
    def test_percentile_savings_cavity_wall_higher_percentile_more_savings(self, retrofit_energy):
        """Test that higher gas percentiles have larger savings for cavity walls."""
        low_savings = retrofit_energy.sample_percentile_savings(
            intervention='cavity_wall_percentile',
            avg_gas_percentile=2,
            n_samples=1000
        )
        
        high_savings = retrofit_energy.sample_percentile_savings(
            intervention='cavity_wall_percentile',
            avg_gas_percentile=9,
            n_samples=1000
        )
        
        assert abs(np.mean(high_savings)) > abs(np.mean(low_savings))
    
    def test_percentile_savings_solid_wall_saves_more_than_cavity(self, retrofit_energy):
        """Test that solid wall insulation saves ~10% more than cavity (business rule)."""
        cavity_savings = retrofit_energy.sample_percentile_savings(
            intervention='cavity_wall_percentile',
            avg_gas_percentile=5,
            n_samples=1000
        )
        
        solid_savings = retrofit_energy.sample_percentile_savings(
            intervention='solid_wall_percentile',
            avg_gas_percentile=5,
            n_samples=1000
        )
        
        # Solid wall should save more (more negative mean)
        assert abs(np.mean(solid_savings)) > abs(np.mean(cavity_savings))
        
        # Should be approximately 10% more
        ratio = abs(np.mean(solid_savings)) / abs(np.mean(cavity_savings))
        assert 1.08 < ratio < 1.12  # 10% +/- 2%
    
    def test_percentile_savings_invalid_percentile_raises(self, retrofit_energy):
        """Test that invalid percentile raises KeyError."""
        with pytest.raises(KeyError):
            retrofit_energy.sample_percentile_savings(
                intervention='loft_percentile',
                avg_gas_percentile=99,  # Invalid, should be 0-9
                n_samples=100
            )
    
    def test_percentile_savings_non_numeric_percentile_raises(self, retrofit_energy):
        """Test that non-numeric percentile raises ValueError."""
        with pytest.raises(ValueError, match="Percentile must be numeric"):
            retrofit_energy.sample_percentile_savings(
                intervention='loft_percentile',
                avg_gas_percentile='invalid',
                n_samples=100
            )
    
    def test_percentile_savings_invalid_intervention_raises(self, retrofit_energy):
        """Test that invalid intervention raises KeyError."""
        with pytest.raises(KeyError, match="No data for intervention"):
            retrofit_energy.sample_percentile_savings(
                intervention='invalid_intervention',
                avg_gas_percentile=5,
                n_samples=100
            )
    
    def test_percentile_savings_samples_follow_normal_distribution(self, retrofit_energy):
        """Test that samples approximately follow normal distribution."""
        samples = retrofit_energy.sample_percentile_savings(
            intervention='loft_percentile',
            avg_gas_percentile=5,
            n_samples=10000
        )
        
        # Get expected mean and sd from config
        expected_mean = retrofit_energy.energysaving_uncertainty_parameters['loft_percentile']['gas'][5]['mean']
        expected_sd = retrofit_energy.energysaving_uncertainty_parameters['loft_percentile']['gas'][5]['sd']
        
        # Check that sample statistics are close to expected
        assert abs(np.mean(samples) - expected_mean) < 0.01
        assert abs(np.std(samples) - expected_sd) < 0.01


# ============================================================================
# TEST CLASS 3: DISTRIBUTION SAMPLING
# ============================================================================

class TestDistributionSampling:
    """Tests for _sample_from_distribution helper method."""
    
    def test_triangular_distribution_returns_correct_size(self, retrofit_energy):
        """Test that triangular sampling returns correct number of samples."""
        dist_params = {'min': 0.05, 'mode': 0.10, 'max': 0.15}
        random_state = np.random.RandomState(42)
        
        samples = retrofit_energy._sample_from_distribution(
            dist_params=dist_params,
            dist_type='triangular',
            n_samples=100,
            random_state=random_state
        )
        
        assert len(samples) == 100
    
    def test_triangular_distribution_within_bounds(self, retrofit_energy):
        """Test that triangular samples are within min-max bounds."""
        dist_params = {'min': 0.05, 'mode': 0.10, 'max': 0.15}
        random_state = np.random.RandomState(42)
        
        samples = retrofit_energy._sample_from_distribution(
            dist_params=dist_params,
            dist_type='triangular',
            n_samples=1000,
            random_state=random_state
        )
        
        assert np.all(samples >= 0.05)
        assert np.all(samples <= 0.15)
    
    def test_triangular_distribution_mode_most_likely(self, retrofit_energy):
        """Test that mode value is most frequent in triangular distribution."""
        dist_params = {'min': 0.05, 'mode': 0.10, 'max': 0.15}
        random_state = np.random.RandomState(42)
        
        samples = retrofit_energy._sample_from_distribution(
            dist_params=dist_params,
            dist_type='triangular',
            n_samples=10000,
            random_state=random_state
        )
        
        # Mean should be close to mode for triangular
        # (actually (min+mode+max)/3, but should be closer to mode than extremes)
        mean_sample = np.mean(samples)
        assert abs(mean_sample - 0.10) < 0.02
    
    def test_uniform_distribution_returns_correct_size(self, retrofit_energy):
        """Test that uniform sampling returns correct number of samples."""
        dist_params = {'min': 0.05, 'max': 0.15}
        random_state = np.random.RandomState(42)
        
        samples = retrofit_energy._sample_from_distribution(
            dist_params=dist_params,
            dist_type='uniform',
            n_samples=100,
            random_state=random_state
        )
        
        assert len(samples) == 100
    
    def test_uniform_distribution_within_bounds(self, retrofit_energy):
        """Test that uniform samples are within min-max bounds."""
        dist_params = {'min': 0.05, 'max': 0.15}
        random_state = np.random.RandomState(42)
        
        samples = retrofit_energy._sample_from_distribution(
            dist_params=dist_params,
            dist_type='uniform',
            n_samples=1000,
            random_state=random_state
        )
        
        assert np.all(samples >= 0.05)
        assert np.all(samples <= 0.15)
    
    def test_invalid_distribution_type_raises(self, retrofit_energy):
        """Test that invalid distribution type raises ValueError."""
        dist_params = {'min': 0.05, 'max': 0.15}
        random_state = np.random.RandomState(42)
        
        with pytest.raises(ValueError, match="Unknown distribution type"):
            retrofit_energy._sample_from_distribution(
                dist_params=dist_params,
                dist_type='invalid',
                n_samples=100,
                random_state=random_state
            )


# ============================================================================
# TEST CLASS 4: ENERGY SAVINGS SAMPLING
# ============================================================================

class TestEnergySavingsSampling:
    """Tests for _sample_energy_savings method."""
    
    def test_energy_savings_returns_dict_with_gas_and_electricity(self, retrofit_energy):
        """Test that energy savings returns dict with gas and electricity keys."""
        result = retrofit_energy._sample_energy_savings(
            intervention='loft_insulation',
            n_samples=100
        )
        
        assert isinstance(result, dict)
        assert 'gas' in result
        assert 'electricity' in result
        assert len(result['gas']) == 100
        assert len(result['electricity']) == 100
    
    def test_energy_savings_loft_insulation_gas_only(self, retrofit_energy):
        """Test that loft insulation saves gas but not electricity."""
        result = retrofit_energy._sample_energy_savings(
            intervention='loft_insulation',
            n_samples=100
        )
        
        # Gas should have positive savings
        assert np.mean(result['gas']) > 0
        
        # Electricity should be all zeros
        assert np.all(result['electricity'] == 0)
    
    def test_energy_savings_cavity_wall_gas_only(self, retrofit_energy):
        """Test that cavity wall insulation saves gas but not electricity."""
        result = retrofit_energy._sample_energy_savings(
            intervention='cavity_wall_insulation',
            n_samples=100
        )
        
        assert np.mean(result['gas']) > 0
        assert np.all(result['electricity'] == 0)
    
    def test_energy_savings_heat_pump_affects_both_fuels(self, retrofit_energy):
        """Test that heat pump reduces gas and increases electricity."""
        result = retrofit_energy._sample_energy_savings(
            intervention='heat_pump_upgrade',
            n_samples=100
        )
        
        # Gas should have high savings (0.90-0.98)
        assert np.mean(result['gas']) > 0.85
        
        # Electricity should be negative (increase in consumption)
        assert np.mean(result['electricity']) < 0
        assert np.mean(result['electricity']) > -0.65
    
    def test_energy_savings_deep_retrofit_both_fuels(self, retrofit_energy):
        """Test that deep retrofit affects both gas and electricity."""
        result = retrofit_energy._sample_energy_savings(
            intervention='deep_retrofit_estimate',
            n_samples=1000
        )
        
        # Gas should have large savings (0.55-0.99)
        assert 0.5 < np.mean(result['gas']) < 1.0
        
        # Electricity should have positive savings (0.20-0.70)
        assert 0.15 < np.mean(result['electricity']) < 0.75
    
    def test_energy_savings_solar_pv_raises_error(self, retrofit_energy):
        """Test that solar_pv raises error in this method."""
        with pytest.raises(ValueError, match="should not be used for solar_pv"):
            retrofit_energy._sample_energy_savings(
                intervention='solar_pv',
                n_samples=100
            )
    
    def test_energy_savings_invalid_intervention_raises(self, retrofit_energy):
        """Test that invalid intervention raises ValueError."""
        with pytest.raises(ValueError, match="No uncertainty parameters found"):
            retrofit_energy._sample_energy_savings(
                intervention='invalid_intervention',
                n_samples=100
            )
    
    def test_energy_savings_samples_within_configured_range(self, retrofit_energy):
        """Test that samples fall within configured min-max ranges."""
        result = retrofit_energy._sample_energy_savings(
            intervention='cavity_wall_insulation',
            n_samples=1000
        )
        
        # From config: gas min=0.073, max=0.155
        assert np.all(result['gas'] >= 0.073)
        assert np.all(result['gas'] <= 0.155)


# ============================================================================
# TEST CLASS 5: MAIN SAMPLING INTERFACE
# ============================================================================

class TestSampleInterventionEnergySavings:
    """Tests for main sample_intervention_energy_savings_monte_carlo method."""
    
    def test_routes_to_solar_pv_calculation(self, retrofit_energy, sample_building_characteristics):
        """Test that solar_pv intervention routes to solar calculation."""
        result = retrofit_energy.sample_intervention_energy_savings_monte_carlo(
            intervention='solar_pv',
            building_chars=sample_building_characteristics,
            region='LN',
            n_samples=100,
            roof_scaling=0.8
        )
        
        # Should return solar-specific dict structure
        assert 'annual_generation_kwh' in result
        assert 'matched_roof_size' in result
    
    def test_routes_to_percentile_calculation(self, retrofit_energy, sample_building_characteristics):
        """Test that percentile interventions route to percentile sampling."""
        result = retrofit_energy.sample_intervention_energy_savings_monte_carlo(
            intervention='loft_percentile',
            building_chars=sample_building_characteristics,
            region='LN',
            n_samples=100,
            roof_scaling=0.8
        )
        
        # Should return simple array for percentile
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
    
    def test_routes_to_standard_calculation(self, retrofit_energy, sample_building_characteristics):
        """Test that standard interventions route to standard sampling."""
        result = retrofit_energy.sample_intervention_energy_savings_monte_carlo(
            intervention='loft_insulation',
            building_chars=sample_building_characteristics,
            region='LN',
            n_samples=100,
            roof_scaling=0.8
        )
        
        # Should return dict with gas and electricity
        assert isinstance(result, dict)
        assert 'gas' in result
        assert 'electricity' in result
    
    def test_invalid_intervention_raises(self, retrofit_energy, sample_building_characteristics):
        """Test that invalid intervention raises ValueError."""
        with pytest.raises(ValueError, match="not found in energy savings parameters"):
            retrofit_energy.sample_intervention_energy_savings_monte_carlo(
                intervention='invalid_intervention',
                building_chars=sample_building_characteristics,
                region='LN',
                n_samples=100,
                roof_scaling=0.8
            )
    
    def test_uses_building_characteristics(self, retrofit_energy):
        """Test that building characteristics are used in calculations."""
        # Create mocked buildings with different gas percentiles
        building_low = Mock(spec=BuildingCharacteristics)
        building_low.avg_gas_percentile = 1  # Low consumption
        building_low.solar_roof_area_estimate = Mock(return_value=20.0)
        
        building_high = Mock(spec=BuildingCharacteristics)
        building_high.avg_gas_percentile = 8  # High consumption
        building_high.solar_roof_area_estimate = Mock(return_value=20.0)
        
        # Sample for percentile intervention
        result_low = retrofit_energy.sample_intervention_energy_savings_monte_carlo(
            intervention='loft_percentile',
            building_chars=building_low,
            region='LN',
            n_samples=1000,
            roof_scaling=0.8
        )
        
        result_high = retrofit_energy.sample_intervention_energy_savings_monte_carlo(
            intervention='loft_percentile',
            building_chars=building_high,
            region='LN',
            n_samples=1000,
            roof_scaling=0.8
        )
        
        # High consumption building should have different (larger) savings
        assert abs(np.mean(result_high)) > abs(np.mean(result_low))


# ============================================================================
# TEST CLASS 6: BUSINESS RULES
# ============================================================================

class TestBusinessRules:
    """Tests for specific business rules and constraints."""
    
    def test_cavity_wall_saves_more_than_loft(self, retrofit_energy):
        """Test that cavity wall saves more gas than loft (business expectation)."""
        loft_result = retrofit_energy._sample_energy_savings(
            intervention='loft_insulation',
            n_samples=1000
        )
        
        cavity_result = retrofit_energy._sample_energy_savings(
            intervention='cavity_wall_insulation',
            n_samples=1000
        )
        
        # Cavity wall should save more on average
        assert np.mean(cavity_result['gas']) > np.mean(loft_result['gas'])
    
    def test_solid_wall_saves_more_than_cavity_wall(self, retrofit_energy):
        """Test that solid wall saves more than cavity wall."""
        cavity_result = retrofit_energy._sample_energy_savings(
            intervention='cavity_wall_insulation',
            n_samples=1000
        )
        
        internal_result = retrofit_energy._sample_energy_savings(
            intervention='internal_wall_insulation',
            n_samples=1000
        )
        
        external_result = retrofit_energy._sample_energy_savings(
            intervention='external_wall_insulation',
            n_samples=1000
        )
        
        # Solid wall (both types) should save more than cavity
        assert np.mean(internal_result['gas']) > np.mean(cavity_result['gas'])
        assert np.mean(external_result['gas']) > np.mean(cavity_result['gas'])
    
    def test_deep_retrofit_has_highest_gas_savings(self, retrofit_energy):
        """Test that deep retrofit has highest gas savings percentage."""
        interventions = [
            'loft_insulation',
            'cavity_wall_insulation',
            'floor_insulation',
            'double_glazing',
            'deep_retrofit_estimate'
        ]
        
        savings = {}
        for intervention in interventions:
            result = retrofit_energy._sample_energy_savings(
                intervention=intervention,
                n_samples=1000
            )
            savings[intervention] = np.mean(result['gas'])
        
        # Deep retrofit should have highest savings
        assert savings['deep_retrofit_estimate'] == max(savings.values())
    
    def test_heat_pump_gas_savings_very_high(self, retrofit_energy):
        """Test that heat pump provides 90%+ gas savings."""
        result = retrofit_energy._sample_energy_savings(
            intervention='heat_pump_upgrade',
            n_samples=1000
        )
        
        # Heat pump should save 90-98% of gas
        assert np.mean(result['gas']) > 0.88
        assert np.mean(result['gas']) < 0.99
    
    def test_heat_pump_increases_electricity_40_60_percent(self, retrofit_energy):
        """Test that heat pump increases electricity by 40-60%."""
        result = retrofit_energy._sample_energy_savings(
            intervention='heat_pump_upgrade',
            n_samples=1000
        )
        
        # Electricity increase should be 40-60% (negative values)
        assert -0.65 < np.mean(result['electricity']) < -0.35


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])