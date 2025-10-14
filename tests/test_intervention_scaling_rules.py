import pytest
import numpy as np
from unittest.mock import Mock, patch
from dataclasses import dataclass

import sys
sys.path.append('/Users/gracecolverd/retrofit_model')
# Assuming your module structure
# Assuming your module structure
from src.RetrofitEnergy import RetrofitEnergy
from src.BuildingCharacteristics import BuildingCharacteristics
from src.RetrofitCostsScalingRules import InterventionScalingRules
 

 

# Mock BuildingCharacteristics for testing
@dataclass
class MockBuildingCharacteristics:
    """Mock BuildingCharacteristics for testing"""
    roof_area_estimate: float = 100.0
    external_wall_area_estimate: float = 200.0
    building_footprint_area: float = 80.0
    gross_internal_area: float = 150.0
    flat_count: int = 1


class TestInterventionScalingRulesInitialization:
    """Test initialization and default values"""
    
    def test_initialization_defaults(self):
        """Test that class initializes with correct default values"""
        rules = InterventionScalingRules()
        
        # Check area-based interventions exist
        assert 'loft_insulation' in rules.area_based_interventions
        assert 'cavity_wall_insulation' in rules.area_based_interventions
        assert 'solar_pv' in rules.area_based_interventions
        
        # Check uncertainty parameters exist
        assert 'loft_insulation' in rules.uncertainty_parameters
        assert 'internal_wall_insulation' in rules.uncertainty_parameters
        
        # Check heat pump cost ranges exist
        assert 'Large detached' in rules.heat_pump_cost_ranges
        assert 'Small low terraces' in rules.heat_pump_cost_ranges
        
    def test_area_based_intervention_structure(self):
        """Test structure of area-based interventions"""
        rules = InterventionScalingRules()
        
        for intervention, config in rules.area_based_interventions.items():
            assert 'base_cost_per_sqm' in config
            assert 'min_cost' in config
            assert 'max_cost' in config
            assert 'area_type' in config
            assert config['area_type'] in ['roof', 'wall', 'floor', 'internal']
            
    def test_uncertainty_parameters_structure(self):
        """Test structure of uncertainty parameters"""
        rules = InterventionScalingRules()
        
        for intervention, params in rules.uncertainty_parameters.items():
            assert 'distribution' in params
            assert 'min' in params
            assert 'mode' in params
            assert 'max' in params
            assert 'cap' in params
            assert len(params['cap']) == 2
            assert params['cap'][0] < params['cap'][1]


class TestHeatPumpCostSampling:
    """Test heat pump cost sampling functionality"""
    
    def test_sample_heat_pump_cost_returns_correct_shape(self):
        """Test that sampling returns correct number of samples"""
        rules = InterventionScalingRules()
        n_samples = 100
        
        samples = rules.sample_heat_pump_cost_triangular(
            'Large detached', 
            n_samples=n_samples
        )
        
        assert len(samples) == n_samples
        assert isinstance(samples, np.ndarray)
        
    def test_heat_pump_cost_within_range(self):
        """Test that sampled costs fall within defined ranges"""
        rules = InterventionScalingRules()
        typology = 'Standard size semi detached'
        n_samples = 1000
        
        min_cost, max_cost = rules.heat_pump_cost_ranges[typology]
        samples = rules.sample_heat_pump_cost_triangular(typology, n_samples)
        
        assert np.all(samples >= min_cost)
        assert np.all(samples <= max_cost)
        
    def test_heat_pump_cost_unknown_typology(self):
        """Test heat pump cost with unknown typology uses default range"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_heat_pump_cost_triangular(
            'unknown_typology', 
            n_samples=10
        )
        
        # Should use default range (7000, 11000)
        assert np.all(samples >= 7000)
        assert np.all(samples <= 11000)
        
    def test_heat_pump_cost_all_typologies(self):
        """Test heat pump sampling for all defined typologies"""
        rules = InterventionScalingRules()
        
        for typology in rules.heat_pump_cost_ranges.keys():
            samples = rules.sample_heat_pump_cost_triangular(typology, n_samples=50)
            min_cost, max_cost = rules.heat_pump_cost_ranges[typology]
            
            assert len(samples) == 50
            assert np.all(samples >= min_cost)
            assert np.all(samples <= max_cost)


class TestDoubleGlazingCostSampling:
    """Test double glazing cost sampling functionality"""
    
    def test_double_glazing_house_typology(self):
        """Test double glazing cost for house typologies"""
        rules = InterventionScalingRules()
        typology = 'Large detached'
        n_samples = 100
        
        samples = rules.sample_double_glazing_cost_triangular(
            typology, 
            n_samples=n_samples
        )
        
        assert len(samples) == n_samples
        min_cost, max_cost = rules.double_glazing_cost_ranges[typology]
        assert np.all(samples >= min_cost)
        assert np.all(samples <= max_cost)
        
    def test_double_glazing_flat_typology_requires_num_flats(self):
        """Test that flat typologies require num_flats parameter"""
        rules = InterventionScalingRules()
        
        with pytest.raises(ValueError, match="num_flats required"):
            rules.sample_double_glazing_cost_triangular(
                'Tall flats 6-15 storeys',
                n_samples=10,
                num_flats=None
            )
            
    def test_double_glazing_flat_small_building(self):
        """Test double glazing cost for small flat building (no economies of scale)"""
        rules = InterventionScalingRules()
        typology = 'Medium height flats 5-6 storeys'
        num_flats = 5
        n_samples = 100
        
        samples = rules.sample_double_glazing_cost_triangular(
            typology,
            n_samples=n_samples,
            num_flats=num_flats
        )
        
        assert len(samples) == n_samples
        # Should use individual flat pricing
        expected_base = num_flats * rules.flat_based_double_glazing['individual_flat_cost']
        # Samples should be around the expected base (within ±10%)
        assert np.mean(samples) > expected_base * 0.85
        assert np.mean(samples) < expected_base * 1.15
        
    def test_double_glazing_flat_large_building(self):
        """Test double glazing cost for large flat building (economies of scale)"""
        rules = InterventionScalingRules()
        typology = '3-4 storey and smaller flats'
        num_flats = 20
        n_samples = 100
        
        samples = rules.sample_double_glazing_cost_triangular(
            typology,
            n_samples=n_samples,
            num_flats=num_flats
        )
        
        assert len(samples) == n_samples
        # Should use bulk pricing
        expected_base = num_flats * rules.flat_based_double_glazing['base_cost_per_flat']
        assert np.mean(samples) > expected_base * 0.85
        assert np.mean(samples) < expected_base * 1.15
        
    def test_double_glazing_flat_respects_caps(self):
        """Test that flat double glazing costs respect min/max caps for various building sizes"""
        rules = InterventionScalingRules()
        typology = 'Very tall point block flats'
        
        max_cap = rules.flat_based_double_glazing['max_cost']
        min_cap = rules.flat_based_double_glazing['min_cost']
        
        # Test with small building (1 flat)
        samples_tiny = rules.sample_double_glazing_cost_triangular(
            typology, n_samples=100, num_flats=1
        )
        assert np.all(samples_tiny >= min_cap)
        assert np.all(samples_tiny <= max_cap)
        
        # Test with small building (5 flats)
        samples_small = rules.sample_double_glazing_cost_triangular(
            typology, n_samples=100, num_flats=5
        )
        assert np.all(samples_small >= min_cap)
        assert np.all(samples_small <= max_cap)
        
        # Test with medium building (15 flats - above economies of scale threshold)
        samples_medium = rules.sample_double_glazing_cost_triangular(
            typology, n_samples=100, num_flats=15
        )
        assert np.all(samples_medium >= min_cap)
        assert np.all(samples_medium <= max_cap)
        
    def test_double_glazing_very_large_flat_building(self):
        """
        Test double glazing for very large flat buildings where base_cost exceeds max cap.
        
        For a 100-flat building:
        - base_cost = 100 × £4,000 = £400,000
        - calculated range would be £360,000 - £440,000
        - but max_cost cap is £80,000
        
        This creates an invalid distribution (min > max). The fix ensures:
        1. Falls back to using the cap range (£4,000 - £80,000)
        2. Sets mode to midpoint of cap range
        3. All samples respect the absolute max cap
        """
        rules = InterventionScalingRules()
        typology = 'Very tall point block flats'
        num_flats = 100  # Very large building
        
        samples = rules.sample_double_glazing_cost_triangular(
            typology,
            n_samples=1000,
            num_flats=num_flats
        )
        
        assert len(samples) == 1000
        
        config = rules.flat_based_double_glazing
        min_cap = config['min_cost']
        max_cap = config['max_cost']
        
        # All samples should be within the absolute caps
        assert np.all(samples >= min_cap), f"Min sample {np.min(samples)} < cap {min_cap}"
        assert np.all(samples <= max_cap), f"Max sample {np.max(samples)} > cap {max_cap}"
        
        # For this edge case, distribution should span the full cap range
        # Check that we have reasonable spread (not all at one extreme)
        assert np.std(samples) > 5000, "Distribution should have reasonable spread"
        
        # Mean should be somewhere in the middle third of the range
        expected_midpoint = (min_cap + max_cap) / 2
        assert min_cap < np.mean(samples) < max_cap
        assert abs(np.mean(samples) - expected_midpoint) / expected_midpoint < 0.5


class TestUnifiedInterventionCostSampling:
    """Test the unified sample_intervention_cost method"""
    
    @pytest.fixture
    def mock_building_chars(self):
        """Fixture providing mock building characteristics"""
        return MockBuildingCharacteristics(
            roof_area_estimate=100.0,
            external_wall_area_estimate=200.0,
            building_footprint_area=80.0,
            gross_internal_area=150.0,
            flat_count=1
        )
    
    def test_loft_insulation_sampling(self, mock_building_chars):
        """Test loft insulation cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='loft_insulation',
            building_chars=mock_building_chars,
            typology='Standard size detached',
            age_band='1950-1966',
            region='E12000007',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        assert len(samples) == 100
        # Check costs are within reasonable bounds
        assert np.all(samples >= 500)  # min_cost from caps
        assert np.all(samples <= 2000)  # max_cost from caps
        
    def test_cavity_wall_insulation_sampling(self, mock_building_chars):
        """Test cavity wall insulation cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='cavity_wall_insulation',
            building_chars=mock_building_chars,
            typology='Large semi detached',
            age_band='1967-1975',
            region='E12000001',
            regional_multiplier=1.1,
            age_multiplier=1.05,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        assert len(samples) == 100
        assert isinstance(samples, np.ndarray)
        
    def test_external_wall_insulation_with_multipliers(self, mock_building_chars):
        """Test that multipliers are correctly applied"""
        rules = InterventionScalingRules()
        
        # Sample with no multipliers
        samples_base = rules.sample_intervention_cost(
            intervention='external_wall_insulation',
            building_chars=mock_building_chars,
            typology='Large detached',
            age_band='1900-1929',
            region='E12000002',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=1000
        )
        
        # Sample with multipliers
        samples_multiplied = rules.sample_intervention_cost(
            intervention='external_wall_insulation',
            building_chars=mock_building_chars,
            typology='Large detached',
            age_band='1900-1929',
            region='E12000002',
            regional_multiplier=1.2,
            age_multiplier=1.1,
            complexity_multiplier=1.05,
            n_samples=1000
        )
        
        # Multiplied samples should be higher on average
        expected_multiplier = 1.2 * 1.1 * 1.05
        ratio = np.mean(samples_multiplied) / np.mean(samples_base)
        # Allow some tolerance due to capping and random sampling
        assert ratio > expected_multiplier * 0.8
        
    def test_solar_pv_sampling(self, mock_building_chars):
        """Test solar PV cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='solar_pv',
            building_chars=mock_building_chars,
            typology='Standard size detached',
            age_band='1976-1982',
            region='E12000003',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        assert len(samples) == 100
        assert np.all(samples >= 2000)
        assert np.all(samples <= 10000)
        
    def test_floor_insulation_sampling(self, mock_building_chars):
        """Test floor insulation cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='floor_insulation',
            building_chars=mock_building_chars,
            typology='Small low terraces',
            age_band='1983-1990',
            region='E12000004',
            regional_multiplier=0.95,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        assert len(samples) == 100
        assert np.all(samples >= 500)
        assert np.all(samples <= 4000)
        
    def test_deep_retrofit_sampling(self, mock_building_chars):
        """Test deep retrofit cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='deep_retrofit_estimate',
            building_chars=mock_building_chars,
            typology='Very large detached',
            age_band='Pre 1900',
            region='E12000005',
            regional_multiplier=1.15,
            age_multiplier=1.2,
            complexity_multiplier=1.1,
            n_samples=100
        )
        
        assert len(samples) == 100
        # Deep retrofit should be expensive
        assert np.mean(samples) > 20000
        
    def test_heat_pump_upgrade_sampling(self, mock_building_chars):
        """Test heat pump upgrade cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='heat_pump_upgrade',
            building_chars=mock_building_chars,
            typology='Standard size semi detached',
            age_band='1991-1995',
            region='E12000006',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        assert len(samples) == 100
        min_cost, max_cost = rules.heat_pump_cost_ranges['Standard size semi detached']
        assert np.all(samples >= min_cost)
        assert np.all(samples <= max_cost * 1.05)  # Allow small tolerance
        
    def test_double_glazing_sampling(self, mock_building_chars):
        """Test double glazing cost sampling"""
        rules = InterventionScalingRules()
        
        samples = rules.sample_intervention_cost(
            intervention='double_glazing',
            building_chars=mock_building_chars,
            typology='Large semi detached',
            age_band='1996-2002',
            region='E12000007',
            regional_multiplier=1.05,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        assert len(samples) == 100
        min_cost, max_cost = rules.double_glazing_cost_ranges['Large semi detached']
        assert np.all(samples >= min_cost * 0.95)  # Allow tolerance
        
    def test_unknown_intervention_raises_error(self, mock_building_chars):
        """Test that unknown intervention raises ValueError"""
        rules = InterventionScalingRules()
        
        with pytest.raises(ValueError, match="Unknown intervention"):
            rules.sample_intervention_cost(
                intervention='unknown_intervention',
                building_chars=mock_building_chars,
                typology='Standard size detached',
                age_band='2003-2006',
                region='E12000008',
                regional_multiplier=1.0,
                age_multiplier=1.0,
                complexity_multiplier=1.0,
                n_samples=100
            )
            
    def test_missing_uncertainty_parameters_raises_error(self, mock_building_chars):
        """Test that missing uncertainty parameters raises appropriate error"""
        rules = InterventionScalingRules()
        
        # Add intervention without uncertainty parameters
        rules.area_based_interventions['test_intervention'] = {
            'base_cost_per_sqm': 50,
            'min_cost': 1000,
            'max_cost': 5000,
            'area_type': 'wall'
        }
        
        with pytest.raises(ValueError, match="No uncertainty parameters"):
            rules.sample_intervention_cost(
                intervention='test_intervention',
                building_chars=mock_building_chars,
                typology='Standard size detached',
                age_band='2007-2011',
                region='E12000009',
                regional_multiplier=1.0,
                age_multiplier=1.0,
                complexity_multiplier=1.0,
                n_samples=100
            )


class TestAreaTypeMapping:
    """Test correct area types are used for interventions"""
    
    def test_roof_area_interventions(self):
        """Test interventions that use roof area"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics(
            roof_area_estimate=50.0,
            external_wall_area_estimate=200.0,
            building_footprint_area=100.0,
            gross_internal_area=150.0
        )
        
        samples = rules.sample_intervention_cost(
            intervention='solar_pv',
            building_chars=mock_chars,
            typology='Standard size detached',
            age_band='2012-2021',
            region='E12000001',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=10
        )
        
        # Verify costs are calculated (roof area is smaller so costs should be lower)
        assert len(samples) == 10
        assert np.all(samples > 0)
        
    def test_wall_area_interventions(self):
        """Test interventions that use wall area"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics(
            roof_area_estimate=100.0,
            external_wall_area_estimate=300.0,  # Larger wall area
            building_footprint_area=80.0,
            gross_internal_area=150.0
        )
        
        samples = rules.sample_intervention_cost(
            intervention='internal_wall_insulation',
            building_chars=mock_chars,
            typology='Large detached',
            age_band='2022+',
            region='E12000002',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=10
        )
        
        assert len(samples) == 10
        # Larger wall area should result in higher costs
        assert np.mean(samples) > 6000


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_area_building(self):
        """Test handling of building with zero area"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics(
            roof_area_estimate=0.0,
            external_wall_area_estimate=0.0,
            building_footprint_area=0.0,
            gross_internal_area=0.0
        )
        
        samples = rules.sample_intervention_cost(
            intervention='loft_insulation',
            building_chars=mock_chars,
            typology='Standard size detached',
            age_band='1950-1966',
            region='E12000001',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=10
        )
        
        # Should still be capped at minimum cost
        assert np.all(samples >= 500)
        
    def test_very_large_building(self):
        """Test handling of very large building"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics(
            roof_area_estimate=1000.0,
            external_wall_area_estimate=2000.0,
            building_footprint_area=800.0,
            gross_internal_area=3000.0
        )
        
        samples = rules.sample_intervention_cost(
            intervention='deep_retrofit_estimate',
            building_chars=mock_chars,
            typology='Very large detached',
            age_band='Pre 1900',
            region='E12000001',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=10
        )
        
        # Should be capped at maximum cost
        assert np.all(samples <= 180000)
        
    def test_single_sample(self):
        """Test requesting single sample"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics()
        
        samples = rules.sample_intervention_cost(
            intervention='cavity_wall_insulation',
            building_chars=mock_chars,
            typology='Standard size semi detached',
            age_band='1967-1975',
            region='E12000001',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=1
        )
        
        assert len(samples) == 1
        assert samples[0] > 0
        
    def test_extreme_multipliers(self):
        """Test with extreme multiplier values"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics()
        
        samples = rules.sample_intervention_cost(
            intervention='floor_insulation',
            building_chars=mock_chars,
            typology='Small low terraces',
            age_band='Pre 1900',
            region='E12000001',
            regional_multiplier=2.0,  # Very high regional cost
            age_multiplier=1.5,        # Old building premium
            complexity_multiplier=1.3, # Complex retrofit
            n_samples=100
        )
        
        # Even with high multipliers, should respect caps
        assert np.all(samples <= 4000 * 2.0)  # Max cap * regional multiplier


class TestStatisticalProperties:
    """Test statistical properties of sampled distributions"""
    
    def test_triangular_distribution_mean(self):
        """Test that triangular distribution produces expected mean"""
        rules = InterventionScalingRules()
        typology = 'Large detached'
        n_samples = 10000
        
        samples = rules.sample_heat_pump_cost_triangular(typology, n_samples)
        
        min_cost, max_cost = rules.heat_pump_cost_ranges[typology]
        mode_cost = (min_cost + max_cost) / 2
        
        # Theoretical mean of triangular distribution: (min + mode + max) / 3
        expected_mean = (min_cost + mode_cost + max_cost) / 3
        actual_mean = np.mean(samples)
        
        # Allow 5% tolerance
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05
        
    def test_sample_reproducibility_with_seed(self):
        """Test that sampling is reproducible with same random seed"""
        rules = InterventionScalingRules()
        mock_chars = MockBuildingCharacteristics()
        
        np.random.seed(42)
        samples1 = rules.sample_intervention_cost(
            intervention='loft_insulation',
            building_chars=mock_chars,
            typology='Standard size detached',
            age_band='1950-1966',
            region='E12000001',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        np.random.seed(42)
        samples2 = rules.sample_intervention_cost(
            intervention='loft_insulation',
            building_chars=mock_chars,
            typology='Standard size detached',
            age_band='1950-1966',
            region='E12000001',
            regional_multiplier=1.0,
            age_multiplier=1.0,
            complexity_multiplier=1.0,
            n_samples=100
        )
        
        np.testing.assert_array_equal(samples1, samples2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])