import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import pandas as pd 

# Assuming these imports work in your actual codebase
import sys 
sys.path.append('/Users/gracecolverd/RetrofitModel')

# Imports from your module
from src.RetrofitCosts import CostEstimator, InterventionConfig, INTERVENTION_CONFIGS
# We will use the MockBuildingCharacteristics defined below for testing
# from src.BuildingCharacteristics import BuildingCharacteristics


# Mock BuildingCharacteristics for testing
@dataclass
class MockBuildingCharacteristics:
    roof_area_estimate: float = 100.0
    external_wall_area_estimate: float = 200.0
    number_of_flats: int = 1


# === FIXTURES (GLOBAL SCOPE) ===

@pytest.fixture
def building_chars():
    """Standard building characteristics for testing"""
    return MockBuildingCharacteristics()

@pytest.fixture
def multi_flat_building():
    """Building with multiple flats and larger wall area for cap testing"""
    return MockBuildingCharacteristics(external_wall_area_estimate=800.0, number_of_flats=4)

@pytest.fixture
def cost_estimator():
    """CostEstimator instance with default configs"""
    return CostEstimator()

@pytest.fixture
def mock_configs():
    """Mock intervention configurations for testing"""
    return {
        'test_roof': InterventionConfig(
            area_type='roof',
            epist_scenarios={
                'central': {
                    'cost_min': 10, 'cost_mode': 20, 'cost_max': 30,
                    'cap_min': 500, 'cap_max': 2000
                }
            }
        ),
        'test_wall': InterventionConfig(
            area_type='wall',
            epist_scenarios={
                'central': {
                    'cost_min': 10, 'cost_mode': 20, 'cost_max': 30,
                    'cap_min': 500, 'cap_max': 8500
                }
            }
        ),
        'test_typology': InterventionConfig(
            area_type='typology_based',
            epist_scenarios={
                'central': {
                    'cost_by_typology': {
                        'Small low terraces': (7000, 9000),
                        'all_unknown_typology': (10000, 15000)
                    },
                    'cap_min': 6000,
                    'cap_max': 25000
                }
            }
        )
    }

# ========================================


class TestCostEstimator:
    """Test suite for CostEstimator class"""
    pass


class TestGetAreaForIntervention:
    """Tests for get_area_for_intervention method"""
    
    def test_roof_area_type(self, cost_estimator, building_chars):
        """Test that roof area type returns correct area"""
        area = cost_estimator.get_area_for_intervention('loft_percentile', building_chars)
        assert area == building_chars.roof_area_estimate
    
    def test_wall_area_type(self, cost_estimator, building_chars):
        """Test that wall area type returns correct area"""
        area = cost_estimator.get_area_for_intervention('cavity_wall_percentile', building_chars)
        assert area == building_chars.external_wall_area_estimate
    
    def test_typology_based_area(self, cost_estimator, building_chars):
        """Test that typology-based interventions return 1.0"""
        area = cost_estimator.get_area_for_intervention('heat_pump_percentile', building_chars)
        assert area == 1.0
    
    def test_unknown_intervention_raises_error(self, cost_estimator, building_chars):
        """Test that unknown intervention raises ValueError"""
        with pytest.raises(ValueError, match="Unknown intervention"):
            cost_estimator.get_area_for_intervention('nonexistent_intervention', building_chars)

 


class TestSampleInterventionCost_original:
    """Tests for sample_intervention_cost method"""
    
    def test_basic_cost_sampling(self, cost_estimator, building_chars):
        """Test basic cost sampling returns array of correct length"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            n_samples=100
        )
        assert len(costs) == 100
        assert all(isinstance(c, (int, float, np.number)) for c in costs)
    
    def test_central_scenario(self, cost_estimator, building_chars):
        """Test sampling with central scenario"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='central',
            n_samples=1000
        )
        # Costs should be within expected range from INTERVENTION_CONFIGS
        assert all(costs >= 500)  # cap_min
        assert all(costs <= 2000)  # cap_max
    
    def test_optimistic_scenario(self, cost_estimator, building_chars):
        """Test that optimistic scenario produces lower costs"""
        np.random.seed(42)
        central_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='central',
            n_samples=1000
        )
        
        np.random.seed(42)
        optimistic_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='optimistic',
            n_samples=1000
        )
        
        assert np.mean(optimistic_costs) < np.mean(central_costs)
    
    def test_pessimistic_scenario(self, cost_estimator, building_chars):
        """Test that pessimistic scenario produces higher costs"""
        np.random.seed(42)
        central_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='central',
            n_samples=1000
        )
        
        np.random.seed(42)
        pessimistic_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='pessimistic',
            n_samples=1000
        )
        
        assert np.mean(pessimistic_costs) > np.mean(central_costs)
    
    def test_regional_multiplier(self, cost_estimator, building_chars):
        """Test that regional multiplier scales costs correctly"""
        np.random.seed(42)
        base_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            regional_multiplier=1.0,
            n_samples=100
        )
        
        np.random.seed(42)
        scaled_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            regional_multiplier=1.5,
            n_samples=100
        )
        
        # Note: Caps may affect this, so we check that scaled is generally higher
        assert np.mean(scaled_costs) >= np.mean(base_costs)
    
    def test_age_multiplier(self, cost_estimator, building_chars):
        """Test that age multiplier scales costs correctly"""
        np.random.seed(42)
        base_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            age_multiplier=1.0,
            n_samples=100
        )
        
        np.random.seed(42)
        scaled_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            age_multiplier=1.3,
            n_samples=100
        )
        
        assert np.mean(scaled_costs) >= np.mean(base_costs)
    
    def test_combined_multipliers(self, cost_estimator, building_chars):
        """Test that multiple multipliers compound correctly"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            regional_multiplier=1.2,
            age_multiplier=1.3,
            n_samples=100
        )
        
        # Costs should be affected by both multipliers
        assert len(costs) == 100
        # With multipliers, costs should still respect caps
        assert all(costs >= 500)
        assert all(costs <= 2000)
    
    def test_caps_applied_correctly(self, cost_estimator):
        """Test that cost caps are enforced"""
        # Use very large area to test caps
        large_building = MockBuildingCharacteristics(roof_area_estimate=1000.0)
        
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=large_building,
            n_samples=100
        )
        
        # All costs should be within caps
        assert all(costs >= 500)  # cap_min
        assert all(costs <= 2000)  # cap_max
    
    def test_typology_based_intervention(self, cost_estimator, building_chars):
        """Test typology-based cost calculation"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='heat_pump_percentile',
            building_chars=building_chars,
            typology='Small low terraces',
            n_samples=100
        )
        
        # Should be within the typology range (7000-9000)
        assert all(costs >= 6000)  # cap_min
        assert all(costs <= 25000)  # cap_max
        # With seed 42, n=100, mean is 8009.64
        assert 7900 <= np.mean(costs) <= 8100
    
    def test_unknown_typology_uses_default(self, cost_estimator, building_chars):
        """Test that unknown typology uses default range"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='heat_pump_percentile',
            building_chars=building_chars,
            typology='Unknown Typology Type',
            n_samples=100
        )
        
        # Should use default range (10000-15000)
        assert all(costs >= 6000)
        assert all(costs <= 25000)
        
        # With seed 42, n=100, mean is 12393.43
        assert 12390 <= np.mean(costs) <= 12400
    
    def test_wall_insulation_multi_flat_scaling(self, cost_estimator, multi_flat_building):
        """Test that wall insulation caps scale for multi-flat buildings"""
        np.random.seed(42)
        single_flat = MockBuildingCharacteristics(
            external_wall_area_estimate=800.0, # Use same large area
            number_of_flats=1
        )
        
        single_costs = cost_estimator.sample_intervention_cost(
            intervention='cavity_wall_percentile',
            building_chars=single_flat,
            n_samples=100
        )
        
        np.random.seed(42)
        multi_costs = cost_estimator.sample_intervention_cost(
            intervention='cavity_wall_percentile',
            building_chars=multi_flat_building, # Uses 800 area, 4 flats
            n_samples=100
        )
        
        # The single flat cost IS capped at 8500.
        assert np.isclose(max(single_costs), 8500)
        
        # The multi-flat cost IS NOT capped at 8500, it's higher.
        # This proves the scaling logic is working.
        assert max(multi_costs) > 8500
        
        # The true max is 22704.44...
        # The scaled cap is (8500 * 4 * 0.8) = 27200. The cost is below this.
        assert np.isclose(max(multi_costs), 22704.44138876939)
        assert max(multi_costs) <= 27200
    
    def test_unknown_scenario_falls_back_to_central(self, cost_estimator, building_chars):
        """Test that unknown scenario falls back to central"""
        np.random.seed(42)
        central_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='central',
            n_samples=100
        )
        
        np.random.seed(42)
        unknown_costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            epist_scenario='nonexistent_scenario',
            n_samples=100
        )
        
        # Should produce same distribution
        np.testing.assert_array_almost_equal(central_costs, unknown_costs)


class TestSampleCostForPackage:
    """Tests for sample_cost_for_package method"""
    
    @patch('src.RetrofitCosts.get_intervention_list')
    def test_single_intervention_package(self, mock_get_list, cost_estimator, building_chars):
        """Test package with single intervention"""
        mock_get_list.return_value = ['loft_percentile']
        
        np.random.seed(42)
        costs = cost_estimator.sample_cost_for_package(
            intervention='package_1',
            building_chars=building_chars,
            wall_type='cavity',
            n_samples=100
        )
        
        assert len(costs) == 100
        assert all(costs > 0)
    
    @patch('src.RetrofitCosts.get_intervention_list')
    def test_multiple_intervention_package(self, mock_get_list, cost_estimator, building_chars):
        """Test package with multiple interventions"""
        mock_get_list.return_value = ['loft_percentile', 'cavity_wall_percentile']
        
        np.random.seed(42)
        package_costs = cost_estimator.sample_cost_for_package(
            intervention='package_2',
            building_chars=building_chars,
            wall_type='cavity',
            n_samples=100
        )
        
        np.random.seed(42)
        loft_costs = cost_estimator.sample_intervention_cost(
            'loft_percentile', building_chars, n_samples=100
        )
        
        np.random.seed(42)
        wall_costs = cost_estimator.sample_intervention_cost(
            'cavity_wall_percentile', building_chars, n_samples=100
        )
        
        # A simple, robust check:
        assert np.mean(package_costs) > np.mean(loft_costs)
        assert np.mean(package_costs) > np.mean(wall_costs)

    
    @patch('src.RetrofitCosts.get_intervention_list')
    def test_package_with_get_intervention_list_exception(self, mock_get_list, cost_estimator, building_chars):
        """Test that if get_intervention_list fails, single intervention is used"""
        mock_get_list.side_effect = Exception("Not a package")
        
        np.random.seed(42)
        costs = cost_estimator.sample_cost_for_package(
            intervention='loft_percentile', # This is NOT a package
            building_chars=building_chars,
            n_samples=100
        )
        
        # Should still work by treating as single intervention
        assert len(costs) == 100
        assert all(costs > 0)
    
    @patch('src.RetrofitCosts.get_intervention_list')
    def test_package_passes_kwargs(self, mock_get_list, cost_estimator, building_chars):
        """Test that all kwargs are passed to individual interventions"""
        mock_get_list.return_value = ['loft_percentile']
        
        # Mock the underlying cost sampler to check what it was called with
        # and provide a return value to prevent the ValueError
        cost_estimator.sample_intervention_cost = MagicMock(return_value=np.zeros(50))
        
        cost_estimator.sample_cost_for_package(
            intervention='package_1',
            building_chars=building_chars,
            wall_type='cavity',
            n_samples=50,
            epist_scenario='optimistic',
            regional_multiplier=1.2,
            age_multiplier=1.1
        )
        
        cost_estimator.sample_intervention_cost.assert_called_once_with(
            intervention='loft_percentile',
            building_chars=building_chars,
            wall_type='cavity',
            n_samples=50,
            epist_scenario='optimistic',
            regional_multiplier=1.2,
            age_multiplier=1.1
        )


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_samples(self, cost_estimator, building_chars):
        """Test behavior with zero samples"""
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            n_samples=0
        )
        assert len(costs) == 0
    
    def test_single_sample(self, cost_estimator, building_chars):
        """Test with single sample"""
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            n_samples=1
        )
        assert len(costs) == 1
        assert isinstance(costs[0], (int, float, np.number))
    
    def test_very_small_area(self, cost_estimator):
        """Test with very small building area"""
        tiny_building = MockBuildingCharacteristics(
            roof_area_estimate=1.0,
            external_wall_area_estimate=1.0
        )
        
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=tiny_building,
            n_samples=10
        )
        
        # Should still apply cap_min
        assert all(costs >= 500)
    
    def test_custom_configs(self, mock_configs):
        """Test CostEstimator with custom configs"""
        custom_estimator = CostEstimator(configs=mock_configs)
        building = MockBuildingCharacteristics()
        
        costs = custom_estimator.sample_intervention_cost(
            intervention='test_roof',
            building_chars=building,
            n_samples=10
        )
        
        assert len(costs) == 10
    
    def test_zero_multipliers(self, cost_estimator, building_chars):
        """Test with zero multipliers (edge case)"""
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            regional_multiplier=0.0,
            age_multiplier=0.0,
            n_samples=10
        )
        
        # With zero multipliers, cost becomes 0, which is then clipped to cap_min
        assert all(costs == 500)
    
    def test_very_large_multipliers(self, cost_estimator, building_chars):
        """Test with very large multipliers"""
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            regional_multiplier=100.0,
            age_multiplier=100.0,
            n_samples=10
        )
        
        # Should be capped at cap_max
        assert all(costs == 2000)


class TestStatisticalProperties:
    """Test statistical properties of cost distributions"""
    
    def test_triangular_distribution_shape(self, cost_estimator, building_chars):
        """Test that costs follow triangular distribution"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            n_samples=10000
        )
        
        # Basic distribution checks
        assert np.std(costs) > 0  # Has variance
        assert np.min(costs) >= 500  # Respects minimum
        assert np.max(costs) <= 2000  # Respects maximum
    
    def test_reproducibility_with_seed(self, cost_estimator, building_chars):
        """Test that setting seed produces reproducible results"""
        np.random.seed(42)
        costs1 = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            n_samples=100
        )
        
        np.random.seed(42)
        costs2 = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars,
            n_samples=100
        )
        
        np.testing.assert_array_equal(costs1, costs2)


# Parametrized tests
class TestParametrizedScenarios:
    """Parametrized tests for different scenarios"""
    
    # Updated with correct mean values for loft_percentile (area=100)
    @pytest.mark.parametrize("scenario,expected_mean_range", [
        # With seed=42, n=1000, area=100, mean is ~1823.96
        ('central', (1820, 1825)),
        # With seed=42, n=1000, area=100, mean is ~1533.85
        ('optimistic', (1530, 1535)),
        # With seed=42, n=1000, area=100, mean is ~2371.15
        ('pessimistic', (2370, 2372))
    ])
    def test_scenario_cost_ranges(self, cost_estimator, building_chars, scenario, expected_mean_range):
        """Test that different scenarios produce costs in expected ranges"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='loft_percentile',
            building_chars=building_chars, # roof_area_estimate=100
            epist_scenario=scenario,
            n_samples=1000
        )
        
        mean_cost = np.mean(costs)
        assert expected_mean_range[0] <= mean_cost <= expected_mean_range[1]

    # Updated with correct mean values for cavity_wall_percentile (area=200)
    @pytest.mark.parametrize("scenario,expected_mean_range", [
        # With seed=42, n=1000, area=200, mean is ~3972.68
        ('central', (3970, 3975)),
        # With seed=42, n=1000, area=200, mean is ~3178.14
        ('optimistic', (3175, 3180)),
        # With seed=42, n=1000, area=200, mean is ~5164.48
        ('pessimistic', (5160, 5165))
    ])
    def test_wall_scenario_cost_ranges(self, cost_estimator, building_chars, scenario, expected_mean_range):
        """Test that wall scenarios produce costs in expected ranges"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='cavity_wall_percentile',
            building_chars=building_chars, # external_wall_area_estimate=200
            epist_scenario=scenario,
            n_samples=1000
        )
        
        mean_cost = np.mean(costs)
        assert expected_mean_range[0] <= mean_cost <= expected_mean_range[1]
    
    # *** NEW TEST ***
    # Calculated mean values for solid_wall_internal_percentile (area=200)
    @pytest.mark.parametrize("scenario,expected_mean_range", [
        # With seed=42, n=1000, area=200, mean is ~19218.52
        ('central', (19215, 19220)),
        # With seed=42, n=1000, area=200, mean is ~15374.81
        ('optimistic', (15370, 15375)),
        # With seed=42, n=1000, area=200, mean is ~25051.29
        ('pessimistic', (25050, 25055))
    ])
    def test_internal_wall_scenario_cost_ranges(self, cost_estimator, building_chars, scenario, expected_mean_range):
        """Test that internal wall scenarios produce costs in expected ranges"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='solid_wall_internal_percentile',
            building_chars=building_chars, # external_wall_area_estimate=200
            epist_scenario=scenario,
            n_samples=1000
        )
        
        mean_cost = np.mean(costs)
        assert expected_mean_range[0] <= mean_cost <= expected_mean_range[1]

    # *** NEW TEST ***
    # Calculated mean values for solid_wall_external_percentile (area=200)
    @pytest.mark.parametrize("scenario,expected_mean_range", [
        # With seed=42, n=1000, area=200, mean is ~22877.05
        ('central', (22875, 22880)),
        # With seed=42, n=1000, area=200, mean is ~18301.64
        ('optimistic', (18300, 18305)),
        # With seed=42, n=1000, area=200, mean is ~29773.25
        ('pessimistic', (29770, 29775))
    ])
    def test_external_wall_scenario_cost_ranges(self, cost_estimator, building_chars, scenario, expected_mean_range):
        """Test that external wall scenarios produce costs in expected ranges"""
        np.random.seed(42)
        costs = cost_estimator.sample_intervention_cost(
            intervention='solid_wall_external_percentile',
            building_chars=building_chars, # external_wall_area_estimate=200
            epist_scenario=scenario,
            n_samples=1000
        )
        
        mean_cost = np.mean(costs)
        assert expected_mean_range[0] <= mean_cost <= expected_mean_range[1]

    @pytest.mark.parametrize("intervention,area_type", [
        ('loft_percentile', 'roof'),
        ('cavity_wall_percentile', 'wall'),
        ('solid_wall_internal_percentile', 'wall'),
        ('solid_wall_external_percentile', 'wall'),
        ('heat_pump_percentile', 'typology_based')
    ])
    def test_all_interventions_produce_valid_costs(self, cost_estimator, building_chars, intervention, area_type):
        """Test that all configured interventions produce valid costs"""
        costs = cost_estimator.sample_intervention_cost(
            intervention=intervention,
            building_chars=building_chars,
            typology='Small low terraces' if area_type == 'typology_based' else None,
            n_samples=10
        )
        
        assert len(costs) == 10
        assert all(costs > 0)
        assert all(np.isfinite(costs))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])