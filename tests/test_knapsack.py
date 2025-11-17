import pytest
import pandas as pd
import numpy as np
from io import StringIO
import sys
sys.path.append('/Users/gracecolverd/RetrofitModel')
from src.GreedyAlgo import true_greedy_knapsack

import pytest
import pandas as pd
import numpy as np
from io import StringIO
import sys


def test_basic_greedy_selection():
    """Test basic greedy selection with sufficient budget."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C', 'D'],
        'cost of interventon_mean': [100, 200, 300, 400],
        'cost_per_net_ton_co2_kg': [10, 15, 5, 20],  # C is most efficient
        'total_ton_co2_saved': [10, 13.33, 60, 20]
    })
    
    budget = 500
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Should select C (300, efficiency 5), then A (100, efficiency 10)
    assert len(selected) == 2
    assert set(selected['upn'].values) == {'C', 'A'}
    assert remaining == 100  # 500 - 300 - 100
    assert selected.iloc[0]['upn'] == 'C'  # Most efficient first


def test_exact_budget_match():
    """Test when selected interventions exactly match the budget."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C'],
        'cost of interventon_mean': [100, 200, 300],
        'cost_per_net_ton_co2_kg': [5, 10, 15],
        'total_ton_co2_saved': [20, 20, 20]
    })
    
    budget = 300
    selected, remaining = true_greedy_knapsack(df, budget)
    
    assert len(selected) == 2  # A and B
    assert remaining == 0
    assert selected['cost of interventon_mean'].sum() == 300


def test_insufficient_budget():
    """Test when budget cannot afford any intervention."""
    df = pd.DataFrame({
        'upn': ['A', 'B'],
        'cost of interventon_mean': [1000, 2000],
        'cost_per_net_ton_co2_kg': [10, 15],
        'total_ton_co2_saved': [100, 133]
    })
    
    budget = 500
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Function should return empty DataFrame and full budget
    assert len(selected) == 0
    assert remaining == 500


def test_single_intervention():
    """Test with only one intervention available."""
    df = pd.DataFrame({
        'upn': ['A'],
        'cost of interventon_mean': [100],
        'cost_per_net_ton_co2_kg': [10],
        'total_ton_co2_saved': [10]
    })
    
    budget = 200
    selected, remaining = true_greedy_knapsack(df, budget)
    
    assert len(selected) == 1
    assert selected.iloc[0]['upn'] == 'A'
    assert remaining == 100


def test_empty_dataframe():
    """Test with empty DataFrame."""
    df = pd.DataFrame(columns=[
        'upn', 
        'cost of interventon_mean', 
        'cost_per_net_ton_co2_kg',
        'total_ton_co2_saved'
    ])
    
    budget = 1000
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Function should return empty DataFrame and full budget
    assert len(selected) == 0
    assert remaining == 1000


def test_zero_budget():
    """Test with zero budget."""
    df = pd.DataFrame({
        'upn': ['A', 'B'],
        'cost of interventon_mean': [100, 200],
        'cost_per_net_ton_co2_kg': [10, 15],
        'total_ton_co2_saved': [10, 13.33]
    })
    
    budget = 0
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Function should return empty DataFrame and zero budget
    assert len(selected) == 0
    assert remaining == 0


def test_custom_column_names():
    """Test with custom column names."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C'],
        'total_cost': [100, 200, 300],
        'efficiency_metric': [5, 10, 15],
        'total_ton_co2_saved': [20, 20, 20]
    })
    
    budget = 400
    selected, remaining = true_greedy_knapsack(
        df, 
        budget,
        cost_column='total_cost',
        efficiency_column='efficiency_metric'
    )
    
    assert len(selected) == 2
    assert remaining == 100


def test_sorting_by_efficiency():
    """Test that interventions are correctly sorted by efficiency."""
    df = pd.DataFrame({
        'upn': ['Worst', 'Best', 'Middle'],
        'cost of interventon_mean': [100, 100, 100],
        'cost_per_net_ton_co2_kg': [50, 5, 25],  # Best has lowest cost per ton
        'total_ton_co2_saved': [2, 20, 4]
    })
    
    budget = 250
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Should select Best first, then Middle
    assert len(selected) == 2
    assert selected.iloc[0]['upn'] == 'Best'
    assert selected.iloc[1]['upn'] == 'Middle'


def test_all_interventions_affordable():
    """Test when all interventions can be afforded."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C'],
        'cost of interventon_mean': [100, 150, 200],
        'cost_per_net_ton_co2_kg': [10, 12, 8],
        'total_ton_co2_saved': [10, 12.5, 25]
    })
    
    budget = 1000
    selected, remaining = true_greedy_knapsack(df, budget)
    
    assert len(selected) == 3
    assert remaining == 550


def test_total_cost_calculation():
    """Test that total cost is correctly calculated."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C'],
        'cost of interventon_mean': [123.45, 234.56, 345.67],
        'cost_per_net_ton_co2_kg': [5, 10, 15],
        'total_ton_co2_saved': [24.69, 23.46, 23.04]
    })
    
    budget = 500
    selected, remaining = true_greedy_knapsack(df, budget)
    
    total_selected_cost = selected['cost of interventon_mean'].sum()
    assert pytest.approx(budget - remaining, 0.01) == total_selected_cost


def test_tie_in_efficiency():
    """Test behavior when multiple interventions have the same efficiency."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C'],
        'cost of interventon_mean': [100, 200, 150],
        'cost_per_net_ton_co2_kg': [10, 10, 10],  # All same efficiency
        'total_ton_co2_saved': [10, 20, 15]
    })
    
    budget = 350
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Should select until budget exhausted
    assert len(selected) >= 1
    assert selected['cost of interventon_mean'].sum() <= budget


def test_very_small_costs():
    """Test with very small intervention costs."""
    df = pd.DataFrame({
        'upn': ['A', 'B', 'C'],
        'cost of interventon_mean': [0.01, 0.02, 0.03],
        'cost_per_net_ton_co2_kg': [0.001, 0.002, 0.003],
        'total_ton_co2_saved': [10, 10, 10]
    })
    
    budget = 0.05
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Budget 0.05 can afford A (0.01) + B (0.02) = 0.03, leaving 0.02
    # Cannot afford C (0.03) since 0.03 > 0.02 remaining
    assert len(selected) == 2
    assert pytest.approx(remaining, 0.001) == 0.02


def test_large_dataset():
    """Test with a larger dataset to ensure scalability."""
    n = 1000
    df = pd.DataFrame({
        'upn': [f'Building_{i}' for i in range(n)],
        'cost of interventon_mean': np.random.uniform(100, 10000, n),
        'cost_per_net_ton_co2_kg': np.random.uniform(1, 50, n),
        'total_ton_co2_saved': np.random.uniform(10, 100, n)
    })
    
    budget = 500000
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Verify budget constraint
    assert selected['cost of interventon_mean'].sum() <= budget
    assert remaining >= 0
    
    # Verify efficiency ordering (only if we selected items)
    if len(selected) > 1:
        efficiencies = selected['cost_per_net_ton_co2_kg'].values
        assert all(efficiencies[i] <= efficiencies[i+1] for i in range(len(efficiencies)-1))


def test_console_output_suppression(capsys):
    """Test that function produces expected console output."""
    df = pd.DataFrame({
        'upn': ['A', 'B'],
        'cost of interventon_mean': [100, 200],
        'cost_per_net_ton_co2_kg': [10, 15],
        'total_ton_co2_saved': [10, 13.33]
    })
    
    budget = 300
    selected, remaining = true_greedy_knapsack(df, budget)
    
    captured = capsys.readouterr()
    assert "Starting true greedy selection" in captured.out
    assert "Selection Complete" in captured.out
    assert "Buildings covered: 2" in captured.out


def test_output_dataframe_structure():
    """Test that output DataFrame has the same structure as input."""
    df = pd.DataFrame({
        'upn': ['A', 'B'],
        'cost of interventon_mean': [100, 200],
        'cost_per_net_ton_co2_kg': [10, 15],
        'total_ton_co2_saved': [10, 13.33],
        'extra_column': ['X', 'Y']
    })
    
    budget = 300
    selected, remaining = true_greedy_knapsack(df, budget)
    
    # Check that all columns are preserved
    assert set(selected.columns) == set(df.columns)
    assert 'extra_column' in selected.columns


def test_console_output_no_selection(capsys):
    """Test console output when no interventions are selected."""
    df = pd.DataFrame({
        'upn': ['A', 'B'],
        'cost of interventon_mean': [1000, 2000],
        'cost_per_net_ton_co2_kg': [10, 15],
        'total_ton_co2_saved': [10, 13.33]
    })
    
    budget = 500
    selected, remaining = true_greedy_knapsack(df, budget)
    
    captured = capsys.readouterr()
    assert "Starting true greedy selection" in captured.out
    assert "No interventions selected" in captured.out or "budget insufficient" in captured.out.lower()