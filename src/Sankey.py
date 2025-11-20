import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os 

def run_sankey_greedy(selected_projects, op):
    
    sk_data=selected_projects
    op_path=os.path.join(op, 'sankey' )
    os.makedirs(op_path  , exist_ok=True ) 
    # Create and show the plot
    fig1 = plot_sankey_3layers(
        sk_data, 
        col_layer2='meta_socio_persona', 
        col_layer3='scenario',
        title='Gas Percentiles Flow Analysis: Costs',
        metric_col='cost_of_intervention_mean',  # Sum costs instead of counting
    
    )

    # Create and show the plot
    fig2 = plot_sankey_3layers(
        sk_data, 
        col_layer2='meta_socio_persona', 
        col_layer3='scenario',
        title='Gas Percentiles Flow Analysis: Tons Saved',
        metric_col='total_ton_co2_saved_mean',  # Sum costs instead of counting
    
    )


    # Create and show the plot
    fig3 = plot_sankey_3layers(
        sk_data, 
        col_layer2='meta_socio_persona', 
        col_layer3='scenario',
        title='Gas Percentiles Flow Analysis: Counts',
        
    
    )
    fig1.write_html(f"{op_path}/sankey_diagram_costs.html")
    fig2.write_html(f"{op_path}/sankey_diagram_tons_saved.html")
    fig2.write_html(f"{op_path}/sankey_diagram_counts.html")

 

    


def plot_sankey_3layers(proc_df, col_layer2, col_layer3, metric_col=None, 
                       min_flow=0, title="Sankey Diagram"):
    """
    Create a simple 3-layer Sankey diagram.
    
    Parameters:
    -----------
    proc_df : pd.DataFrame
        Input dataframe containing 'avg_gas_percentile' column
    col_layer2 : str
        Column name for the second layer
    col_layer3 : str
        Column name for the third layer
    metric_col : str, optional
        Column to sum for flow values. If None, uses count.
    min_flow : float, default=0
        Minimum flow value to display
    title : str
        Plot title
    
    Returns:
    --------
    Plotly figure
    """
    
    df = proc_df.copy()
    
    # Create layer labels with numeric percentile for sorting
    df['percentile_num'] = df['avg_gas_percentile']
    df['L1'] = 'Decile ' + df['avg_gas_percentile'].astype(str)
    df['L2'] = df[col_layer2].astype(str)
    df['L3'] = df[col_layer3].astype(str)
    
    # Calculate flows
    if metric_col:
        flow_12 = df.groupby(['L1', 'L2'])[metric_col].sum().reset_index(name='value')
        flow_23 = df.groupby(['L2', 'L3'])[metric_col].sum().reset_index(name='value')
    else:
        flow_12 = df.groupby(['L1', 'L2']).size().reset_index(name='value')
        flow_23 = df.groupby(['L2', 'L3']).size().reset_index(name='value')
    
    flow_12 = flow_12[flow_12['value'] >= min_flow]
    flow_23 = flow_23[flow_23['value'] >= min_flow]
    
    # Build node list - order L1 by numeric percentile
    l1_order = df[['L1', 'percentile_num']].drop_duplicates().sort_values('percentile_num')
    nodes_L1 = l1_order['L1'].tolist()
    
    nodes_L2 = sorted(df['L2'].unique())
    nodes_L3 = sorted(df['L3'].unique())
    
    all_nodes = nodes_L1 + nodes_L2 + nodes_L3
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    # Create explicit node positions
    n_L1 = len(nodes_L1)
    n_L2 = len(nodes_L2)
    n_L3 = len(nodes_L3)
    
    # X positions (0.01, 0.5, 0.99 for three layers)
    x_pos = [0.01] * n_L1 + [0.5] * n_L2 + [0.99] * n_L3
    
    # Y positions - evenly spaced for each layer
    y_L1 = np.linspace(0.01, 0.99, n_L1).tolist()
    y_L2 = np.linspace(0.01, 0.99, n_L2).tolist()
    y_L3 = np.linspace(0.01, 0.99, n_L3).tolist()
    y_pos = y_L1 + y_L2 + y_L3
    
    # Build links
    sources = []
    targets = []
    values = []
    
    for _, row in flow_12.iterrows():
        sources.append(node_dict[row['L1']])
        targets.append(node_dict[row['L2']])
        values.append(row['value'])
    
    for _, row in flow_23.iterrows():
        sources.append(node_dict[row['L2']])
        targets.append(node_dict[row['L3']])
        values.append(row['value'])
    
    # Create figure with explicit positioning
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            x=x_pos,
            y=y_pos
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(
        title=title,
        font=dict(size=12),
        height=500,
        width=1200
    )
    
    return fig