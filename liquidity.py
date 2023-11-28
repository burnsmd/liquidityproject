import pandas as pd
import numpy as np
from datetime import timedelta
import dash
from dash import dash_table
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objs as go



def load_and_prepare_data(csv_file):
    data = pd.read_csv(csv_file)
    data['BLOCK_DATE'] = pd.to_datetime(data['BLOCK_DATE'])
    data['TOTAL_VOLUME_USD'] = data['IN_USD_VALUE'].fillna(0) + data['OUT_USD_VALUE'].fillna(0)
    return data

# Load and prepare the data
osmosis_data = load_and_prepare_data('osmosis.csv')
latest_date = osmosis_data['BLOCK_DATE'].max()

# Define time periods
periods = {
    'last_7_days': latest_date - pd.Timedelta(days=7),
    'last_30_days': latest_date - pd.Timedelta(days=30),
    'last_60_days': latest_date - pd.Timedelta(days=60)
}

def aggregate_volume_by_pool_and_period(data, periods):
    volume_by_pool_and_period = pd.DataFrame()
    for period_name, start_date in periods.items():
        period_data = data[data['BLOCK_DATE'] > start_date]
        period_volumes = period_data.groupby('POOL_ID')['TOTAL_VOLUME_USD'].sum().reset_index()
        period_volumes.rename(columns={'TOTAL_VOLUME_USD': period_name}, inplace=True)
        volume_by_pool_and_period = volume_by_pool_and_period.merge(period_volumes, on='POOL_ID', how='outer') if not volume_by_pool_and_period.empty else period_volumes
    volume_by_pool_and_period.fillna(0, inplace=True)
    return volume_by_pool_and_period

def create_bar_chart(df, top_n, periods):
    top_pools = df.sort_values(by=list(periods.keys())[-1], ascending=False).head(top_n)
    top_pools['POOL_ID'] = top_pools['POOL_ID'].astype(str)
    return px.bar(top_pools, x='POOL_ID', y=list(periods.keys()), 
                  title="Top Pools by Volume Over Different Time Periods", 
                  labels={'value': 'Total Volume in USD', 'POOL_ID': 'Pool ID'},
                  barmode='group')

# Function to aggregate fees by CHAIN and PLATFORM for each period
def aggregate_fees_by_chain_platform():
    total_fees = pd.DataFrame()
    for period, start_date in periods.items():
        period_data = osmosis_data[osmosis_data['BLOCK_DATE'] > start_date]
        period_totals = period_data.groupby(['CHAIN', 'PLATFORM'])['FEE_USD_VALUE'].sum().reset_index()
        period_totals.rename(columns={'FEE_USD_VALUE': period}, inplace=True)
        if total_fees.empty:
            total_fees = period_totals
        else:
            total_fees = pd.merge(total_fees, period_totals, on=['CHAIN', 'PLATFORM'], how='outer')

    total_fees.fillna(0, inplace=True)
    return total_fees

def aggregate_volume_by_chain_platform(data, periods):
    total_volume = pd.DataFrame()
    for period, start_date in periods.items():
        period_data = data[data['BLOCK_DATE'] > start_date]
        period_totals = period_data.groupby(['CHAIN', 'PLATFORM'])['TOTAL_VOLUME_USD'].sum().reset_index()
        period_totals.rename(columns={'TOTAL_VOLUME_USD': period}, inplace=True)
        if total_volume.empty:
            total_volume = period_totals
        else:
            total_volume = pd.merge(total_volume, period_totals, on=['CHAIN', 'PLATFORM'], how='outer')

    total_volume.fillna(0, inplace=True)
    return total_volume

# Assuming osmosis_data and periods are already defined
volume_by_chain_platform = aggregate_volume_by_chain_platform(osmosis_data, periods)

# Assuming volume_by_chain_platform is already defined
fig_volume_by_chain_platform = go.Figure(data=[go.Table(
    header=dict(values=list(volume_by_chain_platform.columns),
                fill_color='paleturquoise',
                align='center'),
    cells=dict(values=[volume_by_chain_platform[k].tolist() for k in volume_by_chain_platform.columns],
               fill_color='lavender',
               align='center'))
])

# Update table layout for the volume data
fig_volume_by_chain_platform.update_layout(width=1200, height=250)


# Calculate total fees for each period grouped by CHAIN and PLATFORM
fees_by_chain_platform = aggregate_fees_by_chain_platform()

fig_fees_by_chain_platform = go.Figure(data=[go.Table(
    header=dict(values=list(fees_by_chain_platform.columns),
                fill_color='paleturquoise',
                align='center'),
    cells=dict(values=[fees_by_chain_platform[k].tolist() for k in fees_by_chain_platform.columns],
               fill_color='lavender',
               align='center'))
])

# Update table layout
fig_fees_by_chain_platform.update_layout(width=1200, height=250)  # Adjust width and height as needed



# Function to aggregate fees and calculate ranks
def aggregate_and_rank_fees(token_column):
    total_fees = pd.DataFrame()
    for period, start_date in periods.items():
        period_data = osmosis_data[osmosis_data['BLOCK_DATE'] > start_date]
        period_totals = period_data.groupby(token_column)['FEE_USD_VALUE'].sum().reset_index()
        period_totals.rename(columns={'FEE_USD_VALUE': period}, inplace=True)
        total_fees = total_fees.merge(period_totals, on=token_column, how='outer') if not total_fees.empty else period_totals

    total_fees.fillna(0, inplace=True)
    
    # Calculate ranks for each period
    for period in periods.keys():
        total_fees[f'Rank_{period}'] = total_fees[period].rank(ascending=False, method='min')

    # Create Rank Trend column
    total_fees['Rank_Trend'] = total_fees.apply(lambda x: ','.join([str(int(x[f'Rank_{period}'])) for period in periods.keys()]), axis=1)
    return total_fees

# Calculate fees and ranks for OUT_TOKEN and IN_TOKEN
fees_ranks_out_token = aggregate_and_rank_fees('OUT_TOKEN')
fees_ranks_in_token = aggregate_and_rank_fees('IN_TOKEN')

# Prepare data for Volume and Fee calculations
osmosis_data['TOTAL_VOLUME_USD'] = osmosis_data['IN_USD_VALUE'].fillna(0) + osmosis_data['OUT_USD_VALUE'].fillna(0)
daily_totals = pd.DataFrame()
volume_by_pool_and_period = pd.DataFrame()

# Aggregate data for each period
for period, start_date in periods.items():
    period_data = osmosis_data[osmosis_data['BLOCK_DATE'] > start_date]

    # Aggregate data for Fee
    period_fee_totals = period_data.groupby('BLOCK_DATE')['FEE_USD_VALUE'].sum().reset_index()
    period_fee_totals['Period'] = period
    daily_totals = pd.concat([daily_totals, period_fee_totals])

# Create the fee graph
fig_total_fees = px.scatter(daily_totals, x='BLOCK_DATE', y='FEE_USD_VALUE', color='Period', trendline='ols', 
                 title='Daily Total Fee in USD Over the Last 7, 30, and 60 Days')


# Select top 10 tokens for the graphs
top_10_out_token = fees_ranks_out_token.nlargest(10, 'last_60_days')
top_10_in_token = fees_ranks_in_token.nlargest(10, 'last_60_days')

# Select top 25 tokens for the tables
top_25_out_token = fees_ranks_out_token.nlargest(25, 'last_60_days')
top_25_in_token = fees_ranks_in_token.nlargest(25, 'last_60_days')

# Create bar charts for OUT_TOKEN and IN_TOKEN fees (top 10)
fig_in_token = px.bar(top_10_in_token, x='IN_TOKEN', y=list(periods.keys()),
                      title='Top 10 IN_TOKEN Fees Over Different Time Periods',
                      barmode='group')

fig_out_token = px.bar(top_10_out_token, x='OUT_TOKEN', y=list(periods.keys()),
                       title='Top 10 OUT_TOKEN Fees Over Different Time Periods',
                       barmode='group')

# Aggregate volume by pool and period
volume_by_pool_and_period = aggregate_volume_by_pool_and_period(osmosis_data, periods)

# After aggregating data for each period and merging
volume_by_pool_and_period.fillna(0, inplace=True)

# Ensure the columns used for ranking are numeric
for period in periods.keys():
    volume_by_pool_and_period[period] = pd.to_numeric(volume_by_pool_and_period[period], errors='coerce')

# Calculate ranks for each period
for period in periods.keys():
    volume_by_pool_and_period[f'Rank_{period}'] = volume_by_pool_and_period[period].rank(ascending=False, method='min')

# Create the 'Rank Trend' column
volume_by_pool_and_period['Rank_Trend'] = volume_by_pool_and_period.apply(
    lambda row: ','.join([str(int(row[f'Rank_{period}'])) for period in periods.keys()]), axis=1
)

# Create the bar chart
fig_volume_by_pool = create_bar_chart(volume_by_pool_and_period, 25, periods)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Volume', children=[
            html.Div(dcc.Graph(id='volume_by_chain_platform_table', figure=fig_volume_by_chain_platform),
            style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
            html.H2("Top 25 Pools by Volume Over Different Time Periods"),
            dcc.Graph(id='graph_volume_by_pool', figure=fig_volume_by_pool),
            
            html.H2("Volume Table"),
            dash_table.DataTable(
                id='volume_table',
            # Specify the columns to be displayed in the DataTable
                columns=[{"name": i, "id": i} for i in ['POOL_ID', 'Rank_Trend'] + ['last_7_days', 'last_30_days', 'last_60_days']],
                data=volume_by_pool_and_period.to_dict('records'),
                sort_action="native", filter_action="native",
                style_data_conditional=[
                    {'if': {'column_id': col}, 'textAlign': 'right', 'type': 'numeric', 'format': {'specifier': '$,.0f'}}
                    for col in ['last_7_days', 'last_30_days', 'last_60_days']
        ]
    )
        ]),
        dcc.Tab(label='Asset', children=[
            html.Div([
                html.H3('Asset Placeholder Content')
                # Placeholder content for Asset
            ])
        ]),
        dcc.Tab(label='Fees', children=[
            html.Div([
                html.Div(dcc.Graph(id='fees_by_chain_platform_table', figure=fig_fees_by_chain_platform),
            style={'width': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
    
        html.Div([
                html.H2("Fee Graph"),
                dcc.Graph(id='fee_graph', figure=fig_total_fees)
            ]),
    
            html.Div([
                html.H2("Top 10 IN_TOKEN Fees"),
                dcc.Graph(id='graph_in_token', figure=fig_in_token),
                html.H2("Top 25 IN_TOKEN Fees Table"),
                dash_table.DataTable(
                    id='table_in_token',
                    columns=[{"name": i, "id": i} for i in ['IN_TOKEN', 'Rank_Trend'] + list(periods.keys())],
                    data=top_25_in_token.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    sort_action='native'
                )
            ], style={'display': 'inline-block', 'width': '48%', 'margin-right': '2%'}),

            html.Div([
                html.H2("Top 10 OUT_TOKEN Fees"),
                dcc.Graph(id='graph_out_token', figure=fig_out_token),
                html.H2("Top 25 OUT_TOKEN Fees Table"),
                dash_table.DataTable(
                    id='table_out_token',
                    columns=[{"name": i, "id": i} for i in ['OUT_TOKEN', 'Rank_Trend'] + list(periods.keys())],
                    data=top_25_out_token.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    sort_action='native'
                )
            ], style={'display': 'inline-block', 'width': '48%'})# ... Fees layout ...
                        # dcc.Graph(id='fees_by_chain_platform_table', figure=fig_fees_by_chain_platform),
                        # ... Rest of Fees layout ...
                    ])
                ]),
                dcc.Tab(label='TVL', children=[
                    html.Div([
                        html.H3('TVL Placeholder Content')
                        # Placeholder content for TVL
                    ])
                ])
            ])
])

if __name__ == '__main__':
    app.run_server(debug=True)