import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib.dates import date2num
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import plotly.graph_objects as go

file_path = 'PVC Clean2.csv'
df = pd.read_csv(file_path)

#strip any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

#convert "Order Date" to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', format='%m/%d/%Y')

#remove $ signs and commas from "Unit Price"
df['Unit Price'] = df['Unit Price'].replace('[\$,]', '', regex=True).replace(',', '')

#convert to float
df['Unit Price'] = pd.to_numeric(df['Unit Price'], errors='coerce')

#convert "Quantity" to int
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

#define skews to filter
skew_mapping = {
    (4.0, 10.0, 40.0): '4" by 10\' PVC Schedule 40',
    (8.0, 10.0, 40.0): '8" by 10\' PVC Schedule 40',
    (1.0, 10.0, 80.0): '1" by 10\' PVC Schedule 80',
    (2.0, 10.0, 40.0): '2" by 10\' PVC Schedule 40',
    (4.0, 10.0, 80.0): '4" by 10\' PVC Schedule 80',
    (6.0, 10.0, 40.0): '6" by 10\' PVC Schedule 40'
}

#filter rows that have non-null values for Inches, Feet, Schedule
df_filtered = df[df.apply(lambda row: (row['Inches'], row['Feet'], row['Schedule']) in skew_mapping, axis=1)].copy()

#create new column "Skew_Description"
df_filtered['Skew_Description'] = df_filtered.apply(lambda row: skew_mapping[(row['Inches'], row['Feet'], row['Schedule'])], axis=1)

#calculate low & high quantities
low_threshold = df_filtered['Quantity'].quantile(0.5)

#categorize quantities
def categorize_quantity(quantity):
    if quantity < low_threshold:
        return 'Low'
    else:
        return 'High'

#apply function
df_filtered['Quantity Category'] = df_filtered['Quantity'].apply(categorize_quantity)

#filter the DataFrame to include only high quantity
df_high_quantities = df_filtered[df_filtered['Quantity Category'] == 'High'].copy()

#remove specific outliers
outlier_condition = (
    (df_high_quantities['Skew_Description'] == '4" by 10\' PVC Schedule 40') &
    (df_high_quantities['Unit Price'] >= 80.00) &
    (df_high_quantities['Unit Price'] <= 100.00) &
    (df_high_quantities['Order Date'] >= pd.to_datetime('2021-04-01')) &
    (df_high_quantities['Order Date'] <= pd.to_datetime('2022-07-30'))
)

outlier_condition_1 = (
    (df_high_quantities['Skew_Description'] == '4" by 10\' PVC Schedule 80') &
    (df_high_quantities['Unit Price'] >= 50.00) &
    (df_high_quantities['Unit Price'] <= 60.00) &
    (df_high_quantities['Order Date'] >= pd.to_datetime('2021-02-01')) &
    (df_high_quantities['Order Date'] <= pd.to_datetime('2022-07-30'))
)

#combine outlier conditions
combined_outlier_condition = outlier_condition | outlier_condition_1

#remove outliers
df_high_quantities = df_high_quantities[~combined_outlier_condition]

#drop rows with NaN "Order Date" values
df_high_quantities = df_high_quantities.dropna(subset=['Order Date'])

#convert "Order Date" to numerical format
df_high_quantities['Order Date Num'] = date2num(df_high_quantities['Order Date'])

#define legend order and color palette
legend_order = [
    '1" by 10\' PVC Schedule 80',
    '2" by 10\' PVC Schedule 40',
    '4" by 10\' PVC Schedule 40',
    '4" by 10\' PVC Schedule 80',
    '6" by 10\' PVC Schedule 40',
    '8" by 10\' PVC Schedule 40'
]

color_palette = {
    '2" by 10\' PVC Schedule 40': '#008000',    
    '4" by 10\' PVC Schedule 40': '#0000ff',    
    '4" by 10\' PVC Schedule 80': '#ff0000',    
    '1" by 10\' PVC Schedule 80': '#ffa500',    
    '8" by 10\' PVC Schedule 40': '#ee82ee',
    '6" by 10\' PVC Schedule 40': '#27272B'
}

#initialize Dash app
app = Dash(__name__)
server = app.server

#define layout of app
app.layout = html.Div([
    dcc.Dropdown(
        id='supplier-dropdown',
        options=[
            {'label': 'All Suppliers', 'value': 'All'},
            {'label': 'Border States', 'value': 'Border States'},
            {'label': 'Gexpro', 'value': 'Gexpro'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='All'  # Default value
    ),
    dcc.Graph(id='price-trends-graph')
])

@app.callback(
    Output('price-trends-graph', 'figure'),
    [Input('supplier-dropdown', 'value')]
)
def update_graph(selected_supplier):
    if selected_supplier == 'All':
        filtered_data = df_high_quantities
    elif selected_supplier == 'Other':
        filtered_data = df_high_quantities[~df_high_quantities['SUPPLIER'].isin(['Border States', 'Gexpro'])]
    else:
        filtered_data = df_high_quantities[df_high_quantities['SUPPLIER'] == selected_supplier]

    #calculate regression lines for filtered data
    regression_lines = []
    for skew in legend_order:
        skew_data = filtered_data[filtered_data['Skew_Description'] == skew]
        if len(skew_data) > 1:  # Ensure there is more than one data point
            X = skew_data['Order Date Num'].values
            y = skew_data['Unit Price'].values
            if len(np.unique(X)) > 1:  # Ensure there is variability in the data
                try:
                    slope, intercept = np.polyfit(X, y, 1)
                    regression_lines.append((skew, slope, intercept))
                except np.linalg.LinAlgError:
                    print(f"LinAlgError for skew {skew}: Insufficient variability in data")
            else:
                print(f"Insufficient variability in data for skew {skew}")
        else:
            print(f"Not enough data points for skew {skew}")

    fig = go.Figure()

    #add scatter plots for each skew
    for skew in legend_order:
        skew_data = filtered_data[filtered_data['Skew_Description'] == skew]
        fig.add_trace(go.Scatter(
            x=skew_data['Order Date'],
            y=skew_data['Unit Price'],
            mode='markers',
            name=skew,
            text=skew_data['SUPPLIER'],
            marker=dict(color=color_palette[skew]),
            hovertemplate='<b>Supplier</b>: %{text}<br><b>Price</b>: $%{y:.2f}<br><b>Date</b>: %{x}'
        ))

    #add regression lines for each skew
    for skew, slope, intercept in regression_lines:
        skew_data = filtered_data[filtered_data['Skew_Description'] == skew]
        regression_x = pd.date_range(start=skew_data['Order Date'].min(), end=skew_data['Order Date'].max())
        regression_y = slope * date2num(regression_x) + intercept
        fig.add_trace(go.Scatter(
            x=regression_x,
            y=regression_y,
            mode='lines',
            name=f'{skew} Trend ({selected_supplier})' if selected_supplier != 'All' else f'{skew} Trend',
            line=dict(dash='solid', color=color_palette[skew])
        ))

    fig.update_layout(
        title='Unit Price Trends Over Time for PVC Types',
        xaxis_title='Order Date',
        yaxis_title='Unit Price ($)',
        yaxis_tickformat='$,.2f',
        legend_title_text='PVC Type',
        hovermode='closest'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
