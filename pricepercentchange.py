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

#ensure Quantity is a float and replace NaN with 0
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

#define skews to filter
skew_mapping = {
    (1.0, 10.0, 80.0): '1" by 10\' PVC Schedule 80',
    (2.0, 10.0, 40.0): '2" by 10\' PVC Schedule 40',
    (4.0, 10.0, 40.0): '4" by 10\' PVC Schedule 40',
    (4.0, 10.0, 80.0): '4" by 10\' PVC Schedule 80',
    (6.0, 10.0, 40.0): '6" by 10\' PVC Schedule 40',
    (8.0, 10.0, 40.0): '8" by 10\' PVC Schedule 40'   
}

#filter rows that have non-null values for Inches, Feet, Schedule
df_filtered = df[df.apply(lambda row: (row['Inches'], row['Feet'], row['Schedule']) in skew_mapping, axis=1)].copy()

#create a new column "Skew_Description"
df_filtered['Skew_Description'] = df_filtered.apply(lambda row: skew_mapping[(row['Inches'], row['Feet'], row['Schedule'])], axis=1)

low_threshold = df_filtered['Quantity'].quantile(0.50)
high_threshold = df_filtered['Quantity'].quantile(0.50)

#categorize quantities
def categorize_quantity(quantity):
    if quantity < low_threshold:
        return 'Low'
    elif quantity <= high_threshold:
        return 'Medium'
    else:
        return 'High'

#apply the function to categorize quantities
df_filtered['Quantity Category'] = df_filtered['Quantity'].apply(categorize_quantity)

#filter the DataFrame to include only high quantities
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

#calculate percent change for each skew
def calculate_percent_change(df, skew_description):
    df_skew = df[df['Skew_Description'] == skew_description].copy()
    if not df_skew.empty:
        df_skew = df_skew.sort_values(by='Order Date')
        start_value = df_skew['Unit Price'].iloc[0]
        df_skew['Percent Change'] = (df_skew['Unit Price'] - start_value) / start_value * 100
        df_skew['Starting Unit Price'] = start_value  # Keep track of the starting price
    return df_skew

#apply percent change calculation to all skews
df_percent_change = pd.concat([calculate_percent_change(df_high_quantities, skew) for skew in skew_mapping.values()])

#initialize the Dash app
app = Dash(__name__)
server = app.server

#define app layout
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
    dcc.Graph(id='percent-change-graph')
])

@app.callback(
    Output('percent-change-graph', 'figure'),
    [Input('supplier-dropdown', 'value')]
)
def update_graph(selected_supplier):
    if selected_supplier == 'All':
        filtered_data = df_percent_change
    elif selected_supplier == 'Other':
        filtered_data = df_percent_change[~df_percent_change['SUPPLIER'].isin(['Border States', 'Gexpro'])]
    else:
        filtered_data = df_percent_change[df_percent_change['SUPPLIER'] == selected_supplier]

    #recalculate regression lines for the filtered data
    regression_lines = []
    for skew in skew_mapping.values():
        skew_data = filtered_data[filtered_data['Skew_Description'] == skew]
        if len(skew_data) > 1:  # Ensure there is more than one data point
            X = skew_data['Order Date Num'].values
            y = skew_data['Percent Change'].values
            if len(np.unique(X)) > 1:  # Ensure there is variability in the data
                try:
                    slope, intercept = np.polyfit(X, y, 1)
                    starting_price = skew_data['Starting Unit Price'].iloc[0]
                    regression_lines.append((skew, slope, intercept, starting_price))
                except np.linalg.LinAlgError:
                    print(f"LinAlgError for skew {skew}: Insufficient variability in data")
            else:
                print(f"Insufficient variability in data for skew {skew}")
        else:
            print(f"Not enough data points for skew {skew}")


    fig = go.Figure()

    #add scatter plots for each skew with percent change
    for skew in skew_mapping.values():
        skew_data = filtered_data[filtered_data['Skew_Description'] == skew]
        fig.add_trace(go.Scatter(
            x=skew_data['Order Date'],
            y=skew_data['Percent Change'],
            mode='markers',
            name=skew,
            text=skew_data['SUPPLIER'],
            customdata=np.stack((skew_data['Unit Price'], skew_data['Quantity']), axis=-1),
            marker=dict(color=color_palette[skew]),
            hovertemplate='<b>Supplier</b>: %{text}<br><b>Unit Price</b>: $%{customdata[0]:.2f}<br><b>Percent Change</b>: %{y:.2f}%<br><b>Quantity</b>: %{customdata[1]:,}<br><b>Date</b>: %{x}'
        ))

    #add regression lines for percent change for each skew
    for skew, slope, intercept, starting_price in regression_lines:
        skew_data = filtered_data[filtered_data['Skew_Description'] == skew]
        regression_x = pd.date_range(start=skew_data['Order Date'].min(), end=skew_data['Order Date'].max())
        regression_y_percent_change = slope * date2num(regression_x) + intercept
        regression_y_price = starting_price * (1 + regression_y_percent_change / 100)
        fig.add_trace(go.Scatter(
            x=regression_x,
            y=regression_y_percent_change,
            mode='lines',
            name=f'{skew} Trend ({selected_supplier})' if selected_supplier != 'All' else f'{skew} Trend',
            line=dict(dash='solid', color=color_palette[skew]),
            customdata=regression_y_price,
            hovertemplate='<b>Percent Change</b>: %{y:.2f}%<br><b>Unit Price</b>: $%{customdata:.2f}'
        ))


    fig.update_layout(
        title='Percent Change in Unit Price Over Time for PVC Types',
        xaxis_title='Order Date',
        yaxis_title='Percent Change (%)',
        yaxis_tickformat='%d',
        legend_title_text='PVC Type',
        hovermode='closest'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
