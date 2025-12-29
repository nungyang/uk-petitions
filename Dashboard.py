####################
#### Setting up ####
####################

## Loading libraries
import time
import boto3
from dash import Dash, dcc, Output, Input, html, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import geopandas as gpd
from functools import lru_cache
from io import BytesIO
import os

## Setting up connection to AW3
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')  # Railway environment variable
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')  # Railway environment variable
bucket = 'uk-parliament-petitions-bucket'

s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key,
                         aws_secret_access_key=aws_secret_key)

## Creating functions to load files
def load_csv(filename):
    # Fetch the file object from S3
    s3_object = s3_client.get_object(Bucket=bucket, Key=filename)
    
    # Load the CSV file into pandas DataFrame
    file_stream = s3_object['Body']
    df = pd.read_csv(file_stream)
    return df

def load_geojson(filename):
    # Fetch the file object from S3
    s3_object = s3_client.get_object(Bucket=bucket, Key=filename)
    
    # Load the GeoJSON file into GeoDataFrame
    file_stream = s3_object['Body']
    gdf = gpd.read_file(file_stream)
    return gdf



##########################
#### Loading datasets ####
##########################

start = time.time()  # Start timer

@lru_cache(maxsize=1)
def get_constituency_geojson():
    constituencies = load_geojson('static data/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC_-8097874740651686118.geojson')
    constituencies = constituencies[['PCON24CD', 'geometry']]
    constituencies['geometry'] = constituencies['geometry'].simplify(0.005) 
    return constituencies

# Load the constituencies data
constituencies = get_constituency_geojson()

# Loading petition count data for first dashboard
@lru_cache(maxsize=1)
def get_petitions_data():
    petitions_list = load_csv('dynamic data/all_petitions_list.csv')
    petitions_count = load_csv('dynamic data/all_petitions_counts.csv')
    petitions_df = petitions_list.merge(petitions_count,
                                        on = 'petition_id',
                                        how = 'left')
    return petitions_df

petitions_df = get_petitions_data()

# Caching petition data
@lru_cache(maxsize=128)
def get_petition_data(petition_id):
    start = time.time()  # Start timer
    df = petitions_df[petitions_df['petition_id'] == petition_id]
    print(f"Time to retrieve petition data from cache: {time.time() - start} seconds")
    return tuple(df.itertuples(index=False))  # Return as tuple for caching

print(petitions_df.head())

######################
#### Creating app ####
######################

app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
server = app.server

# Custom CSS for tabs
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .tab {
                padding: 12px 24px !important;
                border: none !important;
                background-color: #e9ecef !important;
            }
            .tab--selected {
                background-color: white !important;
            }
            .tabs {
                border-bottom: 1px solid #dee2e6 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create banner
banner = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("UK Petitions Dashboard", className="text-white mb-0")),
        ], align="center", className="g-0"),
    ], fluid=True),
    color="primary",
    dark=True,
)

mytitle = dcc.Markdown(children='', style={'margin': '10px 0 0 0'})

mygraph = dcc.Graph(
    figure={}, 
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    },
    style={'height': '85vh'}
)

#### Setting up dropdowns ####
# Getting list of unique petitions
petition_options = petitions_df[['petition_id', 'petition_title']].drop_duplicates()
petition_quantiles = petitions_df.groupby('petition_id')['signature_count'].quantile(0.95).to_dict()

petition_dropdown = dbc.RadioItems(id='petition-dropdown',
                          options=[{'label': row['petition_title'], 'value': row['petition_id']}
                                   for _, row in petition_options.iterrows()],
                            value=petition_options.iloc[0]['petition_id'],
                            style={'maxHeight': '80vh', 'overflowY': 'auto', 'padding': '10px'}
)

# Getting list of unique constituencies
constituency_options = petitions_df[['PCON24CD', 'constituency_name']].drop_duplicates()

constituency_dropdown = dcc.Dropdown(
    id='analytics-petition-dropdown',
    options=[{'label': row['constituency_name'], 'value': row['PCON24CD']}
             for _, row in constituency_options.iterrows()],
    value=constituency_options.iloc[0]['PCON24CD'],
    clearable=False
)

# App layout
app.layout = html.Div([
    banner,
    dbc.Container([
        dcc.Tabs(id='main-tabs', value='tab-1', children=[
            dcc.Tab(label='Constituency Overview', value='tab-1', children=[
                html.Div([
                    # Filter Section
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select constituency:", className="fw-bold mb-2"),
                                    constituency_dropdown
                                ], md=4)
                            ])
                        ])
                    ], className="mb-4 shadow-sm"),
                    
                    # KPI Cards Row
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Mean Signatures", className="text-muted mb-2"),
                                    html.H3(id='kpi-mean', className="mb-0 text-primary"),
                                    html.Small(id='kpi-mean-detail', className="text-muted")
                                ])
                            ], className="shadow-sm h-100")
                        ], md=3),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Median Signatures", className="text-muted mb-2"),
                                    html.H3(id='kpi-median', className="mb-0 text-success"),
                                    html.Small(id='kpi-median-detail', className="text-muted")
                                ])
                            ], className="shadow-sm h-100")
                        ], md=3),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Highest Constituency", className="text-muted mb-2"),
                                    html.H3(id='kpi-max', className="mb-0 text-danger"),
                                    html.Small(id='kpi-max-name', className="text-muted")
                                ])
                            ], className="shadow-sm h-100")
                        ], md=3),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Total Constituencies", className="text-muted mb-2"),
                                    html.H3(id='kpi-count', className="mb-0 text-info"),
                                    html.Small("Constituencies reporting", className="text-muted")
                                ])
                            ], className="shadow-sm h-100")
                        ], md=3)
                    ], className="mb-4"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Top 10 Petitions", className="mb-0")),
                                dbc.CardBody([
                                    html.Div(id='top-10-table')
                                ])
                            ], className="shadow-sm")
                        ])
                    ])
                ], style={'padding': '20px'})
            ]),

            dcc.Tab(label='Map', value='tab-2', children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Select a Petition", className="mb-3"),
                        petition_dropdown
                    ], style={
                        'backgroundColor': '#f8f9fa', 
                        'padding': '20px',
                        'borderRight': '2px solid #dee2e6',
                        'minWidth': '400px',
                        'maxWidth': '400px'
                    }),

                    dbc.Col([
                        dbc.Row([mytitle], className="g-0", style={'marginBottom': '0'}),
                        dbc.Row([
                            dbc.Col([
                                dcc.Loading(
                                    id="loading",
                                    type="circle",
                                    children=[mygraph]
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Total Signatures", className="text-white mb-2"),
                                        html.H3(id='total-sigs', className="mb-0 text-white")
                                    ])
                                ], className="mb-3 shadow", color="primary", inverse=True, style={'borderRadius': '15px'}),
                                
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Scheduled debate date", className="text-white mb-2"),
                                        html.H3(id='sch-debate-date', className="mb-0 text-white")
                                    ])
                                ], className="mb-3 shadow", color="info", inverse=True, style={'borderRadius': '15px'}),        

                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Constituency with most signatures", className="text-white mb-2"),
                                        html.H3(id='highest-count-con', className="mb-0 text-white")
                                    ])
                                ], className="mb-3 shadow", color="success", inverse=True, style={'borderRadius': '15px'}),       
                            ], width=3)
                        ])
                    ], style={'flex': '1'})
                ], style={'minHeight': '80vh'})
            ])
        ])
    ], fluid=True)
])

# Callback allows components to interact
# Callback for Analytics KPI Cards
@app.callback(
    Output('kpi-mean', 'children'),
    Output('kpi-mean-detail', 'children'),
    Output('kpi-median', 'children'),
    Output('kpi-median-detail', 'children'),
    Output('kpi-max', 'children'),
    Output('kpi-max-name', 'children'),
    Output('kpi-count', 'children'),
    Output('top-10-table', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_kpi_cards(PCON24CD):
    df = petitions_df[petitions_df['PCON24CD'] == PCON24CD].copy()
    
    # Get top 10
    top_10 = df.nlargest(10, 'signature_count')[['petition_id', 'signature_count']].reset_index(drop=True)
    
    # Format signatures with commas
    top_10['signatures_formatted'] = top_10['signature_count'].apply(lambda x: f"{int(x):,}")

    mean_sigs = df['signature_count'].mean()
    median_sigs = df['signature_count'].median()
    max_sigs = df['signature_count'].max()
    max_constituency = df.loc[df['signature_count'].idxmax(), 'constituency_name']
    count = len(df)
    
    # Create DataTable
    table = dash_table.DataTable(
        data=top_10[['petition_id', 'signatures_formatted']].to_dict('records'),
        columns=[
            {'name': 'Petition name', 'id': 'petition_id'},
            {'name': 'Signatures', 'id': 'signatures_formatted'}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Arial, sans-serif'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #dee2e6'
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '1.5'
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'signatures_formatted'},
                'textAlign': 'right',
                'fontWeight': '500'
            }
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            }
        ]
    )
    
    return (
        f"{mean_sigs:,.0f}",
        "per constituency",
        f"{median_sigs:,.0f}",
        "per constituency",
        f"{max_sigs:,}",
        max_constituency,
        f"{count:,}",
        table
    )



#### Setting up app call back, etc for dashboard 2 ####

@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle, 'children'),
    Output('total-sigs', 'children'),
    Output('sch-debate-date', 'children'),
    Output('highest-count-con', 'children'),
    Input('petition-dropdown', 'value')
)
def update_graph(petition_id):  # function arguments come from the component property of the Input
    start = time.time()  # Start timer for callback execution
    
    # Retrieve the cached petition data
    cached_data = get_petition_data(petition_id)
    df = pd.DataFrame(cached_data, columns=petitions_df.columns)
    
    print(f"Time to retrieve and process petition data: {time.time() - start} seconds")  # Time to load data into DataFrame
    
    # Extract petition information
    petition_title = df['petition_title'].iloc[0] if len(df) > 0 else "No data"
    total_signatures = df['signature_count'].sum()
    
    # Get scheduled debate date
    sch_debate_date = df['scheduled_debate_date'].iloc[0] if 'scheduled_debate_date' in df.columns else None
    debate_date_str = str(sch_debate_date) if pd.notna(sch_debate_date) else "Not scheduled"
    
    # Find the constituency with the most signatures
    max_row = df.loc[df['signature_count'].idxmax()]
    highest_count_con = max_row['constituency_name']
    highest_count = max_row['signature_count']
    
    # Get the maximum color scale value
    max_color = petition_quantiles.get(petition_id, df['signature_count'].max())
    
    # Create choropleth figure
    start_plot = time.time()  # Start timer for plotting
    fig = px.choropleth(df,
                        locations='PCON24CD',
                        geojson=constituencies,
                        featureidkey="properties.PCON24CD",
                        color='signature_count',
                        color_continuous_scale="Viridis",
                        range_color=[0, max_color],
                        labels={'signature_count': 'Number of signatures',
                                'PCON24CD': 'Constituency Code',
                                'constituency_name': 'Constituency'},  
                        hover_data={'PCON24CD': False,
                                    'constituency_name': True,     
                                    'signature_count': True}) 
    
    fig.update_geos(
        visible=False,
        projection_scale=0.8,
        center=dict(lat=54.5, lon=-3),
        lataxis_range=[48, 60],
        lonaxis_range=[-10, 4]
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        uirevision='constant',
        coloraxis_colorbar=dict(
            title="Signatures",
            thickness=15,
            len=0.7, 
            x=1.0, 
            xanchor="left",
            y=0.5,  
            yanchor="middle")
    )
    
    print(f"Time to plot the choropleth: {time.time() - start_plot} seconds")  # Time for plotting

    return fig, '# ' + petition_title, f"{total_signatures:,}", debate_date_str, f"{highest_count_con} ({highest_count:,})"


if __name__=='__main__':
    app.run(debug=False)
