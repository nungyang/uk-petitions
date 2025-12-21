from dash import Dash, dcc, Output, Input, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import geopandas as gpd
from functools import lru_cache


petitions = pd.read_csv('Data/petitions.csv')
petition_options = petitions[['petition_id', 'petition_title']].drop_duplicates()

constituencies = gpd.read_file('Data/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC_-8097874740651686118.geojson')
constituencies.geometry = constituencies.geometry.simplify(tolerance=0.05, preserve_topology=True)
constituencies = constituencies[['PCON24CD', 'geometry']]  # Only keep needed columns
petition_quantiles = petitions.groupby('petition_id')['signature_count'].quantile(0.95).to_dict()


## Creating app
app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
server = app.server

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

dropdown = dbc.RadioItems(id='petition-dropdown',
                          options=[{'label': row['petition_title'], 'value': row['petition_id']}
                                   for _, row in petition_options.iterrows()],
                            value=petition_options.iloc[0]['petition_id'],
                            style={'maxHeight': '80vh', 'overflowY': 'auto', 'padding': '10px'}
)

app.layout = html.Div([
    banner,
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("Select a Petition", className="mb-3"),
                dropdown
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
    ], fluid=True)
])


# Callback allows components to interact
@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle, 'children'),
    Output('total-sigs', 'children'),
    Output('sch-debate-date', 'children'),
    Output('highest-count-con', 'children'),
    Input('petition-dropdown', 'value')
)

def update_graph(petition_id):  # function arguments come from the component property of the Input
    
    df = petitions[petitions['petition_id'] == petition_id].copy()

    # Get the petition title for display
    petition_title = df['petition_title'].iloc[0] if len(df) > 0 else "No data"

    # Getting summary stats
    total_signatures = df['signature_count'].sum()

    sch_debate_date = df['scheduled_debate_date'].iloc[0] if 'scheduled_debate_date' in df.columns else None
    debate_date_str = str(sch_debate_date) if pd.notna(sch_debate_date) else "Not scheduled"

    max_row = df.loc[df['signature_count'].idxmax()]
    highest_count_con = max_row['constituency_name']
    highest_count = max_row['signature_count']
    max_color = petition_quantiles.get(petition_id, df['signature_count'].max())


    print(petition_title)
    print(type(petition_title))

    fig = px.choropleth(df,
                        locations='PCON24CD',
                        geojson=constituencies,
                        featureidkey="properties.PCON24CD",
                        color='signature_count',
                        color_continuous_scale="Viridis",
                        range_color=[0, max_color],
                        labels={'signature_count': 'Number of signatures',
                                'PCON24CD': 'Constituency Code',
                                'constituency_name': 'Constituency'},  # Add custom labels
                        hover_data={'PCON24CD': False,
                                    'constituency_name': True,     # Show constituency name
                                    'signature_count': True}) # And signature count
    
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
        len=0.7,  # 70% of plot height
        x=1.0,  # Position on right
        xanchor="left",
        y=0.5,  # Center vertically
        yanchor="middle")
    )

    return fig, '# ' + petition_title, f"{total_signatures:,}", debate_date_str, f"{highest_count_con} ({highest_count:,})"

if __name__=='__main__':
    app.run(debug=False)