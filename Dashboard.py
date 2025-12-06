from dash import Dash, dcc, Output, Input, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import geopandas as gpd

petitions = pd.read_csv('Data/petitions.csv')
petition_options = petitions[['petition_id', 'petition_title']].drop_duplicates()

constituencies = gpd.read_file('Data/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC_-8097874740651686118.geojson')


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
        'modeBarButtonsToRemove': ['select2d', 'lasso2d']
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
                dbc.Row([mytitle], className="g-0", style={'marginBottom': '0'}),                dbc.Row([
                    dbc.Col([mygraph], width=7),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Signatures", className="text-muted mb-2"),
                                html.H3("123,456", className="mb-0")
                            ])
                        ], className="mb-3 shadow-sm", color="primary", outline=True),
                        
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Constituencies", className="text-muted mb-2"),
                                html.H3("650", className="mb-0")
                            ])
                        ], className="mb-3 shadow-sm", color="info", outline=True),
                        
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Average per Area", className="text-muted mb-2"),
                                html.H3("190", className="mb-0")
                            ])
                        ], className="shadow-sm", color="success", outline=True)
                    ], width=5)
                ])
            ], style={'flex': '1'})
        ], style={'minHeight': '80vh'})
    ], fluid=True)
])


# Callback allows components to interact
@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle, 'children'),
    Input('petition-dropdown', 'value')
)

def update_graph(petition_id):  # function arguments come from the component property of the Input
    
    df = petitions[petitions['petition_id'] == petition_id].copy()

    # Get the petition title for display
    petition_title = df['petition_title'].iloc[0] if len(df) > 0 else "No data"

    print(petition_title)
    print(type(petition_title))

    fig = px.choropleth(df,
                        locations='PCON24CD',
                        geojson=constituencies,
                        featureidkey="properties.PCON24CD",
                        color='signature_count',
                        color_continuous_scale="Viridis",
                        range_color=[0, df['signature_count'].quantile(0.95)],
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
        lataxis_range=[49, 61],
        lonaxis_range=[-9, 3]
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

    return fig, '# ' + petition_title  # returned objects are assigned to the component property of the Output

if __name__=='__main__':
    app.run(debug=False)