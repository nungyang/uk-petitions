from dash import Dash, dcc, Output, Input, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import geopandas as gpd

petitions = pd.read_csv('Data/petitions.csv')
df = petitions[petitions['petition_id'].isin([706513])].copy()
constituencies = gpd.read_file('Data/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC_-8097874740651686118.geojson')

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

mytitle = dcc.Markdown(children='')
mygraph = dcc.Graph(figure={}, config={'scrollZoom': False, 'doubleClick': False, 'displayModeBar': False})
dropdown = dcc.Dropdown(options = df['petition_title'].unique().tolist(),
                        value = 'Every school & college to be obliged to have an evacuation chair & training',
                        clearable = False,
                        placeholder = "Choose a petition..." )

app.layout = html.Div([
    banner,
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("Filters", className="mb-3"),
                dbc.Label("Select a Petition:"),
                dropdown
            ], width=2, style={
                'backgroundColor': '#f8f9fa', 
                'padding': '20px',
                'borderRight': '2px solid #dee2e6',
            }),

            dbc.Col([
                dbc.Row([mytitle]),  # Dropdown in top left
                dbc.Row([mygraph])    # Title in top right
            ], style={'flex': '1'})
        ], style={'minHeight': '100vh'})
    ], fluid=True)
])

# Callback allows components to interact
@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle, 'children'),
    Input(dropdown, 'value')
)

def update_graph(petition_name):  # function arguments come from the component property of the Input

    print(petition_name)
    print(type(petition_name))
    fig = px.choropleth(df,
                        locations='PCON24CD',
                        geojson=constituencies,
                        featureidkey="properties.PCON24CD",
                        color='signature_count',
                        color_continuous_scale="Viridis",
                        height=800,
                        range_color=[0, df['signature_count'].quantile(0.95)])
    fig.update_geos(
        visible=False,
        fitbounds="locations"
    )

    fig.update_layout(dragmode=False)

    return fig, '# '+petition_name  # returned objects are assigned to the component property of the Output

if __name__=='__main__':
    app.run(debug=False)