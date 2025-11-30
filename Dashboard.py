from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import geopandas as gpd

petitions = pd.read_csv('Data/petitions.csv')
df = petitions[petitions['petition_id'].isin([706513])].copy()
constituencies = gpd.read_file('Data/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC_-8097874740651686118.geojson')

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
mytitle = dcc.Markdown(children='')
mygraph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options = df['petition_title'].unique().tolist(),
                        value = 'Every school & college to be obliged to have an evacuation chair & training',
                        clearable = False)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([dropdown], width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mytitle], width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph], width=12)
    ]),

], fluid=True)

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
                        height=600,
                        title="Petitions by Constituency")
    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

    return fig, '# '+petition_name  # returned objects are assigned to the component property of the Output

if __name__=='__main__':
    app.run(debug=True, port=8054)