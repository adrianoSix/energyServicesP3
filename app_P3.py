# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 21:06:17 2021

@author: Adriano Caprara, 98007, IST Lisboa
Energy services project 3: Solar radiation regression model for the Canary
    Island of La Palma

Input files:
LaPalmaElMix.csv
ElMixEspanaIEA.csv

allstations.csv

error_metrics.csv
df_XGBR.csv

allmodel_stations.csv


"""

#%% 1. IMPORT CLEAN AND PROCESSED DATA
import pandas as pd
import geopandas
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go


lpmix = pd.read_csv("LaPalmaElMix.csv", delimiter = ";", index_col = 0)
lpmix_percentage = lpmix.div(lpmix.sum(axis = 1), axis = 0) * 100

esmix_raw = pd.read_csv("ElMixEspanaIEA.csv", skiprows = 4, index_col = 0)
esmix_raw = esmix_raw.drop("Units", axis = 1)
esmix = esmix_raw.copy()
esmix = esmix_raw.fillna(0)
esmix_percentage = esmix.div(esmix.sum(axis = 1), axis = 0) * 100

station_loc = geopandas.read_file("db3089b76ba846baba34b8f2d24506d5.gdb-point.shp")

#mapper to connect the object-ID to the names of the sensors
mapper = {"mont-tricias":17, "bodeganorte":19, "centrovis-cab":35, 
          "cabildo":36, "mont-samagallo":23, 
          "lossauces":35, "balsa-laguna":24, "sanantonio":16,
          "loscanarios":2, "fagundo":10, "labombilla":14, "eltime":9, 
          "centrovis-aemet":13, "aeropuer-aemet":12, "balsa-adeyaham":11,
          "salinas":15, "helipuerto":3, "mont-cumbre":37, "mont-picocruz":30}

station_names = list(mapper.keys())
station_names2 = station_names.copy()
station_names2.remove("loscanarios") #remove "loscanarios"

locations = {}
for key in mapper:
    point = station_loc["geometry"][station_loc["OBJECTID"] == mapper[key]].values
    locations[key] = point

names = list(locations.keys())
points = list(locations.values())
lat = []
long = []
for point in points:
    long.append(float(point.x))
    lat.append(float(point.y))
    
loc_df = pd.DataFrame()
loc_df["station_name"] = pd.Series(names)
loc_df["points"] = pd.Series(points)
loc_df["lat"] = pd.Series(lat)
loc_df["long"] = pd.Series(long)
loc_df["size"] = pd.Series(np.ones(19)*9)

stations = {}
for key in mapper:
    stations[key] = pd.read_csv(key + ".csv")
    stations[key]["time"] = pd.to_datetime(stations[key]["time"])
    stations[key] = stations[key].set_index("time", drop = True)
    
#loscanarios analysis... (import results of regression model)

error_metrics = pd.read_csv("error_metrics.csv", index_col = 0)
XGBR_df = pd.read_csv("XGBR_df.csv")

model_results = {}
for name in station_names:
    if name != "loscanarios":
        model_dfi = pd.read_csv("model_" + name + ".csv", index_col = 0)
        model_results[name] = model_dfi
        
#%% FUNCITONS DEFINITION

def render100StackedChart(df, title):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(
            y = df[col],
            x = df.index,
            name = col
            ))
    fig.update_layout(
            yaxis=dict(
            title_text= title + " Electricity Generation Mix",
            ticktext=["0%", "20%", "40%", "60%","80%","100%"],
            tickvals=[0, 20, 40, 60, 80, 100],
            tickmode="array",
            titlefont=dict(size=15),
        ),
        autosize=True,
        # width=1000,
        # height=400,
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': title + " Electricity Generation Mix",
            'y':0.96,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        barmode='relative')
    return fig

def multipleTimeseries(features, station):
    # data = station[[features]]
    # fig = go.Figure()
    # fi
    # for feature in features:
    #     fig.add_trace(go.Line(
    #         x = data.index,
    #         y = 
    #         ))
    data_filt = stations[station][[features]]
    fig = px.line(data_filt, x=data_filt.index, y=data_filt)
    fig.update_layout(title_text="Raw Timeseries with Range Slider")
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                          label="1m",
                          step="month",
                          stepmode="backward"),
                    dict(count=6,
                          label="6m",
                          step="month",
                          stepmode="backward"),
                    dict(count=1,
                          label="YTD",
                          step="year",
                          stepmode="todate"),
                    dict(count=1,
                          label="1y",
                          step="year",
                          stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return fig


def display_map(loc_df):
    fig = px.scatter_mapbox(loc_df, lat="lat", lon="long", 
                            hover_data=["station_name"], 
                            color = loc_df["station_name"],
                            size = "size",
                            zoom = 9.5)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def generate_table(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range((len(dataframe)))
        ])
    ])

#%% 2. BUILD APP
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colors = {
#     'background': '#1A5276',
#     'text': '#FFFF99 '
#     }

server = app.server

app.layout = html.Div(children=[
    html.H3(children="Solar energy potential and radiation forecasting in La Palma, Canarias",
            style={
            'textAlign': 'center',
            "font-weight":"bold"
        }),
    dcc.Tabs(
        id='tabs', 
        value='tab-1', 
        children=[
            dcc.Tab(label='Introduction', value='tab-1'),
            dcc.Tab(label='Meterological Stations Map', value='tab-2'),
            dcc.Tab(label="Exploratory Data Analysis", value = "tab-3"),
            dcc.Tab(label="Building the Model", value = "tab-4"),
            dcc.Tab(label="Model Results", value = "tab-5"),
            ]
        ),
    html.Div(
        id='tabs-content'
        )
])

#RENDER TABS
@app.callback(Output(component_id='tabs-content', component_property='children'), #explicit sintax to remember how the callback works
              Input(component_id='tabs', component_property='value'))

def render_tab(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H5('Purpose of the study',
                    style = {
                        "font-weight":"bold"}
                    ),
            html.H6('The electricity generation mix of La Palma in the Canary Islands Archipelago is extremely reliant on fossil fuels, with little change in the past decade.'
                    + "The spanish national electricity mix, on the other hand, is very different and has been improving constantly in the last 30 years, as evidenced by the plots."),
            dcc.Graph(
                id = "lapalma-mix",
                figure = render100StackedChart(lpmix_percentage, "La Palma")
                ),
            dcc.Graph(
                id = "national-mix",
                figure = render100StackedChart(esmix_percentage, "Spain National")
                ),
            html.H5("Opportunity",
                    style = {
                     "font-weight":"bold"}),
            html.H6("The solar capacity has been a very small, constant fraction in the past years, despite the abundance of the resource."),
            html.H6(" The purpose of this study is to provide a preliminary assessment of the resource, by studying the readings of different sensors located around the island"),
            html.H5("La Palma Weather Sensors",
                    style = {
                     "font-weight":"bold"}),
            html.H6("On the island there are in total 19 sensors with readings of several metereological features, however only one of them has readings for normal total irradiation. To address this issue and to provide an estimate of the solar resource in the island, a regression model will be built using the years of radiation data of the one sensor, and then the model will be used to forecast the data in the other sensors"),
        ])
    
    elif tab == 'tab-2': 
        return html.Div([
            html.H5("Map of the sensors analysed",
                    style = {
                     "font-weight":"bold"}
                    ),
            dcc.Graph(
                id = "sensor-map",
                figure = display_map(loc_df)),
            # html.H6("Of all the sensors, only the sensor -loscanarios- records normal direct irradiation"),
            # html.H5("Exploratory Data Analysis of all the data recorded by the sensors"),
            # html.H6("Which sensor do you want to visualize?"),
            # html.Div(
            #     id = "sensor-content"
            #     ),
        ])
    
    elif tab == 'tab-3': 
        return html.Div([
            html.H5('Exploratory Data Analysis on cleaned data',
                    style = {
                     "font-weight":"bold"}),
            html.H6("WHich plot type do you want?"),
            dcc.RadioItems(
                id = "select-plot-type",
                options = [
                    {'label': "timeseries", 'value': 'timeseries'},
                    {'label': 'surface map', 'value': 'surface map'},
                    {"label": "boxplot", "value": "boxplot"},
                    ],
                value = 'surface map',
                labelStyle={'display': 'inline-block'},
                ),
            html.Div(
                id = "render-select-plot-type"
                )            
        ])
    
    elif tab == 'tab-4':
        return html.Div([
            html.H5("Radiation forecasting model of the station loscanarios",
                    style = {
                     "font-weight":"bold"}),
            html.H6("The main issue of the sensors around the island is that out of 19, only one has recordings of normal direct radiation, essential to perform any feasibility study for PV installments."),
            html.H6("To overcome this, using the data of the sensor 'loscanarios', a regression model will be built using the features 'hum', 'wind', 'maxwind', 'winddir', 'rain', 'radiation', 'temp', 'Hour', 'Month'."),
            html.H6("As all these features are present in all the other sensors, this model will be used to retrofit the radiation according to the values of the other columns, allowing to visualize the radiation in different parts of the island and providing a preliminary analysis of the solar resource assessment"),
            html.H5("Error metrics", 
                    style = {
                     "font-weight":"bold"}),
            html.H6("In the table below are summarized the error metrics of the model developed, as well as a scatterplot and timeseries comparison of the test data vs predicted data"),
            generate_table(error_metrics),
            html.H6("Test data vs Model Predicted data for the monitoring station of loscanarios",
                    style = {
                     "font-weight":"bold"}),
            dcc.Graph(
                id = "scatter-model",
                figure = px.scatter(XGBR_df, x="test data", y = "model predicted data")),
            dcc.Graph(
                id = "timeseries-model",
                figure = px.line(XGBR_df, y=["test data", "model predicted data"])),
            ])
    
    elif tab == "tab-5":
        return html.Div([
            html.H5("Radiation for all sensors",
                    style = {
                     "font-weight":"bold"}),
            html.H6("After having built the model on the data of the sensor 'loscanarios', using the method 'Extreme Gradient Boosting', chosen according to the error metrics, the following radiation profiles were obtained for the other sensors."),
            html.H6("Choose Sensor: "),
            dcc.Dropdown(
                id = "choose-station",
                options = [{"label":x, "value":x} for x in station_names2],
                value = "fagundo"
                ),
            dcc.Graph(
                id = "plot-choose-station")
        ])

#render-select-sensor
@app.callback(Output("render-select-plot-type", "children"),
              Input("select-plot-type", "value"))

def renderPlotType(plotType):
    if plotType == "timeseries":
        return html.Div([
            html.H6("Which sensor?"),
            html.Div([
                dcc.Dropdown(
                    id = "select-sensor",
                    options = [{"label":x, "value":x} for x in loc_df["station_name"]],
                    value = "loscanarios",
                    ),
                dcc.RadioItems(
                    id = "select-feature", #should return list of strings
                    options = [{"label":x, "value":x} for x in stations["loscanarios"].columns[1:-2]],
                    value = "temp",
                    labelStyle={'display': 'inline-block'},
                    ),
                ]),
            dcc.Graph(
                id = "timeseries"
                ),
            ])
    elif plotType == "surface map":
        return html.Div([
            html.H6("Which sensor?"),
            html.Div([
                dcc.Dropdown(
                    id = "select-sensor",
                    options = [{"label":x, "value":x} for x in loc_df["station_name"]],
                    value = "loscanarios",
                    ),
                dcc.RadioItems(
                    id = "select-feature",
                    options = [{"label":x, "value":x} for x in stations["loscanarios"].columns[1:-2]],
                    value = "temp",
                    labelStyle={'display': 'inline-block'},
                    ),
                ]),
            dcc.Graph(
                id = "surface")
            ])       
    elif plotType == "boxplot":
        return html.Div([
            html.H6("Which sensor?"),
            html.Div([
                dcc.Dropdown(
                    id = "select-sensor",
                    options = [{"label":x, "value":x} for x in loc_df["station_name"]],
                    value = "loscanarios",
                    ),
                dcc.RadioItems(
                    id = "select-feature",
                    options = [{"label":x, "value":x} for x in stations["loscanarios"].columns[1:-2]],
                    value = "temp",
                    labelStyle={'display': 'inline-block'},
                    ),
                ]),
            dcc.Graph(
                id = "boxplot")
            ])         
        
#render-select-plot-type
@app.callback(Output("timeseries", "figure"),
              Input("select-sensor", "value"),
              Input("select-feature", "value"))

def displayTimeseries(sensor, feature):
    data_filt = stations[sensor][[feature]].copy()
    data_filt = data_filt.dropna()
    data_filt["Date"] = data_filt.index
    fig = px.line(data_filt, x=data_filt["Date"], y=data_filt[feature])
    fig.update_layout(title_text="Raw Timeseries of: " + feature)
    return fig

@app.callback(Output("surface", "figure"),
              Input("select-sensor", "value"),
              Input("select-feature", "value"))

def displaySurface(sensor, feature):
    if feature not in stations[sensor].columns:
        return go.Figure()
    features = [feature, "Hour"]
    df_map = stations[sensor][features]
    df_map.index = df_map.index.date
    df_map_pivot = df_map.pivot(columns = "Hour")
    df_map_pivot.dropna()
    fig = go.Figure(data=[go.Surface(z=df_map_pivot.values, 
                                 x = np.array(range(24)),
                                 y = df_map.index
                                 )
                      ])
    fig.update_layout(autosize=False,
                  width=1500, height=800,
                  ),
    fig.update_layout(scene = dict(
                    xaxis_title='Hour',
                    yaxis_title='Date',
                    zaxis_title=feature,
                    ))
    fig.update_layout(
        title={
            'text': str("Daily " + features[0] + " over time"),
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title={"text":"Hour"},
        yaxis_title="Date",
        )
    return fig
    
@app.callback(Output("boxplot", "figure"),
              Input("select-sensor", "value"),
              Input("select-feature", "value"))

def render_boxplot(sensor, feature):
    data_filt = stations[sensor][[feature]]
    fig = px.box(data_filt, x = data_filt.index.year, y = data_filt[feature])
    fig.update_layout(title_text = "Boxplot of " + sensor + " for the years recorded: " + feature)
    fig.update_xaxes(title_text='Years')
    return fig    
    
@app.callback(Output("plot-choose-station", "figure"),
              Input("choose-station", "value"))

def render_model(station):
    fig = px.line(model_results[station], y=model_results[station][station])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)