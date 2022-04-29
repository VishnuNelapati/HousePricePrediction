from flask import Flask , render_template , request , redirect , url_for
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pickle
from sklearn.model_selection import train_test_split
from utils import get_base_url

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
# port = 12345
port = 12346
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# Function to get the data
def data():
    return pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/main_df.csv")

# Function to plot geo distribution graph of houses

def geomap1():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    fig = px.scatter_mapbox(zillow[(zillow.notnull()['latitude'] & zillow.notnull()['price'])],
                            lat="latitude", lon="longitude", hover_name="city",
                            color='price', zoom=9, size_max=12, size='price',
                            # width=550,
                            hover_data=['bedrooms', 'bathrooms', 'homeType'])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

# def geomap2():
#     # return components.iframe("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/grid_layer.html" ,width = 650,height=600,scrolling = True)
#     return

def pie():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    home_type = pd.DataFrame(zillow['homeType'].value_counts()).reset_index()
    home_type.columns = ['HomeType','Count']
    fig = px.pie(data_frame=home_type,names = 'HomeType',values = 'Count',labels={'Count':'No of Houses'},color = 'HomeType')
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(width = 600,height = 500,title = "Pie Chart for Different Home-types",title_x = 0.5)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def p2_bar():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    avg_price = zillow[['homeType', 'Price Square FeetPrice/sqft']].groupby(['homeType']).mean().reset_index()
    avg_price.columns = ['homeType', 'Avg Price/sqft']
    avg_price = avg_price.sort_values(by=['Avg Price/sqft'], ascending=False)
    fig = px.bar(data_frame=avg_price, x='homeType', y='Avg Price/sqft', hover_data=['homeType', 'Avg Price/sqft'],
                 color='Avg Price/sqft',
                 title='Average Price/Sqft For Each HomeType')
    fig.update_layout(width=600, height=500, title_x=0.5)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def p3_bar():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    mini_df = zillow[['city', 'CalendarYear built']]

    # Arranging the data frame in the ascending oreder of 'CalendarYear built'
    mini_df = mini_df.sort_values(by='CalendarYear built', ascending=True)

    # Considering top 500 and bottom 500 properties to perform the analysis
    old_properties = mini_df.head(500)
    new_properties = mini_df.tail(500)

    # Identifying the top 10 cities with the oldest properties and newest properties
    old_properties = old_properties['city'].value_counts().rename_axis('City').reset_index(
        name='Number of properties').head(10)
    new_properties = new_properties['city'].value_counts().rename_axis('City').reset_index(
        name='Number of properties').head(10)

    # Plotting bar chart
    # Adjusting figure size and sharing y-axis.

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Cities with old properties", "Cities with new properties"))

    fig.add_trace(go.Bar(x=old_properties['City'], y=old_properties['Number of properties'],
                         marker=dict(color=old_properties['Number of properties'], coloraxis="coloraxis")),
                  1, 1)

    fig.add_trace(go.Bar(x=new_properties['City'], y=new_properties['Number of properties'],
                         marker=dict(color=old_properties['Number of properties'], coloraxis="coloraxis")),
                  1, 2)

    fig.update_layout(showlegend=False, yaxis=dict(title="Number of Properties"), width=800, height=400)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


#----------------------------------------------------------------------------------------------------------------------------------------------------------

def p4_ratings(arg = "Livability" , agg=False):
    zillow = data()
    zillow.set_index('zpid',inplace=True)
    graphs = []
    for arg in ['Livability','Crime','Employment','Schools','Housing']:
        if arg == 'Livability':
            df = zillow[['price', arg]].groupby(by=arg).mean().reset_index()
        else:
            df = zillow[['price', arg]].groupby(by=arg).mean().reset_index().sort_values(by=arg, key=lambda
                g: g + ',', ascending=False)

        fig = px.scatter(data_frame=df, x=arg, y='price', size='price', size_max=20,
                         color='price', height=450, width=600,
                         title=f"Scatter Plot Representation Of Price Against {arg} ratings",
                         hover_data=['price', arg])
        fig.update_layout(title_x=0.5)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        graphs.append(graphJSON)

    return graphs




#==============================================================================================================================================================

zillow_detail_df = data()
zillow_detail_df = zillow_detail_df.set_index('zpid')
regression_df = zillow_detail_df

regression_df = regression_df[['city','price','bathrooms','bedrooms','livingArea',
                    'homeType','taxAssessedValue',
                    'Price Square FeetPrice/sqft','HasHeating','HasCooling','GarageSpaces','HasPool',
                     'FirePlaces','Crime', 'Employment', 'Schools']]

regression_df = regression_df[regression_df.GarageSpaces<=6]
reg_df = regression_df.drop('price',axis=1)

ratings_dict ={'F':1,'D-':2,'D':3,'D+':4,'C-':5,'C':6,'C+':7,
                             'B-':8,'B':9,'B+':10,'A-':11,'A':12,'A+':13}
city_avgdf = zillow_detail_df[['city','price']].groupby(by = ['city']).mean().reset_index().sort_values(by = ['price'])
city_avgdf['ranks'] = city_avgdf['price'].rank()
city_dict = dict(zip(city_avgdf.city.values,city_avgdf.ranks.values.astype(np.int64)))

def format(df):
    d ={}
    for j, i in enumerate(reg_df.columns):
        d.update({i: df[j]})

    df = pd.DataFrame(d, index=[0])
    df = df[reg_df.columns]
    df.Schools = df['Schools'].map(ratings_dict)
    df.Crime = df['Crime'].map(ratings_dict)
    df.Employment = df['Employment'].map(ratings_dict)
    # df.Housing = df['Housing'].map(ratings_dict)
    df.city = df['city'].map(city_dict)
    #
    return df


#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def p5_corr():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    zillow = zillow[['city','price','bathrooms','bedrooms','livingArea',
                        'homeType','taxAssessedValue','lotAreaValue','CalendarYear built',
                        'Price Square FeetPrice/sqft','HasFlooring','HasHeating','HasCooling','GarageSpaces','HasLaundary',
                         'FirePlaces','HasPool','HasSecurity','Stories','Livability','Crime','Employment','Housing','Schools']]

    for i in ['HasFlooring', 'HasHeating', 'HasCooling', 'HasLaundary', 'HasPool', 'HasSecurity']:
        zillow[i] = zillow[i].map({'Yes': 1, 'No': 0})

    zillow.Schools = zillow['Schools'].map(ratings_dict)
    zillow.Crime = zillow['Crime'].map(ratings_dict)
    zillow.Employment = zillow['Employment'].map(ratings_dict)
    zillow.Housing = zillow['Housing'].map(ratings_dict)
    zillow.city = zillow['city'].map(city_dict)

    zillow = pd.concat((zillow, pd.get_dummies(zillow.homeType, drop_first=True)), axis=1)
    zillow.drop(['homeType'], inplace=True, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(zillow.drop("price",axis = 1), zillow[['price']], test_size=0.25, random_state=1)

    train = pd.concat((X_train, y_train), axis=1)

    fig = ff.create_annotated_heatmap(z=np.round(train.corr().values, 2),
                                      x=list(train.select_dtypes(exclude='object').columns),
                                      y=list(train.select_dtypes(exclude='object').columns), colorscale='viridis')
    fig.update_layout(
        title='Correlation Heat Map',
        width=1200,
        height=900,
        hovermode='closest', title_x=0.5, title_y=0.05)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def p5_corr2():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    zillow = zillow[['city','price','bathrooms','bedrooms','livingArea',
                        'homeType','taxAssessedValue',
                        'Price Square FeetPrice/sqft','HasHeating','HasCooling','GarageSpaces','HasPool',
                         'FirePlaces','Crime', 'Employment', 'Schools']]

    for i in ['HasHeating', 'HasCooling','HasPool']:
        zillow[i] = zillow[i].map({'Yes': 1, 'No': 0})

    zillow.Schools = zillow['Schools'].map(ratings_dict)
    zillow.Crime = zillow['Crime'].map(ratings_dict)
    zillow.Employment = zillow['Employment'].map(ratings_dict)
    zillow.city = zillow['city'].map(city_dict)

    zillow = pd.concat((zillow, pd.get_dummies(zillow.homeType, drop_first=True)), axis=1)
    zillow.drop(['homeType'], inplace=True, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(zillow.drop("price",axis = 1), zillow[['price']], test_size=0.25, random_state=1)

    train = pd.concat((X_train, y_train), axis=1)

    fig = ff.create_annotated_heatmap(z=np.round(train.corr().values, 2),
                                      x=list(train.select_dtypes(exclude='object').columns),
                                      y=list(train.select_dtypes(exclude='object').columns), colorscale='viridis')
    fig.update_layout(
        title='Correlation Heat Map',
        width=1200,
        height=900,
        hovermode='closest', title_x=0.5, title_y=0.05)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

#----------------------------------------------------------------------------------------------------------------------------------------------------------


def p6_scatter():
    zillow = data()
    zillow.set_index('zpid', inplace=True)
    zillow = zillow[['city','price','bathrooms','bedrooms','livingArea',
                        'homeType','taxAssessedValue',
                        'Price Square FeetPrice/sqft','HasHeating','HasCooling','GarageSpaces','HasPool',
                         'FirePlaces','Crime', 'Employment', 'Schools']]

    for i in ['HasHeating', 'HasCooling','HasPool']:
        zillow[i] = zillow[i].map({'Yes': 1, 'No': 0})

    zillow.Schools = zillow['Schools'].map(ratings_dict)
    zillow.Crime = zillow['Crime'].map(ratings_dict)
    zillow.Employment = zillow['Employment'].map(ratings_dict)
    zillow.city = zillow['city'].map(city_dict)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Living Area vs Price", "Heating vs Price", "Bathrooms vs Price", "Price/Sqft vs Price","Bedrooms vs Price","Fireplace vs Price"))

    fig.add_trace(go.Scatter(x=zillow.livingArea, y=zillow.price, mode='markers', name = "Living Area"),
                  row=1, col=1 )

    fig.add_trace(go.Scatter(x=zillow.HasHeating, y=zillow.price, mode='markers', name = "Heating"),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=zillow.bathrooms, y=zillow.price, mode='markers' , name = "Bathrooms"),
                  row=1, col=3)

    fig.add_trace(go.Scatter(x=zillow['Price Square FeetPrice/sqft'], y=zillow.price, mode='markers' ,  name = "Price/Sqft"),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=zillow.bedrooms, y=zillow.price, mode='markers', name = "Bedrooms"),
                  row=2, col=2)

    fig.add_trace(go.Scatter(x=zillow.FirePlaces, y=zillow.price, mode='markers' ,  name = "Fireplace"),
                  row=2, col=3)

    fig.update_layout(height=600, width=1000,
                      title_text="Scatter plot",title_x = 0.5)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


#===============================================================================================================================================================

@app.route(f"{base_url}" , methods = ["GET","POST"])
def home():
    if request.method == "POST":
        pipeline1 = pickle.load(open('housemodel.pkl', 'rb'))
        values = [i for i in request.form.values()]
        input = []
        for i in values:
            try:
                input.append(int(i))
            except:
                input.append(i)

        input_df = format(input)
        
        input_df.fillna(20,inplace=True)

        html_df = pd.DataFrame(values).T
        html_df.columns = input_df.columns

        df_html = html_df.to_html(classes="table table-dark")
        pred = f"The predicted house price with above specifications cost approximately ${np.round(pipeline1.predict(input_df))[0]}"

        return render_template("index.html", graphJSON=geomap1(),
                               pie=pie(),
                               p2_bar=p2_bar(),
                               p3_bar=p3_bar() ,
                               p4_graph1 = p4_ratings()[0],
                               p4_graph2=p4_ratings()[1],
                               p4_graph3=p4_ratings()[2],
                               p4_graph4=p4_ratings()[3],
                               p4_graph5=p4_ratings()[4],
                               p5_graph = p5_corr(),
                               p5_graph2 = p5_corr2(),
                               p6_graph = p6_scatter(),
                               values = pred ,
                               df_html = df_html)

    return render_template("index.html" , graphJSON=geomap1() ,
                           pie = pie() ,
                           p2_bar = p2_bar() ,
                           p3_bar = p3_bar() ,
                           p4_graph1=p4_ratings()[0],
                           p4_graph2=p4_ratings()[1],
                           p4_graph3=p4_ratings()[2],
                           p4_graph4=p4_ratings()[3],
                           p4_graph5=p4_ratings()[4],
                           p5_graph = p5_corr(),
                           p5_graph2 = p5_corr2(),
                           p6_graph = p6_scatter())

# @app.route("/predict" , methods = ["GET","POST"])
# def predict():
#     if request.method == "POST":
#         pipeline1 = pickle.load(open('housemodel.pkl', 'rb'))
#         print("Requests",request.form['City'])
#         values = [i for i in request.form.values()]
#         input = []
#         for i in values:
#             try:
#                 input.append(int(i))
#             except:
#                 input.append(i)
#         pred = f"The predicted house price with specifications cost approximately ${np.round(pipeline1.predict(format(input)))[0]}"
#
#         return render_template("index.html", graphJSON=geomap1(), pie=pie(), p2_bar=p2_bar(), p3_bar=p3_bar() , values = pred)
#
#     return render_template("index.html" , graphJSON=geomap1() , pie = pie() , p2_bar = p2_bar() , p3_bar = p3_bar() )


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc4.ai-camp.dev/'
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
