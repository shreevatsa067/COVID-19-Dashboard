# COVID-19 Dashboard Prototype
# By, Shreevatsa Puttige Subramanya (411866)
# This Dashboard gives an overview of the spread of COVID-19 around the world. 
# The graphs show the confirmed cases and the doubling rate for each country. 
#Finally, a SIR model for spread of disease is implemented and is visualised.

import pandas as pd
import numpy as np
import json
import requests
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from datetime import datetime
from sklearn import linear_model
from scipy import signal
from scipy import optimize
from scipy import integrate
reg = linear_model.LinearRegression(fit_intercept=True)

def get_data():
    '''
    This function gets the data through REST API
    The COVID-19 data source in the API is from Johns Hopkins CSSE
    
    Returns:
    --------
    Pandas dataframe of COVID-19 data    
    
    '''
    page = requests.get("https://api.covid19api.com/all")
    json_object = json.loads(page.content)
    pd_dataframe = pd.DataFrame(json_object)
    pd_dataframe = pd_dataframe.drop(['Lat','Lon','CountryCode','CityCode'],axis=1)
    pd_dataframe['Date'] = pd_dataframe['Date'].astype('datetime64[ns]')
    
    #Collecting overall country data for USA
    US_Data = pd_dataframe[(pd_dataframe['Country']=='United States of America') &\
                         (pd_dataframe['Province'] == '')].drop(['Province','City'],axis=1).reset_index(drop=True)
        
    #Deleting the city wise distribution of the US data from the original dataframe
    dataframe = pd_dataframe.drop(pd_dataframe[pd_dataframe['Country'] == 'United States of America'].index)\
        .drop(['Province','City'],axis=1).reset_index(drop=True)
        
    #Groupby apply to get the daily status for each country 
    data_frame = dataframe.groupby(['Country','Date']).agg(np.sum).reset_index()
    
    #Appending the US data to the original dataframe
    df_input = data_frame.append(US_Data, ignore_index=True)
    
    return df_input

def choropleth_map(input_df):
    '''
    creates a choropleth graph object
    
    Parameters:
    ----------
    Dataframe of the current day COVID-19 stats
    
    Returns:
    -------
    fig3: plotly choropleth map graph object
    
    '''
    fig3 = go.Figure(
        data=go.Choropleth(
            locations=input_df['Country'],
            locationmode='country names',
            z=input_df['Confirmed'],
            text=input_df['Deaths'],
            hovertext=input_df['Recovered'],
            colorscale = 'Blues',
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title = 'Confirmed Cases<br>as of<br>'+str(input_df.loc[0,'Date']),
            hovertemplate = input_df.Country + "<br>Confirmed Cases: %{z} <br>Total Deaths: %{text} <br>Total Recovered: %{hovertext}<extra></extra>"
            )
        )
    fig3.update_layout(
        height=600, margin={"r":10,"t":50,"l":10,"b":10}, template='plotly_dark',
        title_text='World map to visualise the spread of COVID-19', title_x=0.5)
                
    return fig3

def doubling_rate(df_input, clmn='Confirmed'):
    '''
    Helper function to group-by calculate the doubling time
    
    Parameters:
    ----------
    df_input: pandas Dataframe
    clmn: The column on which the doubling rate is calculated 
                (Confirmed or confirmed_filtered)
    
    Returns:
    -------
    df_output: Pandas dataframe with the additional columns
    
    '''
    
    pd_DR_result = df_input.groupby('Country').apply(rolling_reg, clmn).reset_index()

    pd_DR_result = pd_DR_result.rename(columns ={clmn:clmn+'_DR','level_1':'index'})

    #Merging the dataset with doubling rate column with the original dataframe
    
    df_output = pd.merge(df_input,pd_DR_result[['index',str(clmn+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output = df_output.drop(columns=['index'])
    return df_output
    
def doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate

        Parameters:
        ----------
        in_array : pandas.series

        Returns:
        ----------
        intercept/slope (Doubling rate): double
    '''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept = reg.intercept_
    slope=reg.coef_

    return intercept/slope

def rolling_reg(df_input, col='Confirmed'):
    ''' Rolling Regression to approximate the doubling time'

        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str - Confirmed or Confirmed_filtered column
        
        Returns:
        ----------
        result: Pandas DataFrame
    '''
    days_back = 3
    result = df_input[col].rolling(
                window = days_back,
                min_periods = days_back).apply(doubling_time_via_regression, raw = False)
    
    return result

def filter_data(df_input,column='Confirmed'):
    '''  Helper function to apply savgol filter to filter the data in confirmed or confirmed_filtered column

        Parameters:
        ----------
        df_input: pd.DataFrame
        column: str - Confirmed or Confirmed_filtered column
            
        Returns:
        ----------
        df_output: Pandas DataFrame with merged additional columns
    
    '''

    df_output = df_input.copy() 

    pd_filtered_result = df_output[['Country',column]].groupby(['Country']).apply(savgol_filter)#.reset_index()

    df_output = pd.merge(df_output,pd_filtered_result[[str(column+'_filtered')]],left_index=True,right_index=True,how='left')
   
    return df_output.copy()

def savgol_filter(df_input,column='Confirmed',window=5):
    ''' Savgol Filter to filter the data in confirmed or confirmed_filtered column  

        Parameters:
        ----------
        df_input : pandas.series
        column : str - Confirmed or Confirmed_filtered column
        window : int - window size (or number of data points) used to filter data

        Returns:
        ----------
        df_result: Pandas DataFrame
            the index of the df_input is retained to merge the dataset in filter_data
    '''

    df_result = df_input

    filter_in = df_input[column].fillna(0)

    result = signal.savgol_filter(np.array(filter_in), window, 1)
    
    df_result[str(column+'_filtered')] = result
    return df_result

#Get the covid-19 data through REST API
covid_data = get_data()
#Dataframe for the choropleth map
df_map = covid_data.loc[covid_data['Date']==covid_data['Date'].max()].reset_index(drop=True)
df_map['Date'] = pd.to_datetime(df_map['Date']).dt.date

pd_api_data = covid_data[['Country', 'Date', 'Confirmed']]

#To obtain the filtered data for the confirmed cases
final_df = filter_data(pd_api_data)

#Calculating the doubling rate for confirmed and confirmed filtered column
final_df = doubling_rate(final_df).reset_index(drop=True)
final_df = doubling_rate(final_df,'Confirmed_filtered').reset_index(drop=True)

#Defining a mask to have doubling rate values for confirmed cases more than 100
mask = final_df['Confirmed']>100
final_df['Confirmed_filtered_DR'] = final_df['Confirmed_filtered_DR'].where(mask, other=np.NaN)
    
app=dash.Dash()

app.layout=html.Div(children=[
    
    dcc.Markdown('''
                 # Covid-19 pandemic Dashboard
                 ## This dashboard shows the spread of COVID-19 pandemic
                 ''', style={
                 'fontFamily': 'sans-serif',
                 'textAlign': 'center',
                 'backgroundColor':'#070B20',
                 'margin': '5px',
                 'color': '#F9690E',
                 'padding': '5px',
                 'borderRadius': '5px'}),
    
    html.Div(dcc.Graph(id='map', figure=choropleth_map(df_map)), style = {
                    'width': '100%',
                    'borderRadius': '5px',
                }),
                    
    dcc.Markdown('''
                 ## The plot below shows the spread of COVID-19 over time for different countries in the dropdown menu
                 ### In the second dropdown, one can select between the actual confirmed and doubling rate\
                 or the filtered confirmed and filtered doubling rate. A savgol filter is used to filter the data
                 ''', style={
                 'fontFamily': 'sans-serif',
                 'textAlign': 'center',
                 'backgroundColor':'#070B20',
                 'color': '#F39C12',
                 'margin': '5px',
                 'padding': '5px',
                 'borderRadius': '5px'}),
                 
    dcc.Dropdown(
            id='country_drop_down',
            options=[{'label': each,'value':each} for each in final_df['Country'].unique()],
            value=['Germany','United States of America'],
            multi=True,
            style={'margin': '10px', 'width':'1000px'}
        ),
    
     dcc.Dropdown(
        id='stats',
        options=[
            {'label': 'Confirmed ', 'value': 'Confirmed'},
            {'label': 'Doubling Rate', 'value': 'Confirmed_DR'},
            {'label': 'Confirmed Filtered', 'value': 'Confirmed_filtered'},
            {'label': 'Doubling Rate Filtered', 'value': 'Confirmed_filtered_DR'}
        ],
        value='Confirmed',
        multi=False,
        style={'width':'300px','margin': '10px'}
        ),
    
    dcc.Graph(id='display_stats',style={'width':'100%'}),
    
    dcc.Markdown('''
                 ## SIR model for spread of disease
                 ''', style={
                 'fontFamily': 'sans-serif',
                 'textAlign': 'center',
                 'backgroundColor':'#070B20',
                 'color': '#F39C12',
                 'margin': '5px',
                 'padding': '5px',
                 'borderRadius': '5px'}),
                 
    dcc.Dropdown(
            id='country_drop_down_sir',
            options=[{'label': each,'value':each} for each in final_df['Country'].unique()],
            value='Germany',
            multi=False,
            style={'margin': '10px','width':'1000px'}
            ),
    
    dcc.Graph(id='sir_curves', style={'width':'100%'})
], style={'backgroundColor':'#B8B8B8'})
                             
                 
@app.callback(
    Output('display_stats', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('stats', 'value')])
def update_figure(country_list,show_doubling):


    if 'Confirmed_DR' in show_doubling or 'Confirmed_filtered_DR' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infections (source johns hopkins, log-scale)'
              }
    
    fig1=go.Figure()
    for each in country_list:

        df_plot=final_df[final_df['Country']==each]
        df_plot[['Country','Date','Confirmed','Confirmed_filtered','Confirmed_DR','Confirmed_filtered_DR',]].groupby(['Country','Date']).agg(np.sum).reset_index()
        fig1.add_traces(go.Scatter(x=df_plot.Date, y=df_plot[show_doubling], mode='markers+lines', 
                                   opacity=0.9, name=each))
        fig1.update_layout(template='plotly_dark', height=720,
                           xaxis={'title':'Timeline',
                          'tickangle':-45,
                          'nticks':20,
                          'tickfont':dict(size=14,color="#FFFFFF")
                          },
                           yaxis=my_yaxis
                         )

    return fig1

@app.callback(
    Output('sir_curves', 'figure'),
    [Input('country_drop_down_sir', 'value')])

def SIR_curves(country_list_sir):
    
    df_sir=covid_data[covid_data['Country']==country_list_sir].reset_index(drop=True)
    ydata=np.array(df_sir[['Confirmed', 'Recovered']].reset_index(drop=True).iloc[20:,])
    N0=10000000
    
    def SIR_model_eq(SIR,t,beta,gamma):
        ''' Simple SIR disease spread model
            S: susceptible population
            t: time step
            I: infected people
            R: recovered people
            beta: Infection rate
            gamma: Recovery rate
            
            S+I+R= N (constant size of population)
        '''
        S,I,R=SIR
        dS_dt=-beta*S*I/N0           
        dI_dt=beta*S*I/N0-gamma*I
        dR_dt=gamma*I
        return dS_dt,dI_dt,dR_dt

    def integrate_sir(x, beta, gamma):
        
        return integrate.odeint(SIR_model_eq, (S0, I0, R0), t, args=(beta, gamma))[:,1] 
    
    fig2=go.Figure()
    window_sir = 30
    fitted_final=[]
    
    for i in range(len(ydata)):
        if i%window_sir ==0:
            if len(ydata[i:])>5:
                y_new=ydata[i:i+window_sir,0]
                t=np.arange(len(y_new))
                #Initialize parameters of SIR model
                R0=ydata[i,1]
                I0=y_new[0]
                S0=N0-I0-R0
                #Calculate optimized beta and gamma
                popt, pcov = optimize.curve_fit(integrate_sir, t, y_new)
                fitted=integrate_sir(t, *popt)
                fitted_final.extend(fitted)
    
    fig2.add_traces(go.Scatter(x=df_sir.Date[20:,], y=fitted_final, mode='markers+lines', 
                               opacity=0.9, name='SIR estimated'))
    
    fig2.add_traces(go.Bar(x=df_sir.Date, y=df_sir.Confirmed, opacity=0.7,
        name=country_list_sir+' - Confirmed cases'))
    
    fig2.update_layout(template='plotly_dark', height=720,
                xaxis={'title':'Timeline', 'tickangle':-45,'nticks':20,
                       'tickfont':dict(size=14,color="#FFFFFF")
                       },
                yaxis={'type':"log",
                       'title':'Confirmed infected people'
                       })
    
    return fig2
    
                 
if __name__ == '__main__':
    
    app.run_server(debug=True, use_reloader=False)
