from arch import arch_model
from datetime import datetime as dt
from tvDatafeed import TvDatafeed, Interval
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from retry import retry
import statsmodels.api as sm
import streamlit as st



date = dt.today().date()
@retry((Exception), tries=15, delay=1, backoff=0)
def get_ticker_data(country, ticker,n,freq,date):

    country_exchange = {'Egypt': 'EGX'}
    interval_dic = {'Daily':Interval.in_daily, 'Weekly':Interval.in_weekly, 'Monthly':Interval.in_monthly}
    tv = TvDatafeed()
    response = tv.get_hist(symbol=f'{ticker}',
                    exchange=country_exchange[country],
                    interval=interval_dic[freq],
                    n_bars=n)['close']
    response = pd.DataFrame(response)

    return response


def calculate_returns(x, period, log=True):
    """calculates returns data for a series

    Args:
        x (pd.Series): pandas series to be 
        period (int): 1-day returns, 3-day returns etc. 

    Returns:
        x_rt (pd.Series): Returns for thes series
    """
    if log:
        x_rt = np.log(1+x.pct_change(period))[period:]
    
    else:
        x_rt = x.pct_change(period)[period:]

    return x_rt

def model(x_rt):
    garch = arch_model(x_rt, vol='EGARCH', mean='AR',lags=2,p=1, o=1, q=1, dist='skewt', rescale=True)
    garch_result = garch.fit(cov_type='robust',update_freq=5, disp="off")
    return garch_result

def calculate_risk_premium(x_rt, vix):
    x_rt = x_rt.reindex(vix.index)
    risk_premium = vix['cond_vol']/x_rt.rolling(window=30).std()
    return risk_premium

def all_plotting(close,vix, risk_premium):

    close = close.reindex(vix.index)

    fig = make_subplots(rows = 2, cols = 1,
                        vertical_spacing=0.01, shared_xaxes = True,specs=[[{"secondary_y": True}],[{"secondary_y": False}]],
                        row_heights=[0.7,0.3]
                        )
    fig.add_trace(
        go.Scatter(x = close.index, y = close, mode='lines', name='Price',
                   opacity=0.7, line=dict(color='skyblue', width=1)), secondary_y=False, row=1, col=1
        )
    fig.add_trace(
        go.Scatter(x = close.index, y =vix['cond_vol'], name='Volatility',
                   mode='lines', opacity=0.5,line=dict(color='red', width=1)), secondary_y=True,row=1,col=1
        )

    fig.append_trace(
        go.Scatter(x = close.index, y =risk_premium, mode='lines',name='Risk Premium',
        line=dict(width=1)),row=2, col=1
        )
    

    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volatility", secondary_y=True)  

    fig.update_yaxes(title_text="Risk Premium Multiple", secondary_y=False, row=2,col=1)  

    fig.update_layout(width=2000, height=1000, autosize=True)
    fig.update_layout(hovermode="x unified",grid=dict(rows=2, columns=1),hoversubplots="axis")



    return fig



def plot_clusters(vix):

    fig2 = go.Figure()
    fig2 = px.scatter(y = vix['cond_vol'], x=vix.index, color=vix['Regime'], color_continuous_scale='Reds')
                                
    fig2.update_yaxes(title_text="Volatility Regimes")
    return fig2



def main(ticker, interval):

    try:
        df = get_ticker_data('Egypt', ticker, n=1500, freq=interval, date=date)    
    except:
        st.write('Invalid Ticker')
    period = 1
    x_rt = calculate_returns(df['close'],period,log=False)

    garch_result = model(x_rt)

    vix = pd.DataFrame(garch_result.conditional_volatility).dropna()


    MarkovModel = sm.tsa.MarkovRegression(endog=vix, k_regimes=3,
                                switching_variance=True, switching_trend=True).fit(search_reps=30)

    MarkovModel.smoothed_marginal_probabilities['regime']=MarkovModel.smoothed_marginal_probabilities.apply(np.argmax,axis=1)
    probs = MarkovModel.smoothed_marginal_probabilities
    probs.columns = ['Low','Medium','High', 'Regime']
    vix['Regime'] = probs['Regime']

    risk_premium = calculate_risk_premium(x_rt,vix)

    fig = all_plotting(df['close'],vix, risk_premium)
    fig2 = plot_clusters(vix)
    return vix,fig, fig2, risk_premium, df['close']




################################################################
#########################################################

st.set_page_config(layout="wide")
ticker = st.text_input(label="Enter Ticker in (Caps): ",
              value='EGX30',
              key='ticker')
interval = st.selectbox(label='Enter Desired Interval: ',
                        options=['Daily','Weekly','Monthly'],
                        key='interval')


vix,fig, fig2,risk_premium, close = main(ticker,interval)


st.plotly_chart(fig)

hv = vix['cond_vol']/risk_premium
historical_stats = pd.DataFrame(data = {'Regime':vix['Regime'],'Volatility':vix['cond_vol'],'Historical_Volatility' :hv, 'Risk_Premium':risk_premium})

historical_stats.set_index('Regime',drop=True, inplace=True)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current Regime", f'{historical_stats.index[-1]}')

col2.metric("Current Volatility", f'{historical_stats.Volatility.iloc[-1]:.2f}',
            f'{historical_stats.Volatility.pct_change(1).iloc[-1]:.2f}')

col3.metric("30 Period Historical Volatility", f'{historical_stats.Historical_Volatility.iloc[-1]:.2f}',
            f'{historical_stats.Historical_Volatility.pct_change(1).iloc[-1]:.2f}')

col4.metric("Risk Premium", f'{historical_stats.Risk_Premium.iloc[-1]:.2f}',
            f'{historical_stats.Risk_Premium.pct_change(1).iloc[-1]:.2f}')

col5.metric("Current Price", close[-1],
            f'{close.pct_change(1)[-1]:.3f}')


st.plotly_chart(fig2)
