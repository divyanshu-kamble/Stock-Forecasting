import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
import yfinance as yf
import numpy as np
from plotly import graph_objs as go
import pandas as pd
import import_ipynb
from prophet import Prophet
import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
from prophet.plot import plot_plotly, plot_components_plotly
# from model_show import Stock

window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2)

def nearest_business_day(DATE: date):
    if DATE.weekday() == 5:
        DATE = DATE - timedelta(days=1)

    if DATE.weekday() == 6:
        DATE = DATE + timedelta(days=1)
    return DATE

TODAY = date.today()
DEFAULT_START = date(2013, 1, 1)
START = sub_columns[0].date_input("From", value=DEFAULT_START,max_value=TODAY - timedelta(days=1))
END = sub_columns[1].date_input("To", value=TODAY,max_value=TODAY, min_value=START)

START = nearest_business_day(START)
END = nearest_business_day(END)

STOCKS = np.array(["ITC.NS", "IDEA","TCS.NS","MARUTI.NS","HDFCLIFE.NS","BRITANNIA.NS","BHARTIARTL.NS","BAJAJ-AUTO.NS"])
SYMB = window_selection_c.selectbox("select stock", STOCKS)

chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400)




def load_data(symbol, start, end):
    data = yf.download(symbol, start, end).reset_index()
    return data

data_load_state = st.text("Load data ...")
data = load_data(SYMB, START, END)
data_load_state.text("Loading data ... Done!")

st.write("###")

st.subheader("Raw data")
st.write(data.tail())


def plot_raw_data(fig, data, symbol):
    """
    Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
    """
    fig = fig.add_trace(
        go.Scatter(
            x=data.Date,
            y=data['Close'],
            mode="lines",
            name=symbol,
        )
    )
    return fig


# df_train = data[['Date', 'Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

fig = go.Figure()
fig = plot_raw_data(fig, data, SYMB)
fig.update_layout(
            # width=chart_width,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            legend=dict(
                x=0,
                y=0.99,
                traceorder="normal",
                font=dict(size=12),
            ),
            autosize=True,
            width=1800,
            height=800,
            template="plotly_dark",
)

st.write(fig)


@st.cache(show_spinner=False)
def train_model(df,df1):
    max_date = max(df["Date"])
#     max_date = nearest_business_day(date.today())
    changepoint_prior_scale = 0.05
    weekly_seasonality = False
    daily_seasonality = False
    monthly_seasonality = True
    yearly_seasonality = True
    changepoints = None
    training_years = 8

    model = Prophet(daily_seasonality=daily_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    changepoints=changepoints)

    # stock_history = data[data['Date'] > (max_date - pd.DateOffset(years=training_years))]
    # stock_history =self.train_data[self.train_data['ds']]>(max_date -pd.DateOffset(years=training_years))
    # model.fit(stock_history)
    stock_history = df[df['Date'] > (max_date - pd.DateOffset(years=training_years))]
    model.fit(stock_history.rename(columns={"Date": "ds", "Close": "y"}))
    future = df1
    future.columns = ['ds']
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    fig1 = plot_plotly(model, forecast)
    fig1.update_xaxes(rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])

    )
    )
    return fig1


# def predict_date(start, end):
#     # start_dt = nearest_business_day(start_dt)
#     # end_dt = nearest_business_day(end_dt)
#     DF = pd.DataFrame({'date': pd.date_range(start=start, end=end)})
#     return DF

st.sidebar.markdown("## Forecasts")
forecast = st.sidebar.container()
sub_columns_1 = forecast.columns(2)

START_f = sub_columns_1[0].date_input("From", value=TODAY, max_value=TODAY + timedelta(days=1))
END_f = sub_columns_1[1].date_input("To", value=TODAY + timedelta(days=3), max_value=TODAY+ timedelta(days=360), min_value=START_f)

START_f = nearest_business_day(START_f)
END_f = nearest_business_day(END_f)

DF = pd.DataFrame({'date': pd.date_range(start=START_f, end=END_f)})
# predict_date(START_f, END_f)

st.sidebar.markdown("## Train")
train_test_forecast_c = st.sidebar.container()

if train_test_forecast_c.button('TRAIN'):
    fig2 = train_model(data, DF)
    st.text("MODEL TRAINED")
else:
    st.text("Press the TRAIN button")



st.write("***")
st.write("###")


def forecast(df, model):
    future = df
    future.columns = ['ds']
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # model.plot(forecast)
    # forecast = model.predict(future)
    return forecast


# st.sidebar.markdown("## FORECAST")
# forecast_c = st.sidebar.container()
# forecast_c.button('FORECAST')
# fcst = forecast(DF, model)


# def forecast_fig(model, forecast):
#     fig = plot_plotly(model, forecast)
#     fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
#         buttons=list([
#             dict(count=1, label="1m", step="month", stepmode="backward"),
#             dict(count=6, label="6m", step="month", stepmode="backward"),
#             dict(count=1, label="YTD", step="year", stepmode="todate"),
#             dict(count=1, label="1y", step="year", stepmode="backward"),
#             dict(step="all")
#         ])
#
#     )
#                      )
#     return fig.show()


st.subheader("Forecasted Data")
# fig2 = forecast_fig(model, forecast)
try:
    fig2.update_layout(
        # width=chart_width,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        legend=dict(
            x=0,
            y=0.99,
            traceorder="normal",
            font=dict(size=12),
        ),
        autosize=True,
        width=1800,
        height=800,
        template="plotly_dark",
    )
    st.write(fig2)
except:
    st.text("press train")