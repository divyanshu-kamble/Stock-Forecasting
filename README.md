# Stock-Forecasting
This is a repository which has python programme to showcase stock forecast. It is a prediction app which was built for prediction of the stock data, this will help the user to make better descisions for investment.

## Interface and Forecasting

![stock app1](https://user-images.githubusercontent.com/97751164/185205432-5f6e4e5b-a140-4e01-94e0-68df166f4491.PNG)

![stock app2](https://user-images.githubusercontent.com/97751164/185205764-0629cac9-fd56-4397-8aa8-4ca03893fe27.PNG)

## Introduction

Stock prediction web app was bulit using the time series model(FBprophet) the data wasgathered from yfinance api, yfinance has all the stock data needed it also provides daily as well as hourly data. The aim was to predictthe stock forecasting in real time, with the stockthat you wish to predict, currently it only containsa list of stock but will modlfy in future for feeding the stock name.

The model trains the data in real time and shows prediction this eliminates the use for continuosly modifying the dataas it changes daily, also the training time is ververy minimum. Fb prophet was used as a time series model.

Plotly was used to show interactive graphsas it improves graphical functionality over matplotlib.

## Web app and Docker

Streamlit was used to create a web app framework, itis fast and easy to implement and has various functionality. Lastly Docker was used to deploy the web app.



## How to Run
`docker run -it -p 8501:8501 divyanshuk/stockapp:v1.3`

Open this on your browser

`curl http://localhost:8501/`

## How to Build
`docker build -t divyanshuk/stockapp:v1.3 .`

## Tools used

Python - The language of choice for the project 
Numpy - NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
Plotly - It is used to visualize the stock data in web app with ineractive model.
Pandas - Used to manipulate the stock data from yfinance and create the data for training
Yfinance - Used to gather the stock data from web.
FB prophet -Is an open-source algorithm for generating time-series models that will train the stock data and forecast it.
Seaborn - Seaborn is a library that uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.
Streamlit - Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time.
Docker - Docker is used for deployment of the streamlit web app.
