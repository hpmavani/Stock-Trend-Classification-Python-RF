from stock_indicators import indicators
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from stock_indicators import indicators, Quote
import yfinance as yf
import joblib
# RSI 
# MACD
# WILL %R

RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80

STD_DEVS = 2

def create_quotes_list(ticker): 
    sp500 = yf.Ticker(ticker)
    try:
        HLC_df = sp500.history(period = "max")
        HLC_df = HLC_df.reset_index().dropna()
        quotes_list = [
            Quote(d, o, h, l, c, v)
            for d, o, h, l, c, v
            in zip(HLC_df['Date'], HLC_df['Open'], HLC_df['High'], 
                   HLC_df['Low'], HLC_df['Close'], HLC_df['Volume'])
        ]
    except:
        print("Failed to get ticker data.")
    return quotes_list, HLC_df
        
def compute_features(quotes_list, HLC_df, close_lag_period): 
    features_df = pd.merge(
        bollinger_bands(quotes_list), calc_macd_signal(quotes_list)).merge(
            calc_RSI(quotes_list)).merge(
            calc_trend_strength(quotes_list)
        )
    features_df["Close Trend"] = calc_lagged_close(HLC_df, close_lag_period)
    features_df["Close"] = HLC_df["Close"]
    features_df = features_df.dropna()
    return features_df
        
#Returns a Pandas Series with trend = 1 for price went up, trend = 0 for price went down
def calc_lagged_close(hlc, period): 
    close_prices = hlc["Close"]
    close_trends =  pd.Series(np.nan, index=range(len(hlc)))
    close_trends = (close_prices[period:] > close_prices.shift(period)[period:]).astype(int)
    return close_trends
    
def calc_trend_strength(quotes_list): 
    adx = indicators.get_adx(quotes_list)
    date = pd.Series([i.date for i in adx])
    adx_values = pd.Series([i.adx for i in adx])
    pdi_values = pd.Series([i.pdi for i in adx])
    mdi_values = pd.Series([i.mdi for i in adx])
    adx_df = pd.DataFrame({"Date":date, "ADX":adx_values, "PDI": pdi_values, "MDI": mdi_values})
    adx_df["ADX"] = np.where(adx_df["PDI"] < adx_df["MDI"], (adx_df["ADX"] * -1), adx_df["ADX"])
    return adx_df

def calc_macd_signal(quotes_list): 
    #macd - signal 
    #some other threshold for when macd - signal is decreasing, 
    # the area between the macd and signal is decreasing dA/dt
    macds = indicators.get_macd(quotes_list, 12, 26, 9)
    dates = pd.Series([i.date for i in macds])
    macd_vals = pd.Series([i.macd for i in macds])
    signal_vals = pd.Series([i.signal for i in macds])
    macd_df = pd.DataFrame({"Date": dates, "Macd Values": macd_vals, "Signal Values": signal_vals})
    macd_df["Macd Area Sum"] = (macd_df["Macd Values"] - macd_df["Signal Values"]).abs().rolling(window = 10).sum()

    macd_df["Macd Area Change"] = macd_df["Macd Area Sum"].diff(5)

    #negative indicates compressing  --> possible trend reveral
    #positive indicates expanding --> trend strengthening
    macd_df["Macd Difference"] = (macd_df["Macd Values"] > macd_df["Signal Values"]).astype(int)
    
    return macd_df

#SMA + 2STD
def bollinger_bands(quotes_list): 
    bbs = indicators.get_bollinger_bands(quotes_list, 20, standard_deviations=2)
    dates = pd.Series([i.date for i in bbs])
    smas = pd.Series([i.sma for i in bbs])
    lower_bands = pd.Series([i.lower_band for i in bbs])
    upper_bands = pd.Series([i.upper_band for i in bbs])
    z_scores = pd.Series([i.z_score for i in bbs])
    return pd.DataFrame({"Date": dates, "SMA": smas, "LB":lower_bands, "UP": upper_bands, "Z_Score": z_scores})

#SMA - 2STD
def calc_RSI(quotes_list): 
    rsis = indicators.get_rsi(quotes_list, 14)
    dates = pd.Series([i.date for i in rsis])
    rsi_value = pd.Series([i.rsi for i in rsis])
    rsi_df = pd.DataFrame({"Date": dates, "RSI": rsi_value})
    rsi_df["RSI MEAN"] = rsi_df["RSI"].rolling(window = 20).mean()
    rsi_df["RSI STD"] = rsi_df["RSI"].rolling(window = 20).std()
    rsi_df["RSI Z-Score"] = ((rsi_df["RSI"] - rsi_df["RSI MEAN"])/rsi_df["RSI STD"])
    return rsi_df

def test_train(features_df, split_level): 
    split_index = int(len(features_df) * split_level)
    #Train Set
    xTrain = features_df.iloc[:split_index]
    yTrain = xTrain["Close Trend"]
    #Test Set
    xTest = features_df.iloc[split_index:]
    yTest = xTest["Close Trend"]
    xTrain = xTrain.drop(["Close Trend"], axis = 1)
    xTest = xTest.drop(["Close Trend"], axis = 1)
    return xTrain, yTrain, xTest, yTest

def calc_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="max")
    vix_data = vix_data.drop(["Volume", "Dividends", "Stock Splits"], axis = 1)
    vix_data.reset_index(inplace=True)
    vix_data["Date"] = pd.to_datetime(vix_data["Date"]).dt.tz_localize(None).dt.normalize()
    vix_data = vix_data.set_axis(["Date", "Vix Open", "Vix High", "Vix Low", "Vix Close"], axis=1)
    return vix_data

def predict(ticker, RF): 
    ticker_pred = yf.Ticker(ticker)
    today = ticker_pred.history(period="3mo").reset_index()
    quotes = [Quote(d, o, h, l, c, v) for d, o, h, l, c, v in zip(today['Date'], today['Open'], today['High'], 
                    today['Low'], today['Close'], today['Volume'])]
    latest_df = pd.merge(bollinger_bands(quotes), calc_macd_signal(quotes)).merge(
        calc_RSI(quotes)).merge(
        calc_trend_strength(quotes)
        )
    latest_df["Close Trend"] = calc_lagged_close(today, 5)
    latest_df["Close"] = today["Close"]
    latest_df = latest_df.dropna()
    latest_df = latest_df.tail(1).drop(["Date", "Close", "Close Trend"], axis=1)
    
    y_pred = RF.predict(latest_df)
    return y_pred[0]

    