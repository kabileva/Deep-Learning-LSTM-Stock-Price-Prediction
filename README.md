## Simple LSTM model for stock price prediction
The minute data for around 6 months of KOSPI top-50 stocks was used for training.

### Downloading the data
The data can be downloaded with scrapto.py in kospi-scraper dir.
*Requirements: Windows 32bit, Kiwoom Open Api*

### Training
- StockPricePredictionNoFeatures.py - for training based on OHLC average.
- StockPricePredictionFeatures.py - for training based on OHLC average and volume data,

### Trading simulation
trading_simulation.py - for combining the data obtained by training with different parameters
and simulating the trading.
Currently there are 2 trading strategies:
1. Buy when rise (STRATEGY = 'buy_when_rise'): buy when is predicted to rise, sell when predicted to fall and profit is bigger than SELL_WHEN_PROFIT.
*Try to change SELL_WHEN_PROFIT and BUY_WHEN_PREDICTED to improve the result.*
2. Buy when rise is 0.7% (STRATEGY = 'buy_when_0.7pct'):
- Buy when expected rise is 0.7%
- If rise: sell when increase by 7%
- If fall: sell if fall under -1.5%, sell if recovers to 0.34%

### Initial parameters:
epochs = 5
optimizer = 'adagrad'
loss = 'mean_squared_error'
REGULARIZATION = False

### The result data is obtained using the following parameters:
- result-data-12.24-adagrad/:
 - epochs = 5
 - optimizer = 'adagrad'
 - loss = 'mean_squared_error'
 - REGULARIZATION = False

*trained based on OHLC average using StockPricePredictionNoFeatures.py*

- result-data-12.24-volume-adagrad/:
 - epochs = 5
 - optimizer = 'adagrad'
 - loss = 'mean_squared_error'
 - REGULARIZATION = False

*trained based on OHLC average and volume using StockPricePredictionFeatures.py*

- result-data-12.24-adam/:
 - epochs = 5
 - optimizer = 'adam'
 - loss = 'mean_squared_error'
 - REGULARIZATION = False

*trained based on OHLC average using StockPricePredictionNoFeatures.py*

- result-data-12.24-volume-adam/:
 - epochs = 5
 - optimizer = 'adam'
 - loss = 'mean_squared_error'
 - REGULARIZATION = False

*trained based on OHLC average and volume using StockPricePredictionFeatures.py*

- result-data-12.24-sgd/:
 - epochs = 5
 - optimizer = 'sgd'
 - loss = 'mean_squared_error'
 - REGULARIZATION = False

*trained based on OHLC average using StockPricePredictionNoFeatures.py*

- result-data-12.24-volume-sgd/:
 - epochs = 5
 - optimizer = 'sgd'
 - loss = 'mean_squared_error'
 - REGULARIZATION = False

*trained based on OHLC average and volume using StockPricePredictionFeatures.py*

### Combining the data
It was found out that different parameters work best for different stocks.
Therefore, the data obtained by training with different parameters is combined to *combined_log.csv* 
using trading_simulation.py

### Result Achieved
- NRMSE (normalized root mean squared error) on TEST set is around 0.8% if the data obtained by training with different parameters is combined.
- NRMSE (normalized root mean squared error) on TRAINING set is around 0.4% if the data obtained by training with different parameters is combined.

The overfitting observed by comparing test NRMSE and train NRMSE most probably occured because the data set is relatively small. However, possibly it can be reduced by adding regularization.

- Average daily profit can be 2-5%, depending on the trading strategy used. 
