import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
import pmdarima as pm
from pmdarima import model_selection
import statsmodels.api as sm
import warnings
pd.plotting.register_matplotlib_converters()

other_sales = pd.read_csv('salesOfOther.csv')

other_sales['OrderDate'] = pd.to_datetime(other_sales['OrderDate'])
other_sales['Month'] = other_sales['OrderDate'].dt.to_period('M')
monthly_sales = other_sales.groupby(['Month', 'ProductID'])['OrderQty'].sum().reset_index()
last_order_date = other_sales['OrderDate'].max()
predicted_sales = []
purchaseOrderIncrement = {}

for product_id in monthly_sales['ProductID'].unique():
    product_sales = monthly_sales[monthly_sales['ProductID'] == product_id].copy()
    
    # Convert Month period to datetime for comparison
    product_sales['OrderDate'] = product_sales['Month'].dt.to_timestamp()
    if product_sales['OrderDate'].max() < last_order_date - timedelta(days=30):
        continue
        
    product_sales = product_sales.set_index('Month').sort_index()
    product_sales.reset_index()
    split_point = int(len(product_sales) * 0.8)
    train_data = product_sales.iloc[:split_point]
    test_data = product_sales.iloc[split_point:]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            # print(f'p: {p}, d: {d}, q: {q}')
            model = sm.tsa.SARIMAX(endog=train_data['OrderQty'], order=(2,1,1), disp=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(test_data))
            plt.figure(figsize=(12, 6))
            plt.scatter(range(len(test_data)), test_data['OrderQty'], color='blue', marker='o', label='Actual Sales')
            plt.scatter(range(len(test_data)), forecast.values, color='red', marker='x', label='Predicted Sales')
            plt.title(f'Sales Prediction vs Actual for Product {product_id}')
            plt.xlabel('Time Period (Months)')
            plt.ylabel('Order Quantity')
            plt.legend()
            plt.grid(True)
            plt.show()
            # Comment these lines to disable plotting
        except np.linalg.LinAlgError as e:
            try:
                model = sm.tsa.SARIMAX(endog=product_sales['OrderQty'], order=(2,1,0), disp=False)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(steps=1)
            except:
                print(f"LinAlgError for parameters p={2}, d={1}, q={1}: {e}")
        except Exception as e:
            print(f"Exception for parameters p={2}, d={1}, q={1}: {e}")
            
    
    predicted_sales.append({
        'ProductID': product_id,
        'PredictedOrderQty': forecast.values[0],
        'LastTwoMonthSales': product_sales['OrderQty'].iloc[-2:].mean(),
    })
