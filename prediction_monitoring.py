import pandas as pd
import numpy as np
import github3
import io
import boto3
import json
import requests
import matplotlib.pyplot as plt
import datetime
import sys
import os

### setup ###
CLIENT_ID = os.environ['TD_CLIENT_ID']
refresh_token = os.environ['TD_REFRESH_TOKEN']
account_num = os.environ['TD_ACCOUNT_NUM']

s3 = boto3.resource('s3', region_name='us-east-2')


gh = github3.login(token=os.environ['GITHUB_ACCESS_TOKEN'])
repository = gh.repository('arkm97', 'covered-calls')

def convert_log_return_to_price(log_return, current_price):

    return np.exp(log_return * 4.28) * current_price

plt.rcParams['figure.figsize'] = 10, 6

### refresh access token ###
RESET_PARAMS = {'grant_type': 'refresh_token', 'refresh_token': refresh_token, 'client_id': CLIENT_ID}
r = requests.post('https://api.tdameritrade.com/v1/oauth2/token', data=RESET_PARAMS)

if r.ok:
    access_token = r.json()['access_token']

### get price history ###
r = requests.get('https://api.tdameritrade.com/v1/marketdata/AAPL/pricehistory',
                 headers={'Authorization':'Bearer ' + access_token},
                 params={'apikey':CLIENT_ID,
                         'periodType': 'day',
                         'period': 10,
                         'frequencyType': 'minute',
                         'frequency': 30})

df = pd.DataFrame(r.json()['candles']).astype({'datetime': 'datetime64[ms]'})
df = df.set_index('datetime')

### get model prediction ###
use_backfilled = True
bucket_objects = s3.Bucket('automatedtradedata').objects.all()

model_outputs = [] # list of dates associated with each model output object in the bucket
options_quotes = []
for i in bucket_objects:
    if i.key[:1] == 'm':
        model_outputs.append(i.key[13:])
    else:
        options_quotes.append(i.key[14:])

if use_backfilled:
    model_outputs = list(pd.DataFrame({'first_10': pd.Series(model_outputs).str[:10], 'full': pd.Series(model_outputs)}).groupby('first_10').max()['full'])

prediction_date = (datetime.date.today() - datetime.timedelta(days=10))
prediction_date = model_outputs[np.argmin(np.abs(np.array([pd.to_datetime(i[:10]) for i in model_outputs]) - np.array(pd.to_datetime(prediction_date))))]

options_data = json.loads(s3.meta.client.get_object(Bucket='automatedtradedata', Key=f'options_quote_{prediction_date[:10]}')['Body'].read())
options_data['underlyingPrice']

data = s3.meta.client.get_object(Bucket='automatedtradedata', Key=f'model_output_{prediction_date}')['Body'].read()
df_prediction = pd.read_json(json.loads(data))
df_prediction['x_converted'] = convert_log_return_to_price(df_prediction['x'], options_data['underlyingPrice'])


##### Prediction monitoring #####
### create plot ###
fig, ax = plt.subplots()
ax2 = ax.twinx()
df[prediction_date:(pd.to_datetime(prediction_date) + pd.Timedelta(9, 'd')).strftime('%Y-%m-%d')]['open'].hist(ax=ax, label=f'price history since {prediction_date}\n(grain: 1 sample per 30 min)', bins=50, alpha=.5)
df[pd.to_datetime(prediction_date) + pd.Timedelta(9, 'd'):(pd.to_datetime(prediction_date) + pd.Timedelta(9, 'd')).strftime('%Y-%m-%d')]['open'].hist(ax=ax, bins=20, label=f'prices on {(pd.to_datetime(prediction_date) + pd.Timedelta(9, "d")).strftime("%Y-%m-%d"):10}\n(grain: 1 sample per 30 min)', alpha=.5, color='C2')
# ax2.plot([options_data['underlyingPrice'], options_data['underlyingPrice']], [df_prediction.raw.min(), df_prediction.raw.max()], color='C3', label='price at time of prediction')
df_prediction.plot(x='x_converted', y=['raw'], ax=ax2, color=['C1', 'C2'])

ax.legend(loc=2)
ax2.legend([f'model output on {prediction_date}'], loc=1)
ax.set_xlim(.9 * df.open.min(), 1.1 * df.open.max())
ax2.set_ylim(df_prediction.raw.min(), )
ax.grid(False)
ax.set_yticks([])
ax2.set_yticks([])
ax.set_ylabel('# of samples @ given price')
ax2.set_ylabel('predicted density @ given price\n(over observation window)')
ax.set_title('Predicted Prices Over Next 10 Days\nVS\nObserved Prices Over Next 10 Days', fontsize=14)

buffer = io.BytesIO()
fig.savefig(buffer, format='png', bbox_inches='tight')

### upload image to github pages site ###
contents = repository.file_contents('images/prediction_vs_outcome.png')
push_status = contents.update('update image', buffer.getvalue())
sys.stdout.write(str(push_status))


##### Transaction monitoring #####
transactions_request = requests.get(f'https://api.tdameritrade.com/v1/accounts/{account_num}/transactions',
                 headers={'Authorization':'Bearer ' + access_token},)

df_transactions = pd.DataFrame(transactions_request.json())

options_written = df_transactions.loc[(df_transactions.type=='TRADE') & (df_transactions.description=='SELL TRADE')]
options_assigned = df_transactions.loc[(df_transactions.type=='TRADE') & (df_transactions.transactionSubType=='OA')]
buy_underlying = df_transactions.loc[(df_transactions.type=='TRADE') & (df_transactions.transactionSubType=='BY')]

account_request = requests.get(f'https://api.tdameritrade.com/v1/accounts/{account_num}',
                                        headers={'Authorization':'Bearer ' + access_token},)

liquidation_value_now = account_request.json()['securitiesAccount']['currentBalances']['liquidationValue']


price_history_past_year = requests.get('https://api.tdameritrade.com/v1/marketdata/AAPL/pricehistory',
                 headers={'Authorization':'Bearer ' + access_token},
                 params={'apikey':CLIENT_ID,
                         'periodType': 'year',
                         'period': 1,
                         'frequencyType': 'daily',
                         'frequency': 1})

prices_past_year = pd.DataFrame(price_history_past_year.json()['candles']).astype({'datetime': 'datetime64[ms]'})
prices_past_year = prices_past_year.set_index('datetime')

price_history_past_month = requests.get('https://api.tdameritrade.com/v1/marketdata/AAPL/pricehistory',
                 headers={'Authorization':'Bearer ' + access_token},
                 params={'apikey':CLIENT_ID,
                         'periodType': 'month',
                         'period': 1,
                         'frequencyType': 'daily',
                         'frequency': 1})

prices_past_month = pd.DataFrame(price_history_past_month.json()['candles']).astype({'datetime': 'datetime64[ms]'})
prices_past_month = prices_past_month.set_index('datetime')

df_transactions = pd.merge_asof(df_transactions.astype({'transactionDate': 'datetime64[ns]'}).sort_values('transactionDate'),
              prices_past_year[['close']].sort_index(),
              left_on='transactionDate',
              right_index=True)

df_transactions['cash_value'] = df_transactions.netAmount.cumsum()
df_transactions['underlying_position'] = np.nan
df_transactions.loc[df_transactions.transactionSubType=='BY', 'underlying_position'] = 100
df_transactions.loc[df_transactions.description=='OPTION ASSIGNMENT', 'underlying_position'] = 0
df_transactions['underlying_position'] = df_transactions['underlying_position'].fillna(method='ffill')
df_transactions['long_stock_value'] = df_transactions['underlying_position'] * df_transactions['close']
df_transactions['portfolio_value'] = df_transactions['cash_value'].fillna(0) + df_transactions['long_stock_value'].fillna(0)
df_transactions['portfolio_value'] = df_transactions['portfolio_value'] + liquidation_value_now - df_transactions['portfolio_value'].iloc[-1]


days_since_strategy_start = (pd.Timestamp('today') - df_transactions.loc[df_transactions.transactionSubType=='BY'].transactionDate.min()).days
cc_gain_since_start = (df_transactions.portfolio_value.iloc[-1] / df_transactions.portfolio_value.iloc[0]) ** (365 / days_since_strategy_start) - 1
underlying_on_strategy_start = prices_past_year.loc[prices_past_year.index.date == df_transactions.loc[df_transactions.transactionSubType=='BY'].transactionDate.min().date()]
underlying_gain_since_start = (prices_past_year.close.iloc[-1] / underlying_on_strategy_start.close.values[0]) ** (365 / days_since_strategy_start) - 1

days_since_10_ago = (pd.Timestamp('today') - df.index[0]).days
cc_gain_since_10_ago = (df_transactions.portfolio_value.iloc[-1] / df_transactions.iloc[np.argmin(abs(df_transactions.transactionDate - df.index[0]))].portfolio_value) ** (365 / days_since_10_ago) - 1
underlying_gain_since_10_ago = (df.iloc[-1].close / df.iloc[0].close) ** (365 / days_since_10_ago) - 1

days_since_1_month_ago = (pd.Timestamp('today') - prices_past_month.index[0]).days
cc_gain_since_1_month_ago = (df_transactions.portfolio_value.iloc[-1] / df_transactions.iloc[np.argmin(abs(df_transactions.transactionDate - prices_past_month.index[0]))].portfolio_value) - 1# ** (365 / days_since_1_month_ago) - 1
underlying_gain_since_1_month_ago = (prices_past_month.iloc[-1].close / prices_past_month.iloc[0].close) - 1 # ** (365 / days_since_10_ago) - 1

fig, ax = plt.subplots()
prices_past_month.open.plot(ax=ax, alpha=0, ls='--', marker='.')
plt.errorbar(prices_past_month.index, prices_past_month.open, yerr=prices_past_month['high'] - prices_past_month['low'], elinewidth=.5, marker='.', ls='', alpha=.5, color='C0')
for written in options_written.iterrows():
    try:
        written_info = written[1]['transactionItem']
        transaction_date = written[1]['transactionDate'][:10]
        if pd.to_datetime(transaction_date) > prices_past_month.index.min():

            ax.text(x=transaction_date, y=5 + float(written_info['instrument']['symbol'].split('C')[1]), s=f"sold:\n{written_info['instrument']['description']}\npremium: ${100 * written_info['price']}", color='C1', fontsize=7)

            # arrow pointing to exp time/price
            time_to_exp = (pd.to_datetime(written_info['instrument']['optionExpirationDate'][:10], infer_datetime_format=True) + pd.Timedelta(14, 'h') - pd.to_datetime(transaction_date, infer_datetime_format=True))
            ax.arrow(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=float(written_info['instrument']['symbol'].split('C')[1]), dx=time_to_exp.days + time_to_exp.seconds/3600/24, dy=0, ls=':', alpha=0.5, color='C1', head_width=0, length_includes_head=True, head_length=0)
            ax.scatter(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=float(written_info['instrument']['symbol'].split('C')[1]), alpha=0.5, color='C1', marker='o', s=100)
            ax.scatter(x=pd.to_datetime(written_info['instrument']['optionExpirationDate'][:10], infer_datetime_format=True) + pd.Timedelta(24, 'h'), y=float(written_info['instrument']['symbol'].split('C')[1]), alpha=0.5, color='C1', marker='o', s=100, facecolors='none')

            # arrow pointing to time of transaction
            ax.arrow(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=4 + float(written_info['instrument']['symbol'].split('C')[1]), dx=0, dy=df.open.min() - 9 - float(written_info['instrument']['symbol'].split('C')[1]), ls=':', alpha=0.5, color='C1')

    except:
        continue

for assigned in options_assigned.iterrows():
    try:
        assigned_info = assigned[1]['transactionItem']
        transaction_date = assigned[1]['transactionDate'][:10]
        if pd.to_datetime(transaction_date, infer_datetime_format=True) > df.index.min():

            ax.text(x=transaction_date, y=-5 + float(assigned_info['price']), s=f"assigned to sell 100 shares {assigned_info['instrument']['symbol']}\n @ ${assigned_info['price']} / share", color='C3', fontsize=7)

            # arrow pointing to time of transaction
            ax.arrow(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=4 + float(assigned_info['price']), dx=0, dy=df.open.min() - 9 - float(assigned_info['price']), ls=':', alpha=0.5, color='C3')
            ax.scatter(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=float(assigned_info['price']), alpha=0.5, color='C3', marker='o', s=100)


    except:
        continue

for trade in buy_underlying.iterrows():
    try:
        underlying_trade_info = trade[1]['transactionItem']
        transaction_date = trade[1]['transactionDate'][:10]
        if pd.to_datetime(transaction_date, infer_datetime_format=True) > df.index.min():

            ax.text(x=pd.to_datetime(transaction_date, infer_datetime_format=True) - pd.Timedelta(3, 'd'), y=-5 + float(underlying_trade_info['price']), s=f"bought:\n100 shares {underlying_trade_info['instrument']['symbol']}\n @ ${underlying_trade_info['price']} / share", color='C2', fontsize=7)
            ax.scatter(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=-5 + float(underlying_trade_info['price']), s=100, marker='o', color='C2', alpha=.5)

            # arrow pointing to time of transaction
            ax.arrow(x=pd.to_datetime(transaction_date, infer_datetime_format=True) + pd.Timedelta(10, 'h'), y=float(underlying_trade_info['price']), dx=0, dy=df.open.min() - 9 - float(underlying_trade_info['price']), ls=':', alpha=0.5, color='C2')

    except:
        continue

# strategy performance vs underlying alone
fig.text(x=.13, y=.85, s=f"Covered call strategy:  {100 * cc_gain_since_1_month_ago:.2f}% (absolute) since {str(prices_past_month.index.min())[:10]},  {100 * cc_gain_since_start:.2f}% (annualized) since start ({str(buy_underlying.transactionDate.min())[:10]})", alpha=.5)
fig.text(x=.1365, y=.82, s=f"     Underlying alone:  {100 * underlying_gain_since_1_month_ago:.2f}% (absolute) since {str(prices_past_month.index.min())[:10]},  {100 * underlying_gain_since_start:.2f}% (annualized) since start ({str(buy_underlying.transactionDate.min())[:10]})", alpha=.5)

ax.set_xlim(prices_past_month.index.min() - pd.Timedelta(2, 'd'), prices_past_month.index.max() + pd.Timedelta(5, 'd'))
ax.set_ylim(prices_past_month.open.min() - 5, prices_past_month.open.max() + 12)
ax.set_xlabel('')
ax.set_ylabel('price\n(grain: 1 sample per day)')
ax.set_title('Recent Transaction History', fontsize=14)

buffer = io.BytesIO()
import pdb; pdb.set_trace()
fig.savefig(buffer, format='png', bbox_inches='tight')

### upload image to github pages site ###
contents = repository.file_contents('images/transaction_history.png')
push_status = contents.update('update image', buffer.getvalue())
sys.stdout.write(str(push_status))

##### Strategy performance monitoring #####
fig, ax = plt.subplots()

(df_transactions.set_index('transactionDate').portfolio_value / df_transactions.loc[df_transactions.transactionDate == pd.to_datetime(buy_underlying.transactionDate.min(), infer_datetime_format=True)].portfolio_value.values).plot(ls=':', marker='o', ax=ax, color='C1', label='Portfolio value')
plt.errorbar(prices_past_year.index, prices_past_year.open / underlying_on_strategy_start.close.values[0], yerr=(prices_past_year['high'] - prices_past_year['low']) / underlying_on_strategy_start.close.values[0], elinewidth=.5, marker='.', ls='', alpha=.5, color='C0', label='Underlying')

plt.xlim(buy_underlying.transactionDate.min(), pd.Timestamp('today'))

plt.yticks(ax.get_yticks(), [f'{100 * (i - 1):.0f}%' for i in ax.get_yticks()])
for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), ['k' if i >= 1 else 'C3' for i in ax.get_yticks()]):
    ticklabel.set_color(tickcolor)

plt.title('All-Time Strategy Performance')
plt.ylabel('All-time gain/loss\n(absolute)')
plt.xlabel('')
plt.grid(axis='y', ls=':')

plt.legend()

buffer = io.BytesIO()
fig.savefig(buffer, format='png', bbox_inches='tight')

### upload image to github pages site ###
contents = repository.file_contents('images/strategy_performance.png')
push_status = contents.update('update image', buffer.getvalue())
sys.stdout.write(str(push_status))
