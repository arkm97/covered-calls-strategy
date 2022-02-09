import pandas as pd
import requests
import numpy as np
import datetime
import time
import sys
from alerts import send_alert, send_mailgun, Alert
from scipy.integrate import trapezoid
import predictions
import gloess
import json
import boto3
from botocore.config import Config
import os

client = boto3.client('lambda', region_name='us-east-2', config=Config(read_timeout=120))
s3 = boto3.resource('s3', region_name='us-east-2')

def refresh_access_token(refresh_token, CLIENT_ID, write_status_to_stdout=False):
    """Refresh access tokens for TD API
    """

    RESET_PARAMS = {'grant_type': 'refresh_token', 'refresh_token': refresh_token, 'client_id': CLIENT_ID}
    r = requests.post('https://api.tdameritrade.com/v1/oauth2/token', data=RESET_PARAMS)
    access_token = None
    refresh_token = None
    if r.ok:
        access_token = r.json()['access_token']
        # refresh_token = r.json()['refresh_token']

    if write_status_to_stdout:
        sys.stdout.write('successfully refreshed access token? --> ')
        sys.stdout.write(f'{r.status_code == 200}')
        sys.stdout.write('\n')

    return r.status_code, access_token, refresh_token


def api_request_get(request_url, access_token, params={}, verbose=False):
    """Send get request to TD API"""
    request_response = requests.get(request_url,
                                    headers={'Authorization': 'Bearer ' + access_token},
                                    params=params)

    if verbose:
        sys.stdout.write(f'request status code: {request_response.status_code}')
    return request_response


def api_request_post(request_url, access_token, data, verbose=False):
    """Send post request to TD API"""
    request_response = requests.post(request_url,
                                    headers={'Authorization': 'Bearer ' + access_token,
                                             'Content-type': 'application/json',
                                             'Accept': 'application/json'},
                                    json=data)
    if verbose:
        sys.stdout.write(f'request status code: {request_response.status_code}')
    return request_response


def check_account_status(acct_num, access_token):
    """Run checks to ensure account is in healthy status.
            Check the following:
                account is not closing only restricted
                account is not in call
                account has no reg T call
                account has no maintenance call/requirement
                account has positive buying power

        returns:
            bool account_status_ok = conjunction of all checks
            <request obj> account_status = raw result from account status api request
    """
    account_status_ok = False

    account_status = api_request_get(f'https://api.tdameritrade.com/v1/accounts/{acct_num}', access_token,
                                    params={'fields':['positions', 'orders']})
    # if status API request fails, return
    if account_status.status_code != 200:
        return account_status_ok, account_status

    not_closing_only_restricted = account_status.json()['securitiesAccount']['isClosingOnlyRestricted'] == False

    # initial
    not_in_call_initial = account_status.json()['securitiesAccount']['initialBalances']['isInCall'] == False
    no_regT_call_initial = account_status.json()['securitiesAccount']['initialBalances']['regTCall'] == 0.0
    no_maintenance_call_initial = account_status.json()['securitiesAccount']['initialBalances']['maintenanceCall'] == 0.0
    no_maintenance_req_initial = account_status.json()['securitiesAccount']['initialBalances']['maintenanceRequirement'] < \
                                    account_status.json()['securitiesAccount']['initialBalances']['equity']
    positive_buying_power_initial = account_status.json()['securitiesAccount']['initialBalances']['buyingPower'] > 0.0

    # current
    no_regT_call_current = account_status.json()['securitiesAccount']['currentBalances']['regTCall'] == 0.0
    no_maintenance_call_current = account_status.json()['securitiesAccount']['currentBalances']['maintenanceCall'] == 0.0
    no_maintenance_req_current = account_status.json()['securitiesAccount']['currentBalances']['maintenanceRequirement'] < \
                                    account_status.json()['securitiesAccount']['currentBalances']['equity']
    positive_buying_power_current = account_status.json()['securitiesAccount']['currentBalances']['buyingPower'] > 0.0

    # projected
    not_in_call_projected = account_status.json()['securitiesAccount']['projectedBalances']['isInCall'] == False
    no_regT_call_projected = account_status.json()['securitiesAccount']['projectedBalances']['regTCall'] == 0.0
    no_maintenance_call_projected = account_status.json()['securitiesAccount']['projectedBalances']['maintenanceCall'] == 0.0
    positive_buying_power_projected = account_status.json()['securitiesAccount']['currentBalances']['buyingPower'] > 0

    account_status_ok = not_closing_only_restricted & \
                        not_in_call_initial & no_regT_call_initial & no_maintenance_call_initial & no_maintenance_req_initial & positive_buying_power_initial &\
                        no_regT_call_current & no_maintenance_call_current & no_maintenance_req_current & positive_buying_power_current &\
                        not_in_call_projected & no_regT_call_projected & no_maintenance_call_projected & positive_buying_power_projected

    # if status is not ok, send alert
    if account_status_ok == False:
        send_mailgun(f"ALERT: account status check failed\n\n\n\n\n {account_status.json()}", 'ALERT: account status check failed!')

    return account_status_ok, account_status


def buy_underlying(symbol, quantity, acct_num, access_token):

    order = {"orderType": "MARKET",
      "session": "NORMAL",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [{"instruction": "BUY",
                              "quantity": quantity,
                              "instrument": {"symbol": symbol,"assetType": "EQUITY"}}]}

    order_request = api_request_post(f'https://api.tdameritrade.com/v1/accounts/{acct_num}/orders', access_token, data=order, verbose=True)

    return order_request


def write_call(symbol, acct_num, access_token):

    order = {"orderType": "MARKET",
      "session": "NORMAL",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [{"instruction": "SELL_TO_OPEN",
                              "quantity": 1,
                              "instrument": {"symbol": symbol,"assetType": "OPTION"}}]}

    order_request = api_request_post(f'https://api.tdameritrade.com/v1/accounts/{acct_num}/orders', access_token, data=order, verbose=True)

    return order_request

def convert_log_return_to_price(log_return, current_price):

    return np.exp(log_return * 4.28) * current_price

if __name__ == "__main__":

    # if True, don't actually execute any trades
    dark_mode = True
    alert_subject_line = 'alert from testing'

    if '-live' in sys.argv:
        dark_mode = False
        alert_subject_line = 'LIVE: alert from trading strategy'

    email_alert = Alert(alert_subject_line, 'arkm97@gmail.com')

    # auth_txt = open('auth.txt').read()
    # access_token = auth_txt.split()[2][1:-2]
    # refresh_token = auth_txt.split()[4][1:-2]
    # credentials_txt = pd.read_table('credentials.txt', header=None, engine='python', sep=' = ', skipinitialspace=True, index_col=0).T
    # CLIENT_ID = credentials_txt.CLIENT_ID.values[0][1:-1]
    # acct_num = credentials_txt.acct.astype(int).values[0]
    acct_num = os.environ['TD_ACCOUNT_NUM']
    CLIENT_ID = os.environ['TD_CLIENT_ID']
    refresh_token = os.environ['TD_REFRESH_TOKEN']

    # refresh token
    token_request_status_code, access_token, refresh_token = refresh_access_token(refresh_token, CLIENT_ID)
    if token_request_status_code == 200:
        try:
            # check status
            account_status_ok, account_status = check_account_status(acct_num, access_token)

            # check that market is open:
            equity_market_hours = api_request_get('https://api.tdameritrade.com/v1/marketdata/EQUITY/hours', access_token)
            option_market_hours = api_request_get('https://api.tdameritrade.com/v1/marketdata/OPTION/hours', access_token)
            if equity_market_hours.status_code != 200:
                sys.stdout.write(f'\nequity_market_hours.json() \n --> {equity_market_hours.json()}')
            equity_market_open = equity_market_hours.json()['equity']['EQ']['isOpen']
            option_market_open = option_market_hours.json()['option']['EQO']['isOpen']
            if not equity_market_open:
                sys.stdout.write(f'\nequity market closed')


            ######## if underlying position isn't open, open it ########

            # check if underlying long position is open
            underlying_position_open = False
            if 'positions' in account_status.json()['securitiesAccount'].keys():
                for position in account_status.json()['securitiesAccount']['positions']:
                    if (position['instrument']['symbol'] == 'AAPL') & (position['longQuantity'] == 100):
                        underlying_position_open = True

            sys.stdout.write(f'\nunderlying position open? {underlying_position_open}')

            # equity_market_open = True # comment this when done TESTING
            # open underlying long position
            if equity_market_open & (('positions' not in account_status.json()['securitiesAccount'].keys()) | (underlying_position_open == False)):
                no_day_trade_restriction = account_status.json()['securitiesAccount']['roundTrips'] < 2  # roundTrips < 2 in order to open
                underlying_quote = api_request_get('https://api.tdameritrade.com/v1/marketdata/AAPL/quotes', access_token)
                sufficient_buying_power = account_status.json()['securitiesAccount']['currentBalances']['buyingPower'] > 100 * underlying_quote.json()['AAPL']['mark'] + 500 # $500 buying power buffer

                if account_status_ok & no_day_trade_restriction & sufficient_buying_power:
                    sys.stdout.write(f'\nbuying 100 shares of AAPL @ ${underlying_quote.json()["AAPL"]["askPrice"]}/share')
                    email_alert.add_to_message(f'\nbuying 100 shares of AAPL @ ${underlying_quote.json()["AAPL"]["askPrice"]}/share')
                    if not dark_mode:
                        buy_underlying_response = buy_underlying('AAPL', 100, acct_num, access_token)
                        sys.stdout.write(f'\nresponse status code: {buy_underlying_response.status_code}')
                        if buy_underlying_response.status_code != 201:
                            sys.stdout.write(f'\nfull response: {buy_underlying_response.json()}')
                    # check that position was opened after 1 minute (and again after 5 minutes?)
                    time.sleep(60) # TESTING uncomment this when done testing

                    # check that account status is ok
                    account_status_ok, account_status = check_account_status(acct_num, access_token)
                    sys.stdout.write(f'\naccount status ok post-trade?: {account_status_ok}')
                    email_alert.add_to_message(f'\naccount status ok post-trade?: {account_status_ok}\naccount status: \n\{account_status.json()}')
                    if 'positions' in account_status.json()["securitiesAccount"].keys():
                        sys.stdout.write(f'\npositions: {account_status.json()["securitiesAccount"]["positions"]}')
                        email_alert.add_to_message(f'\npositions: {account_status.json()["securitiesAccount"]["positions"]}')

                    # verify AAPL position is open
                    if 'positions' in account_status.json()['securitiesAccount'].keys():
                        for position in account_status.json()['securitiesAccount']['positions']:
                            if (position['instrument']['symbol'] == 'AAPL') & (position['longQuantity'] == 100):
                                underlying_position_open = True
                else:
                    sys.stdout.write(f'\nNo open position, did not buy underlying.\naccount status ok? {account_status_ok}\nno day trade restriction? {no_day_trade_restriction}\nsufficient buying power? {sufficient_buying_power}')
                    email_alert.add_to_message(f'No open position, did not buy underlying.\naccount status ok? {account_status_ok}\nno day trade restriction? {no_day_trade_restriction}\nsufficient buying power? {sufficient_buying_power}')

            ### volatility model ###
            # pull price history for past 2 months (input for model)
            price_history_request = api_request_get('https://api.tdameritrade.com/v1/marketdata/AAPL/pricehistory', access_token,
                params={'apikey':CLIENT_ID,
                         'periodType': 'month',
                         'period': 2,
                         'frequencyType': 'daily',
                         'frequency': 1})

            price_data = pd.json_normalize(pd.read_json(price_history_request.text).candles)
            price_data.datetime = pd.to_datetime(price_data.datetime, unit='ms').astype(str)
            price_data = price_data.set_index('datetime')

            if '-local' in sys.argv:
                # run model locally
                _predicted_density_4_weeks_later_raw = predictions.predict_density_price(price_data, str(price_data.index.max())[:10], weeks_after=4)

            else:
                # run model on AWS lambda
                request_data = {"price_data": price_data.to_dict(),
                                "date": str(price_data.index.max())[:10],
                                "weeks_before": 4,
                                "weeks_after": 4}

                response = client.invoke(FunctionName='prediction',
                                         InvocationType='RequestResponse',
                                         LogType='Tail',
                                         Payload=json.dumps(request_data))

                response_payload = response['Payload'].read()

                _predicted_density_4_weeks_later_raw = pd.read_json(response_payload)

                print(f'\n\nresponse status code: {response["StatusCode"]}')
                print(f'\nresponse payload: {response_payload[:100]}')


            # store model predictions for later. need this for validation & future improvements
            s3.meta.client.put_object(Body=json.dumps(_predicted_density_4_weeks_later_raw.to_json()), Bucket='automatedtradedata', Key=f'model_output_{datetime.date.today()}')

            ######## if call isn't written ########
            # get AAPL options chain quote
            params={'symbol': 'AAPL',
                    'contractType': 'CALL',
                    'strikeCount': 20,
                    'range': 'OTM',
                    'fromDate': datetime.date.today().strftime('%Y-%m-%d'),
                    'toDate': (datetime.date.today() + datetime.timedelta(weeks=2)).strftime('%Y-%m-%d')}
            options_quote = api_request_get('https://api.tdameritrade.com/v1/marketdata/chains', access_token, params=params)
            exp_times = list(options_quote.json()['callExpDateMap'].keys())
            strike_list = list(options_quote.json()['callExpDateMap'][exp_times[0]].keys())

            # store options quotes for later
            s3.meta.client.put_object(Body=json.dumps(options_quote.json()), Bucket='automatedtradedata', Key=f'options_quote_{datetime.date.today()}')

            # analyze 1st, 2nd, etc OTM calls, write if within risk tolerance
            for n_otm in np.arange(0, 20):

                call_written = False
                # check current positions
                account_status_ok, account_status = check_account_status(acct_num, access_token)
                if 'positions' in account_status.json()['securitiesAccount'].keys():
                    for position in account_status.json()['securitiesAccount']['positions']:
                        if (position['instrument']['assetType'] == 'OPTION') & (position['shortQuantity'] == 1):
                            call_written = True

                # option_market_open = True # TESTING comment this when done testing
                # call_written = False # TESTING comment this when done testing
                # underlying_position_open = True # TESTING comment this when done testing
                if option_market_open & (call_written == False) & (underlying_position_open == True):

                    no_day_trade_restriction = account_status.json()['securitiesAccount']['roundTrips'] < 2

                    options_symbol = options_quote.json()['callExpDateMap'][exp_times[0]][strike_list[n_otm]][0]['symbol']

                    _predicted_density_4_weeks_later = pd.concat([convert_log_return_to_price(_predicted_density_4_weeks_later_raw['x'], options_quote.json()['underlyingPrice']), _predicted_density_4_weeks_later_raw['raw'] / _predicted_density_4_weeks_later_raw['raw'].sum()], axis=1)
                    _call_data = options_quote.json()['callExpDateMap'][exp_times[0]][strike_list[n_otm]][0]
                    opened_at = options_quote.json()['underlyingPrice']
                    strike = _call_data['strikePrice']
                    premium = _call_data['bid']
                    _predicted_density_4_weeks_later['payoff_2w_0otm'] = strike - opened_at - np.array([max(0, strike - i) for i in _predicted_density_4_weeks_later['x']]) + premium
                    _payoff_with_call = trapezoid(_predicted_density_4_weeks_later['payoff_2w_0otm'] * _predicted_density_4_weeks_later['raw'] /  _predicted_density_4_weeks_later['raw'].sum(), _predicted_density_4_weeks_later['x'])
                    _payoff_with_call_variance = trapezoid((_predicted_density_4_weeks_later['payoff_2w_0otm'] - _payoff_with_call) ** 2 * _predicted_density_4_weeks_later['raw'] /  _predicted_density_4_weeks_later['raw'].sum(), _predicted_density_4_weeks_later['x'])

                    _predicted_density_4_weeks_later['payoff_no_call'] =  _predicted_density_4_weeks_later['x'] - opened_at
                    _payoff_without_call = trapezoid(_predicted_density_4_weeks_later['payoff_no_call'] * _predicted_density_4_weeks_later['raw'] /  _predicted_density_4_weeks_later['raw'].sum(), _predicted_density_4_weeks_later['x'])
                    _payoff_without_call_variance = trapezoid((_predicted_density_4_weeks_later['payoff_no_call'] - _payoff_without_call) ** 2 * _predicted_density_4_weeks_later['raw'] /  _predicted_density_4_weeks_later['raw'].sum(), _predicted_density_4_weeks_later['x'])

                    marginal_payoff = _payoff_with_call - _payoff_without_call
                    marginal_payoff_variance = trapezoid((_predicted_density_4_weeks_later['payoff_2w_0otm'] - _predicted_density_4_weeks_later['payoff_no_call'] - marginal_payoff) ** 2 * _predicted_density_4_weeks_later['raw'] /  _predicted_density_4_weeks_later['raw'].sum(), _predicted_density_4_weeks_later['x'])

                    # use model to predict if trade is within risk tolerance
                    within_risk_tolerance = marginal_payoff_variance / (opened_at ** 2) <= .006
                    email_alert.add_to_message(f'payoff for {options_symbol}: {_payoff_with_call:.5f}\npmarginal payoff: {marginal_payoff:.5f}\nmarginal payoff variance scaled: {marginal_payoff_variance / (opened_at ** 2):.5f}\nwithin risk tolerance? {within_risk_tolerance}')

                    # write call
                    if account_status_ok & no_day_trade_restriction & within_risk_tolerance & (strike > opened_at):
                        sys.stdout.write(f'\nwriting call: {options_symbol}')
                        email_alert.add_to_message(f'\nwriting call: {options_symbol}')
                        if not dark_mode:
                            write_call_response = write_call(options_symbol, acct_num, access_token)
                            sys.stdout.write(f'\nresponse status code: {write_call_response.status_code}')
                            if write_call_response.status_code != 201:
                                sys.stdout.write(f'\nfull response: {write_call_response.json()}')
                        # check that position was opened after 1 minute (and again after 5 minutes?)
                        time.sleep(60) # TESTING uncomment this when done testing

                        # check that account status is ok
                        account_status_ok, account_status = check_account_status(acct_num, access_token)
                        sys.stdout.write(f'\naccount status ok post-trade?: {account_status_ok}')
                        email_alert.add_to_message(f'\naccount status ok post-trade?: {account_status_ok}')
                        if 'positions' in account_status.json()['securitiesAccount'].keys():
                            sys.stdout.write(f'\npositions: {account_status.json()["securitiesAccount"]["positions"]}')
                            email_alert.add_to_message(f'\npositions: {account_status.json()["securitiesAccount"]["positions"]}')

        except Exception as e:
            sys.stdout.write(str(e))
            email_alert.add_to_message(f'failed with exception: \n\n{e}')

        email_alert.send_mailgun()
