import pandas as pd
import requests
import numpy as np
import datetime
import time
import sys

auth_txt = open('auth.txt').read()
access_token = auth_txt.split()[2][1:-2]
refresh_token = auth_txt.split()[4][1:-2]
credentials_txt = pd.read_table('credentials.txt', header=None, engine='python', sep=' = ', skipinitialspace=True, index_col=0).T
CLIENT_ID = credentials_txt.CLIENT_ID.values[0][1:-1]
acct = credentials_txt.acct.astype(int).values[0]

def refresh_access_token(write_status_to_stdout=False):

    RESET_PARAMS = {'grant_type': 'refresh_token', 'refresh_token': refresh_token, 'client_id': CLIENT_ID}
    r = requests.post('https://api.tdameritrade.com/v1/oauth2/token', data=RESET_PARAMS)
    if r.ok:
        access_token = r.json()['access_token']

    if write_status_to_stdout:
        sys.stdout.write('successfully refreshed access token? --> ')
        sys.stdout.write(f'{r.status_code == 200}')
        sys.stdout.write('\n')

    return r.status_code, access_token

def api_request_get(request_url, params={}):
    """Send get request to TD api, refreshing access token if necessary
    """
    request_response = requests.get(request_url,
                                    headers={'Authorization': 'Bearer ' + access_token},
                                    params=params)

    if request_response.status_code == 401:
        token_response, access_token = refresh_access_token()
        if token_response != 200:
            sys.stdout.write('bad access token')
            return
        else:
            request_response = requests.get(request_url,
                                            headers={'Authorization': 'Bearer ' + access_token},
                                            params=params)
            return request_response
    else:
        return request_response

def buy_underlying(quantity):

    order = {"orderType": "MARKET",
      "session": "NORMAL",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [{"instruction": "BUY",
                              "quantity": quantity,
                              "instrument": {"symbol": "AAPL","assetType": "EQUITY"}}]}

    order_request = requests.post(f'https://api.tdameritrade.com/v1/accounts/{acct}/orders',
                                    headers={'Authorization': 'Bearer ' + access_token},
                                    data=order)

    # if access token is denied
    if order_request.status_code == 401:
        token_response, access_token = refresh_access_token()
        if token_response != 200:
            sys.stdout.write('bad access token')
            return 401
        else:
            order_request = requests.post(f'https://api.tdameritrade.com/v1/accounts/{acct}/orders',
                                            headers={'Authorization': 'Bearer ' + access_token},
                                            data=order)
    else:
        return order_request

def write_call(symbol):

    order = {"orderType": "MARKET",
      "session": "NORMAL",
      "duration": "DAY",
      "orderStrategyType": "SINGLE",
      "orderLegCollection": [{"instruction": "SELL",
                              "quantity": 1,
                              "instrument": {"symbol": symbol,"assetType": "OPTION"}}]}

    order_request = requests.post(f'https://api.tdameritrade.com/v1/accounts/{acct}/orders',
                                    headers={'Authorization': 'Bearer ' + access_token},
                                    data=order)

    # if access token is denied
    if order_request.status_code == 401:
        token_response, access_token = refresh_access_token()
        if token_response != 200:
            sys.stdout.write('bad access token')
            return 401
        else:
            order_request = requests.post(f'https://api.tdameritrade.com/v1/accounts/{acct}/orders',
                                            headers={'Authorization': 'Bearer ' + access_token},
                                            data=order)
    else:
        return order_request

def check_account_status():
        ## checks to ensure account is in healthy status:
        account_status = api_request_get(f'https://api.tdameritrade.com/v1/accounts/{acct}',
                                        params={'fields':['positions', 'orders']})

        not_closing_only_restricted = account_status.json()['securitiesAccount']['isClosingOnlyRestricted'] == False

        # initial
        not_in_call_initial = account_status.json()['securitiesAccount']['initialBalances']['isInCall'] == False
        no_regT_call_initial = account_status.json()['securitiesAccount']['initialBalances']['regTCall'] == 0.0
        no_maintenance_call_initial = account_status.json()['securitiesAccount']['initialBalances']['maintenanceCall'] == 0.0
        no_maintenance_req_initial = account_status.json()['securitiesAccount']['initialBalances']['maintenanceRequirement'] == 0.0
        positive_buying_power_initial = account_status.json()['securitiesAccount']['initialBalances']['buyingPower'] > 0.0

        # current
        no_regT_call_current = account_status.json()['securitiesAccount']['currentBalances']['regTCall'] == 0.0
        no_maintenance_call_current = account_status.json()['securitiesAccount']['currentBalances']['maintenanceCall'] == 0.0
        no_maintenance_req_current = account_status.json()['securitiesAccount']['currentBalances']['maintenanceRequirement'] == 0.0
        positive_buying_power_current = account_status.json()['securitiesAccount']['currentBalances']['buyingPower'] > 0.0

        # projected
        not_in_call_projected = account_status.json()['securitiesAccount']['projectedBalances']['isInCall'] == False
        no_regT_call_projected = account_status.json()['securitiesAccount']['projectedBalances']['regTCall'] == 0.0
        no_maintenance_call_projected = account_status.json()['securitiesAccount']['projectedBalances']['maintenanceCall'] == 0.0
        positive_buying_power_projected = account_status.json()['securitiesAccount']['currentBalances']['buyingPower'] > 0

        #current buying power
        buying_power = account_status.json()['securitiesAccount']['currentBalances']['buyingPower']


        account_status_ok = not_closing_only_restricted & \
                            not_in_call_initial & no_regT_call_initial & no_maintenance_call_initial & no_maintenance_req_initial & positive_buying_power_initial &\
                            no_regT_call_current & no_maintenance_call_current & no_maintenance_req_current & positive_buying_power_current &\
                            not_in_call_projected & no_regT_call_projected & no_maintenance_call_projected & positive_buying_power_projected

        return account_status_ok, account_status

if __name__ == "__main__":

    _, access_token = refresh_access_token(write_status_to_stdout=True)

    account_status_ok, account_status = check_account_status()

    sys.stdout.write('account status ok? --> ')
    sys.stdout.write(f'{account_status_ok}')
    sys.stdout.write('\n')

    # todo: check that market is open:
    equity_market_hours = api_request_get('https://api.tdameritrade.com/v1/marketdata/{EQUITY}/hours')
    option_market_hours = api_request_get('https://api.tdameritrade.com/v1/marketdata/{OPTION}/hours')

    # main trading logic
    # check current positions
    account_status_ok, account_status = check_account_status()

    ######## if AAPL position isn't open, open it ########
    aapl_position_open = False
    if (account_status.json().keys().__contains__('positions') == True):
        for position in account_status.json()['securitiesAccount']['positions']:
            if (position['instrument']['symbol'] == 'AAPL') & (position['longQuantity'] == 100):
                aapl_position_open = True

    if (account_status.json().keys().__contains__('positions') == False) | (aapl_position_open == False):

        ok_to_open_position = True

        # check that roundTrips < 2
        if account_status.json()['securitiesAccount']['roundTrips'] >= 2:
            ok_to_open_position = False

        # get AAPL quote
        aapl_quote = api_request_get('https://api.tdameritrade.com/v1/marketdata/{AAPL}/quotes')

        # check that buying power can support opening the new position (with some buffer)
        buffer = 500.
        if account_status.json()['securitiesAccount']['currentBalances']['buyingPower'] < 100 * aapl_quote.json()['askPrice'] + buffer:
            ok_to_open_position = False

        # TODO: if model score says > x% chance underlying lose more than usual premium by next week, don't open the position

        # open position
        if account_status_ok & ok_to_open_position:
            buy_underlying_response = buy_underlying(100)
            sys.stdout.write(f'\nbuying 100 shares of AAPL\ @ ${aapl_quote.json()['askPrice']}/share')
            sys.stdout.write(f'\nresponse status code: {buy_underlying_response.status_code}')
            sys.stdout.write(f'\nfull response: {buy_underlying_response.json()}')
            # check that position was opened after 1 minute (and again after 5 minutes?)
            time.sleep(60)

            # check that account status is ok
            account_status_ok, account_status = check_account_status()
            sys.stdout.write(f'\naccount status ok post-trade?: {account_status_ok}')
            if (account_status.json().keys().__contains__('positions') == True):
                sys.stdout.write(f'\npositions: {account_status.json()['securitiesAccount']['positions']}')

            # verify AAPL position is open
            if (account_status.json().keys().__contains__('positions') == True):
                for position in account_status.json()['securitiesAccount']['positions']:
                    if (position['instrument']['symbol'] == 'AAPL') & (position['longQuantity'] == 100):
                        aapl_position_open = True


    ######## if call isn't written ########
    call_written = False
    # check current positions
    account_status_ok, account_status = check_account_status()
    if (account_status.json().keys().__contains__('positions') == True):
        for position in account_status.json()['securitiesAccount']['positions']:
            if (position['instrument']['assetType'] == 'OPTION') & (position['shortQuantity'] == 1):
                call_written = True

    if (call_written == False) | (aapl_position_open == True):

        ok_to_write_call = True

        # check that roundTrips < 2
        if account_status.json()['securitiesAccount']['roundTrips'] >= 2:
            ok_to_write_call = False

        # get AAPL options chain quote
        params={'symbol': 'AAPL',
                'contractType': 'CALL',
                'strikeCount': 1,
                'range': 'OTM',
                'fromDate': datetime.date.today().strftime('%Y-%m-%d'),
                'toDate': (datetime.date.today() + datetime.timedelta(weeks=1)).strftime('%Y-%m-%d')}
        options_quote = api_request_get('https://api.tdameritrade.com/v1/marketdata/chains', params=params)

        # TODO: if model score says > x% chance underlying GAINS more than usual premium by next week, don't write the call

        # write call
        if account_status_ok & ok_to_write_call:
            write_call_response = write_call(options_quote.json()['symbol'])
            sys.stdout.write(f'\nwriting call: {options_quote.json()['symbol']}')
            sys.stdout.write(f'\nresponse status code: {write_call_response.status_code}')
            sys.stdout.write(f'\nfull response: {write_call_response.json()}')
            # check that position was opened after 1 minute (and again after 5 minutes?)
            time.sleep(60)

            # check that account status is ok
            account_status_ok, account_status = check_account_status()
            sys.stdout.write(f'\naccount status ok post-trade?: {account_status_ok}')
            if (account_status.json().keys().__contains__('positions') == True):
                sys.stdout.write(f'\npositions: {account_status.json()['securitiesAccount']['positions']}')
