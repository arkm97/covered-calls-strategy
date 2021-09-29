## What is this project?

This is a simple application I designed to automate a covered call trading strategy. A single script pulls quotes and handles orders using TD Ameritrade's API. I do supporting analysis (backtesting, model development) offline to keep the script as lightweight as possible.

## What is the trading stratetgy?

In short: write OTM calls on AAPL while long the underlying 100 shares. If the strategy is successful, the premia gained from the sold calls make up for the limited upside imposed by the calls. The strategy (roughly) shorts volatility of the underlying shares. It is most profitable when expected future volatility (at the time the call is written) is much higher than the volatility realized at expiration.

### a quick example

Buy 100 shares at the market.  Write (sell) an OTM call and collect the premium.

if the sold option expires out of the money (assuming no early exercise)...

...I keep the shares and the premium. Returns after expiration are linear with the underlying (until a new call is written on the same underlying shares).

if the option expires in the money...

...I still keep the premium, but miss the upside (100 x (share price - strike)) on the underlying shares when the contract is assigned to me. The underlying position can be reopened and another call can be written.

### more details
Most of the design choices in the strategy come from capital and computing resource constraints. While premia tend to increase with time to expiration, I find writing calls 2-3 weeks to expiration gives a good balance of return and flexibility (i.e. not locked into a contract with many months to expiration; since I can only buy about 100 shares at once with my own money, I can only write one contract at a time). I tend to write calls as close to the money as I can: given the short timeframe, these are generally the only contracts with premia large enough to make the trade viable. I use AAPL as the underlying security because I'm mostly bullish long-term (after all, I am exposed to big decreases in the underlying shares). Hypothetically the strategy could be closer to market-neutral with the correct short position(s), but my brokerage limits these types of positions.

## How does the strategy perform?
