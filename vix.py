"""
VIX Calculation

Assumptions in the white paper example:

- Calculation is being performed at 9:46am EST
- Near-term options are "standard" SPX options with 25 DTE
- Next-term options are PM-settled Weekly SPX options with 32 DTE
- Standard options expire at open of trading on open of Settlement day
- Weekly options expire at 4pm of settlement date

- Risk free rates are based on US Treasury CMT rates, with cubic spline interp
- Near-term rate is 0.0305%
- Next-term rate is 0.0286%

In order to make this calculation extendable to real life, it is assumed that
the current date (quote date) is 1/27/2020. This makes the near-term
expiration date 2/21/2020, and the next-term expiration date 2/28/2020.

"""

import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline


def map_exact_expiration(option_df):
    """
    Map exact times to option series expiration types.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    exp_map = {'standard': pd.to_timedelta('09:30:00'),
               'weekly': pd.to_timedelta('16:00:00')}

    expiration_time = option_df['expiration_type'].map(exp_map)
    option_df['expiration'] = option_df['expiration'] + expiration_time

    return option_df


def calculate_mid(option_df):
    """
    Calculate the mid prices for puts and calls.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    option_df['mid'] = (option_df['ask'] + option_df['bid']) / 2

    return option_df


def rough_forwards(option_df):
    """
    Find the rough forward strike i.e. minimum abs diff for put and call.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    unstacked = option_df.set_index(['expiration', 'strike', 'option_type'])
    unstacked = unstacked['mid'].unstack().reset_index()
    unstacked['diff'] = abs(unstacked['CALL'] - unstacked['PUT'])
    forwards = unstacked.loc[unstacked.groupby('expiration')['diff'].idxmin()]

    return forwards


def granular_forwards(forward_df, yield_df, quote_date):
    """
    Calculate implied forward by: f = rough strike + e^rt * (call - put)

    :param forward_df: pd.DataFrame with the rough forward prices
    :param yield_df: pd.DataFrame of the risk free rate curve
    :param quote_date: datetime-like
    :return: pd.DataFrame
    """

    forward_df['dte'] = forward_df['expiration'].dt.date - quote_date.date()

    # Calculate the minutes to expiry
    forward_df['mte'] = forward_df['expiration'] - quote_date
    forward_df['mte'] = forward_df['mte'] / pd.Timedelta('1 minute')

    ###########################################################################
    # Manual fix to validate against white paper
    # forward_df.loc[forward_df.index[0], 'mte'] = 854 + 510 + 34560
    # forward_df.loc[forward_df.index[1], 'mte'] = 854 + 900 + 44640
    ###########################################################################

    # Calculate the years to expiry
    forward_df['ttm'] = forward_df['mte'] / (60 * 24 * 365)

    # Interpolate the appropriate risk free rate for each expiry
    interp = InterpolatedUnivariateSpline(
        x=yield_df['ttm'], y=yield_df['yield'], k=3)

    forward_df['rfr'] = forward_df['ttm'].apply(interp)

    ###########################################################################
    # Manual fix to validate against white paper
    # forward_df.loc[forward_df.index[0], 'rfr'] = 0.000305
    # forward_df.loc[forward_df.index[1], 'rfr'] = 0.000286
    ###########################################################################

    # Calculate the implied forward
    spread = forward_df['CALL'] - forward_df['PUT']
    discount = np.exp(forward_df['rfr'] * forward_df['ttm'])
    forward_df['fwd_price'] = forward_df['strike'] + discount * spread

    return forward_df


def find_k_zero(option_df, forward_df):
    """
    Determine K0 for each expiry - the strike directly below the actual fwd

    :param option_df: pd.DataFrame of the option data
    :param forward_df: pd.DataFrame with the forward prices
    :return: pd.DataFrame
    """

    fwd_dict = forward_df.set_index('expiration')['fwd_price'].to_dict()
    fwd_col = option_df['expiration'].map(fwd_dict)
    filtered = option_df[option_df['strike'] < fwd_col]
    k_zeroes = filtered.groupby('expiration')['strike'].agg('max')
    forward_df = forward_df.set_index('expiration')
    forward_df['atmf'] = k_zeroes
    forward_df = forward_df.reset_index()

    return forward_df


def filter_itm(option_df, forward_df):
    """
    Filter out in-the-money-forward options (except for K0).

    :param option_df: pd.DataFrame of the option data
    :param forward_df: pd.DataFrame with the forward prices
    :return: pd.DataFrame
    """

    atmf_dict = forward_df.set_index('expiration')['atmf'].to_dict()
    k_zero_map = option_df['expiration'].map(atmf_dict)

    itm_call_filter = ((option_df['option_type'] == 'CALL')
                       & (option_df['strike'] >= k_zero_map))

    itm_put_filter = ((option_df['option_type'] == 'PUT')
                      & (option_df['strike'] <= k_zero_map))

    option_df = option_df[itm_call_filter | itm_put_filter]

    return option_df


def filter_tails(option_df):
    """
    Filter options past two consecutive zero bids.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    option_df = option_df.groupby('expiration').apply(group_tail_filter)
    option_df.reset_index(drop=True, inplace=True)

    return option_df


def group_tail_filter(group):
    """
    Custom group function for filtering tails of each expiry.

    :param group: pd.DataFrame group object
    :return: pd.DataFrame group object
    """

    group = group.sort_values('strike')

    call_filter = ((group['bid'] == 0)
                   & (group['bid'].shift(-1) == 0)
                   & (group['option_type'] == 'CALL'))

    put_filter = ((group['bid'] == 0)
                  & (group['bid'].shift() == 0)
                  & (group['option_type'] == 'PUT'))

    dub_zero = group[call_filter | put_filter]
    call_cut = dub_zero.loc[dub_zero['option_type'] == 'CALL', 'strike'].min()
    put_cut = dub_zero.loc[dub_zero['option_type'] == 'PUT', 'strike'].max()
    group = group[(group['strike'] < call_cut) & (group['strike'] > put_cut)]

    return group


def filter_zeroes(option_df):
    """
    Filter out options with zero bid.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    option_df = option_df[option_df['bid'] != 0]

    return option_df


def atmf_average(option_df):
    """
    Average the mid prices for puts and calls at the K0 strike.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """
    option_df = option_df.groupby('expiration').apply(group_average_atmf)
    option_df.reset_index(drop=True, inplace=True)

    return option_df


def group_average_atmf(group):
    """
    Custom group function for calculating the average mid prices

    :param group: pd.DataFrame group object
    :return: pd.DataFrame group object
    """

    aggs = {'strike': pd.Series.unique,
            'expiration': pd.Series.unique,
            'bid': 'mean',
            'ask': 'mean',
            'expiration_type': pd.Series.unique,
            'option_type': lambda x: ''.join(x.unique()),
            'mid': 'mean'}

    dub_group = group.groupby('strike').agg(aggs)

    return dub_group


def delta_k(option_df):
    """
    Calculate the delta K for strike contribution weightings.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    averages = option_df.groupby('expiration').apply(group_dk)
    option_df['dK'] = averages.reset_index(drop=True)

    return option_df


def group_dk(group):
    """
    Custom group function for calculating the delta-K for each expiry

    :param group: pd.DataFrame group object
    :return: pd.DataFrame group object
    """

    up = group['strike'].shift(-1)
    down = group['strike'].shift()
    average = (up - down) / 2
    average.iloc[0] = group['strike'].iloc[1] - group['strike'].iloc[0]
    average.iloc[-1] = group['strike'].iloc[-1] - group['strike'].iloc[-2]

    return average


def strike_contributions(option_df, forward_df):
    """
    Calculate the weighted average variance contributions for each strike.

    :param option_df: pd.DataFrame of the option data
    :param forward_df: pd.DataFrame with the forward prices
    :return: pd.DataFrame
    """

    rfr_map = forward_df.set_index('expiration')['rfr'].to_dict()
    option_df['rfr'] = option_df['expiration'].map(rfr_map)
    ttm_map = forward_df.set_index('expiration')['ttm'].to_dict()
    option_df['ttm'] = option_df['expiration'].map(ttm_map)

    option_df['k2'] = option_df['strike'] ** 2

    option_df['contribution'] = (option_df['dK'] / option_df['k2']
                                 * np.exp(option_df['rfr'] * option_df['ttm'])
                                 * option_df['mid'])

    return option_df


def total_variance(option_df):
    """
    Calculate the total variance per expiry.

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    variance = option_df.groupby('expiration').apply(group_total_variance)

    return variance


def group_total_variance(group):
    """
    Custom group function to calculate the total variance per expiration

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    group_variance = (2 / group['ttm']) * group['contribution']
    group_variance = group_variance.sum()

    return group_variance


def variance_adj(forward_df):
    """
    Calculate the forward variance adjustment

    :param option_df: pd.DataFrame of the option data
    :return: pd.DataFrame
    """

    fwd_ratio = (forward_df['fwd_price'] / forward_df['atmf']) - 1
    fwd_ratio = fwd_ratio ** 2
    fwd_adj = (1 / forward_df['ttm']) * fwd_ratio
    fwd_adj.index = forward_df['expiration']

    return fwd_adj


def interpolate_vix(sigma_squared, forward_df, target_days):
    """
    Interpolate the constant-maturity quotation point.

    :param sigma_squared: pd.DataFrame of the calculated variances
    :param forward_df: pd.DataFrame of the implied forwards
    :param target_days: int of the days to target for the constant maturity
    :return: float of the VIX value
    """

    target_mins = target_days * 60 * 24
    m1 = forward_df.loc[forward_df['mte'] < target_mins, 'mte'].idxmax()
    m1 = forward_df.loc[m1]

    m2 = forward_df.loc[forward_df['mte'] > target_mins, 'mte'].idxmin()
    m2 = forward_df.loc[m2]

    tw1 = (m2['mte'] - target_mins) / (m2['mte'] - m1['mte'])
    tw2 = (target_mins - m1['mte']) / (m2['mte'] - m1['mte'])

    m1_var = m1['ttm'] * sigma_squared[m1['expiration']] * tw1
    m2_var = m2['ttm'] * sigma_squared[m2['expiration']] * tw2

    eff_variance = m1_var + m2_var

    ann_variance = eff_variance * (365 * 24 * 60 / target_mins)

    vix = np.sqrt(ann_variance) * 100

    return vix


def calculate_vix(quote_date, yield_df, option_df, target_days=30):
    """
    Main entry point/wrapper function for the VIX calculation.

    NOTE: The data for yields and options should look similar to below.

    ttm,yield
    0.0821,0.0153
    0.1666,0.0155
    0.2500,0.0154
    0.5000,0.0156

    expiration,strike,option_type,bid,ask,expiration_type
    2-21-2020,3000,PUT,3.50,3.70,standard
    2-21-2020,3100,PUT,6.80,7.00,standard
    2-21-2020,3200,PUT,14.00,14.20,standard
    2-21-2020,3300,CALL,54.30,54.70,standard
    2-21-2020,3400,CALL,8.30,8.50,standard
    2-28-2020,3000,PUT,5.20,5.40,weekly
    2-28-2020,3200,PUT,18.50,18.80,weekly
    2-28-2020,3400,CALL,11.60,11.90,weekly
    2-28-2020,3500,CALL,1.05,1.20,weekly


    :param quote_date: datetime-like for the moment of quotation (e.g. now)
    :param yield_df: pd.DataFrame of risk-free spot (zero) interest rate curve
    :param option_df: pd.DataFrame of the option chain quotes
    :param target_days: int of the days to target for the constant maturity
    :return: float of the VIX value
    """

    options = map_exact_expiration(option_df)
    options = calculate_mid(options)
    forwards = rough_forwards(options)
    forwards = granular_forwards(forwards, yield_df, quote_date)
    forwards = find_k_zero(options, forwards)
    options = filter_itm(options, forwards)
    options = filter_tails(options)
    options = filter_zeroes(options)
    options = atmf_average(options)
    options = delta_k(options)
    options = strike_contributions(options, forwards)
    t_var = total_variance(options)
    fwd_adj = variance_adj(forwards)
    sigma_squared = t_var - fwd_adj
    vix = interpolate_vix(sigma_squared, forwards, target_days)

    return vix


if __name__ == '__main__':

    # Set the date for the VIX quote (a reasonable assumption would be 'now')
    quote_date = pd.Timestamp('1-27-2020 09:46:00')

    # Get interest rate data
    yields = pd.read_csv('wp_yields.csv')
    yields['maturity'] = pd.to_datetime(yields['maturity'])
    # Convert yield tenors to years for use with the interpolator
    yields['ttm'] = (yields['maturity'] - quote_date) / pd.Timedelta('1Y')

    # Get the option chain quotes
    options = pd.read_csv('wp_data.csv')
    options['expiration'] = pd.to_datetime(options['expiration'])

    # Run the calculation
    vix = calculate_vix(quote_date, yields, options)

    print(vix)
