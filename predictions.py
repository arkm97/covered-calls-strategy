import numpy as np
import pandas as pd
import gloess
import warnings

def custom_fourier(p, df):
    """
    df: [('index'), 'x', 'f(x)']
    """
    out = 0
    for idx, row in df.iterrows():
        out += row['f(x)'] * np.exp(-1j * p * row['x'])

    return out

# integrated vertex functions
def integrated_vertex_fn(m, p, p0):

    b = m ** 2

    return -((2 * np.sqrt(-4 * b - p + 0j) * np.arctan(np.sqrt(p + 0j) / np.sqrt(-4 * b - p + 0j))) / np.sqrt(p + 0j)) + (2 * np.sqrt(-4 * b - p0 + 0j) * np.arctan(np.sqrt(p0 + 0j) / np.sqrt(-4 * b - p0 + 0j))) / np.sqrt(p0 + 0j)

def integrated_vertex_fn_mixed(m_phi, m_psi, p, p0):

    a = m_phi ** 2 - m_psi ** 2
    b = m_psi ** 2

    numerator_1 = -1 * (-2 * np.sqrt(-a ** 2 - 2 * a * p - p * (4 * b + p) + 0j) * p0 * np.arctan((-a - p) / np.sqrt(-a ** 2 - 2 * a * p - p * (4 * b + p) + 0j)) + 2 * p * np.sqrt(-a ** 2 - 2 * a * p0 - p0 * (4 * b + p0) + 0j) * np.arctan((-a - p0) / np.sqrt(-a ** 2 - 2 * a * p0 - p0 * (4 * b + p0) + 0j)) + (a + p) * p0 * np.log(b) - p * (a + p0) * np.log(b))

    numerator_2 =  1 * (-2 * np.sqrt(-a ** 2 - 2 * a * p - p * (4 * b + p) + 0j) * p0 * np.arctan((-a + p) / np.sqrt(-a ** 2 - 2 * a * p - p * (4 * b + p) + 0j)) + 2 * p * np.sqrt(-a ** 2 - 2 * a * p0 - p0 * (4 * b + p0) + 0j) * np.arctan((-a + p0) / np.sqrt(-a ** 2 - 2 * a * p0 - p0 * (4 * b + p0) + 0j)) + (a + p) * p0 * np.log(a + b) - p * (a + p0) * np.log(a + b))

    denominator = 2 * p * p0

    return (numerator_1 + numerator_2) / denominator

# renormalized amplitudes
def A_visible_visible(s_t_u, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi):

    s, t, u = s_t_u

    A = -1 * lambda_phi + 1/2 * 1/(32 * np.pi ** 2) * lambda_phi ** 2 * (integrated_vertex_fn(m_phi, s, s0) + integrated_vertex_fn(m_phi, t, t0) + integrated_vertex_fn(m_phi, u, u0)) + 1/36 * 1/(32 * np.pi ** 2) * lambda_phi_psi ** 2 * (integrated_vertex_fn(m_psi, s, s0) + integrated_vertex_fn(m_psi, t, t0) + integrated_vertex_fn(m_psi, u, u0))
    return A

def A_visible_hidden(s_t_u, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi):

    s, t, u = s_t_u

    A = -1 * lambda_phi_psi + 1/36 * 1/(32 * np.pi ** 2) * lambda_phi_psi ** 2 * (integrated_vertex_fn_mixed(m_phi, m_psi, t, t0) + integrated_vertex_fn_mixed(m_phi, m_psi, u, u0)) + 1/12 * 1/(32 * np.pi ** 2) * lambda_phi * lambda_psi * (integrated_vertex_fn(m_phi, s, s0) + integrated_vertex_fn(m_psi, s, s0))
    return A

def A_mixed_mixed(s_t_u, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi):

    s, t, u = s_t_u

    A = -1 * lambda_phi_psi + 1/36 * 1/(32 * np.pi ** 2) * lambda_phi_psi ** 2 * (integrated_vertex_fn_mixed(m_phi, m_psi, s, s0) + integrated_vertex_fn_mixed(m_phi, m_psi, t, t0) + integrated_vertex_fn_mixed(m_phi, m_psi, u, u0)) + 1/12 * 1/(32 * np.pi ** 2) * lambda_phi * lambda_psi * (integrated_vertex_fn(m_phi, t, t0) + integrated_vertex_fn(m_psi, t, t0) + integrated_vertex_fn(m_phi, u, u0) + integrated_vertex_fn(m_psi, u, u0))
    return A

def A_combined(s_t_u, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi, normalize=1):

    s_t_u_vv, s_t_u_vh, s_t_u_m = s_t_u

    A_vv = A_visible_visible(s_t_u_vv, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi)

    A_vh = A_visible_hidden(s_t_u_vh, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi)

    A_m = A_mixed_mixed(s_t_u_m, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi)

    return (pd.Series(A_vv).fillna(0) + pd.Series(A_vh).fillna(0) + pd.Series(A_m).fillna(0)) / normalize


def data_window(data, row, weeks):
    if weeks < 0:
        df = data.loc[(data.index >= (row.index + pd.Timedelta(weeks, 'W'))[0]) & (data.index <= (row.index[0]))]
    else:
        df = data.loc[(data.index <= (row.index + pd.Timedelta(weeks, 'W'))[0]) & (data.index >= (row.index[0]))]
    return df

def p_distributions(df, weeks=4):
    if weeks < 0:
        _price = np.log(df.close / df.close.tail(1)[0]) / 4.28 # 4.28 normalization based on max possible log return
        _vol = _price.rolling(3).std() / 1.00051 # do this instead of volatility, the exact definition for spread is not important
        _time = -(_price.tail(1).index[0] - _price.index).days.values

    else:
        _price = np.log(df.close / df.close.head(1)[0]) / 4.28 # 4.28 normalization based on max possible log return
        _vol = _price.rolling(3).std() / 1.00051 # do this instead of volatility, the exact definition for spread is not important
        _time = -(_price.head(1).index[0] - _price.index).days.values

    # hist
    _price_bins, _price_bin_centers, _price_values = gloess.custom_hist(_price, _price.min() * 1.25, _price.max() * 1.25, .002, show_plots=False)
    _vol_bins, _vol_bin_centers, _vol_values = gloess.custom_hist(_vol, _price.min() * 1.25, _price.max() * 1.25, .002, show_plots=False)
    if weeks < 0:
        t_min = weeks * 7 * 1.5 * .9 # (extra 50% around week window, 90% before, 10% after)
        t_max = -1 * weeks * 7 * 1.5 * .1 # (extra 50% around week window, 90% before, 10% after)
        _time_bins, _time_bin_centers, _time_values = gloess.custom_hist(_time, t_min, t_max, 1, show_plots=False)
    else:
        t_min = -1 * weeks * 7 * 1.5 * .1 # (extra 50% around week window, 10% before, 90% after)
        t_max = weeks * 7 * 1.5 * .9 # (extra 50% around week window, 10% before, 90% after)
        _time_bins, _time_bin_centers, _time_values = gloess.custom_hist(_time, t_min, t_max, 1, show_plots=False)

    # smoothing
    _price_df = pd.concat([pd.Series(_price_bin_centers), pd.Series(_price_values)], axis=1)
    _price_df.columns = ['x', 'y']
    _price_df_smoothed = np.clip(gloess.gaussian_lowess(_price_df, s=.5, order=2, spread=100), a_max=None, a_min=0)
    _price_df_smoothed = pd.concat([_price_df['x'], pd.Series(_price_df_smoothed)], axis=1)
    _price_df_smoothed.columns = ['x', 'f(x)']

    _vol_df = pd.concat([pd.Series(_vol_bin_centers), pd.Series(_vol_values)], axis=1)
    _vol_df.columns = ['x', 'y']
    _vol_df_smoothed = np.clip(gloess.gaussian_lowess(_vol_df, s=.5, order=2, spread=100), a_max=None, a_min=0)
    _vol_df_smoothed = pd.concat([_vol_df['x'], pd.Series(_vol_df_smoothed)], axis=1)
    _vol_df_smoothed.columns = ['x', 'f(x)']

    _time_df = pd.concat([pd.Series(_time_bin_centers), pd.Series(_time_values)], axis=1)
    _time_df.columns = ['x', 'y']
    _time_df_smoothed = np.clip(gloess.gaussian_lowess(_time_df, s=.4, order=2, spread=100), a_max=None, a_min=0)
    _time_df_smoothed = pd.concat([_time_df['x'], pd.Series(_time_df_smoothed)], axis=1)
    _time_df_smoothed.columns = ['x', 'f(x)']

    ## fourier transforms
    # _p = np.linspace(-2000, 2000, 100)
    _p = np.logspace(np.log10(10), np.log10(3000), 50)#
    _p = list(-1 * _p[::-1]) + list(_p)

    _price_p_distribution = [custom_fourier(i, _price_df_smoothed) for i in _p]
    _price_p_distribution = pd.concat([pd.Series(_p), pd.Series(_price_p_distribution)], axis=1)
    _price_p_distribution.columns = ['p', 'f(p)']
    _price_p_distribution['RE[f(p)]'] = np.real(_price_p_distribution['f(p)'])
    _price_p_distribution['IM[f(p)]'] = np.imag(_price_p_distribution['f(p)'])

    _vol_p_distribution = [custom_fourier(i, _vol_df_smoothed) for i in _p]
    _vol_p_distribution = pd.concat([pd.Series(_p), pd.Series(_vol_p_distribution)], axis=1)
    _vol_p_distribution.columns = ['p', 'f(p)']
    _vol_p_distribution['RE[f(p)]'] = np.real(_vol_p_distribution['f(p)'])
    _vol_p_distribution['IM[f(p)]'] = np.imag(_vol_p_distribution['f(p)'])

    _time_p_distribution = [custom_fourier(i, _time_df_smoothed) for i in _p]
    _time_p_distribution = pd.concat([pd.Series(_p), pd.Series(_time_p_distribution)], axis=1)
    _time_p_distribution.columns = ['p', 'f(p)']
    _time_p_distribution['RE[f(p)]'] = np.real(_time_p_distribution['f(p)'])
    _time_p_distribution['IM[f(p)]'] = np.imag(_time_p_distribution['f(p)'])

    return {'_price_p_distribution': _price_p_distribution,
            '_vol_p_distribution': _vol_p_distribution,
            '_time_p_distribution': _time_p_distribution}

def Mandelstam_df(p1, p2, p3, p4, suffix=None):
    """
    p1 = ['p1_2_time', 'p1_3_time', 'p1_2_price', 'p1_3_price']
    .
    .
    .
    p4
    """
    # sum p1, p2 component-wise
    p1_p2_0 = p1.iloc[:,0].values + p2.iloc[:,0].values
    p1_p2_1 = p1.iloc[:,1].values + p2.iloc[:,1].values
    p1_p2_2 = p1.iloc[:,2].values + p2.iloc[:,2].values
    p1_p2_3 = p1.iloc[:,3].values + p2.iloc[:,3].values

    # Minkowski inner product to form (p1 + p2) ** 2
    s = p1_p2_2 ** 2 + p1_p2_3 ** 2 - p1_p2_0 ** 2 - p1_p2_1 ** 2

    # p1 - p3 component-wise
    p1_p3_0 = p1.iloc[:,0].values - p3.iloc[:,0].values
    p1_p3_1 = p1.iloc[:,1].values - p3.iloc[:,1].values
    p1_p3_2 = p1.iloc[:,2].values - p3.iloc[:,2].values
    p1_p3_3 = p1.iloc[:,3].values - p3.iloc[:,3].values

    # Minkowski inner product to form (p1 - p3) ** 2
    t = p1_p3_2 ** 2 + p1_p3_3 ** 2 - p1_p3_0 ** 2 - p1_p3_1 ** 2

    # p1 - p3 component-wise
    p1_p4_0 = p1.iloc[:,0].values - p4.iloc[:,0].values
    p1_p4_1 = p1.iloc[:,1].values - p4.iloc[:,1].values
    p1_p4_2 = p1.iloc[:,2].values - p4.iloc[:,2].values
    p1_p4_3 = p1.iloc[:,3].values - p4.iloc[:,3].values

    # Minkowski inner product to form (p1 - p4) ** 2
    u = p1_p4_2 ** 2 + p1_p4_3 ** 2 - p1_p4_0 ** 2 - p1_p4_1 ** 2

    _out = pd.DataFrame.from_dict({'s' + suffix: s, 't' + suffix: t, 'u' + suffix: u})

    return _out

def predict_density_price(data, day, weeks_before=4, weeks_after=4):
    '''Predict relative probability density of log return
        during given window following a given observation
        day.

        returns a DataFrame with
        x = log_10(price(t) / price(t_0)) / 4.28
        y = relative probability density

    params:

    df data — raw data with close prices indexed by day
    str day — date at which to make prediction, e.g. 2018-02-03
    int weeks_before — size of window before observation day,
                        used to make prediction; default 4
                        (4 was used for training)
    int weeks_after — size of window after observation day;
                        default 4 (4 was used for training)

    '''

    ####################
    # Data preparation #
    ####################

    row = data.loc[day]

    # get observation windows around each day
    df_initial = data_window(data, row, -weeks_before)

    # get p distributions for each initial/final window
    p_initial = p_distributions(df_initial, -weeks_before)

    # get observation windows around each day
    df_final = data_window(data, row, weeks_before)

    # get p distributions for each initial/final window
    p_final = p_distributions(df_final, weeks_before)

    # get sample of initial price distribution
    distribution = '_price_p_distribution'
    n_samples = 1000
    _p_initial_real_sample = p_initial[distribution]['p'].sample(n_samples, replace=True, weights=np.abs(p_initial[distribution]['RE[f(p)]'] / np.abs(p_initial[distribution]['RE[f(p)]']).sum()))
    _p_initial_imag_sample = p_initial[distribution]['p'].sample(n_samples, replace=True, weights=np.abs(p_initial[distribution]['IM[f(p)]'] / np.abs(p_initial[distribution]['IM[f(p)]']).sum()))
    _sample = pd.DataFrame(np.vstack([np.array(_p_initial_real_sample).reshape(2, int(n_samples / 2)), np.array(_p_initial_imag_sample).reshape(2, int(n_samples / 2))])).T
    _sample.columns = ['p1_1', 'p2_1', 'p1_2', 'p2_2'] # p1 is "incoming particle 1", p2 is "incoming 2"; p1 and p2 are sampled from the same distributions
    _sample['total_p_1'] = _sample['p1_1'] + _sample['p2_1'] # _2 is 3rd component of p (real price fourier components)
    _sample['total_p_2'] = _sample['p1_2'] + _sample['p2_2'] # _3 is the 4th component of p (imag price fourier components)

    # get sample of initial time distribution
    distribution = '_time_p_distribution'
    n_samples = 1000
    _p_initial_real_sample = p_initial[distribution]['p'].sample(n_samples, replace=True, weights=np.abs(p_initial[distribution]['RE[f(p)]'] / np.abs(p_initial[distribution]['RE[f(p)]']).sum()))
    _p_initial_imag_sample = p_initial[distribution]['p'].sample(n_samples, replace=True, weights=np.abs(p_initial[distribution]['IM[f(p)]'] / np.abs(p_initial[distribution]['IM[f(p)]']).sum()))
    _sample_time = pd.DataFrame(np.vstack([np.array(_p_initial_real_sample).reshape(2, int(n_samples / 2)), np.array(_p_initial_imag_sample).reshape(2, int(n_samples / 2))])).T
    _sample_time.columns = ['p1_1', 'p2_1', 'p1_2', 'p2_2'] # p1 is "incoming particle 1", p2 is "incoming 2"; p1 and p2 are sampled from the same distributions
    _sample_time['total_p_1'] = _sample_time['p1_1'] + _sample_time['p2_1'] # _2 is 3rd component of p (real price fourier components)
    _sample_time['total_p_2'] = _sample_time['p1_2'] + _sample_time['p2_2'] # _3 is the 4th component of p (imag price fourier components)

    # merge price/time
    _sample_price_time = _sample.merge(_sample_time, how='cross', suffixes=['_price', '_time'])

    # sample from uniform candidate momentum distribution
    # _p = np.logspace(np.log10(10), np.log10(2000), 50)#
    # _p = list(-1 * _p[::-1]) + list(_p)
    _p = np.linspace(-2000, 2000, 100)

    # sample from final time distribution
    _time_bins, _time_bin_centers, _time_values = gloess.custom_hist(np.random.normal(loc=7 * weeks_after, scale=1, size=7 * weeks_after), -3, 7 * (weeks_after + 1), 1, show_plots=False)
    _time_df = pd.concat([pd.Series(_time_bin_centers), pd.Series(_time_values)], axis=1)
    _time_df.columns = ['x', 'y']
    _time_df_smoothed = np.clip(gloess.gaussian_lowess(_time_df, s=.4, order=2, spread=100), a_max=None, a_min=0)
    _time_df_smoothed = pd.concat([_time_df['x'], pd.Series(_time_df_smoothed)], axis=1)
    _time_df_smoothed.columns = ['x', 'f(x)']
    _time_p_distribution = [custom_fourier(i, _time_df_smoothed) for i in  np.linspace(-2000, 2000, 100)]
    _time_p_distribution = pd.concat([pd.Series( np.linspace(-2000, 2000, 100)), pd.Series(_time_p_distribution)], axis=1)
    _time_p_distribution.columns = ['p', 'f(p)']
    _time_p_distribution['RE[f(p)]'] = np.real(_time_p_distribution['f(p)'])
    _time_p_distribution['IM[f(p)]'] = np.imag(_time_p_distribution['f(p)'])
    _p_final_real_sample = _time_p_distribution['p'].sample(n_samples, replace=True, weights=np.abs(_time_p_distribution['RE[f(p)]'] / np.abs(_time_p_distribution['RE[f(p)]']).sum()))
    _p_final_imag_sample = _time_p_distribution['p'].sample(n_samples, replace=True, weights=np.abs(_time_p_distribution['IM[f(p)]'] / np.abs(_time_p_distribution['IM[f(p)]']).sum()))
    _sample_time = pd.DataFrame(np.vstack([np.array(_p_final_real_sample).reshape(2, int(n_samples / 2)), np.array(_p_final_imag_sample).reshape(2, int(n_samples / 2))])).T

    _p_final_candidates = pd.concat([pd.DataFrame(np.random.choice(_p, size=(4000, 4))), _sample_time.rename(columns={0: 'p3_1', 1: 'p4_1', 2: 'p3_2', 3: 'p4_2'})], ignore_index=False, axis=1).rename(columns={0: 'p3_3', 1: 'p3_4', 2: 'p4_3', 3: 'p4_4'})

    # merge initial with candidates (need initials for scoring each candidate)
    p_initial_final_merged = _sample_price_time.sample(4000).merge(_p_final_candidates.rename(columns={0: 'p3_1', 1: 'p3_2', 2: 'p3_3', 3: 'p3_4', 4: 'p4_1', 5: 'p4_2', 6: 'p4_3', 7: 'p4_4'}), how='cross')

    # convert to form LSZ Model can score
    p_initial_final_mandelstam = pd.concat([Mandelstam_df(p_initial_final_merged[['p1_1_time', 'p1_2_time', 'p1_1_price', 'p1_2_price']],
                                                          p_initial_final_merged[['p2_1_time', 'p2_2_time', 'p2_1_price', 'p2_2_price']],
                                                          p_initial_final_merged[['p3_1', 'p3_2', 'p3_3', 'p3_4']],
                                                          p_initial_final_merged[['p4_1', 'p4_2', 'p4_3', 'p4_4']], suffix='_vv'),
                                             Mandelstam_df(p_initial_final_merged[['p3_1', 'p3_2', 'p3_3', 'p3_4']].sample(frac=1),
                                                          p_initial_final_merged[['p4_1', 'p4_2', 'p4_3', 'p4_4']].sample(frac=1),
                                                          p_initial_final_merged[['p3_1', 'p3_2', 'p3_3', 'p3_4']],
                                                          p_initial_final_merged[['p4_1', 'p4_2', 'p4_3', 'p4_4']], suffix='_vh'),
                                             Mandelstam_df(p_initial_final_merged[['p1_1_time', 'p1_2_time', 'p1_1_price', 'p1_2_price']],
                                                          p_initial_final_merged[['p2_1_time', 'p2_2_time', 'p2_1_price', 'p2_2_price']],
                                                          p_initial_final_merged[['p3_1', 'p3_2', 'p3_3', 'p3_4']],
                                                          p_initial_final_merged[['p4_1', 'p4_2', 'p4_3', 'p4_4']], suffix='_m')], axis=1)

    ###########
    # Scoring #
    ###########

    s0 = 1
    t0 = 1
    u0 = 1
    lambda_phi = -129.1
    lambda_psi = 729.7
    lambda_phi_psi = 564.0
    m_phi = -9.63511273e-01
    m_psi = 3.64887268e-02

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        p_initial_final_mandelstam['score_vv'] = A_visible_visible(p_initial_final_mandelstam[['s_vv', 't_vv', 'u_vv']].values.T, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi)
        p_initial_final_mandelstam['score_vh'] = A_visible_hidden(p_initial_final_mandelstam[['s_vh', 't_vh', 'u_vh']].values.T, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi)
        p_initial_final_mandelstam['score_m'] = A_mixed_mixed(p_initial_final_mandelstam[['s_m', 't_m', 'u_m']].values.T, s0, t0, u0, lambda_phi, lambda_psi, lambda_phi_psi, m_phi, m_psi)

    p_initial_final_merged[['score_vv', 'score_vh', 'score_m']] = np.abs(p_initial_final_mandelstam[['score_vv', 'score_vh', 'score_m']])

    p_final_prediction = pd.DataFrame(p_initial_final_merged.groupby('p3_3').score_vv.sum('sum') + 1j * p_initial_final_merged.groupby('p3_4').score_vv.sum('sum') + p_initial_final_merged.groupby('p4_3').score_vv.sum('sum') + 1j * p_initial_final_merged.groupby('p4_4').score_vv.sum('sum'))
    p_final_prediction['vh'] = pd.DataFrame(p_initial_final_merged.groupby('p3_3').score_vh.sum('sum') + 1j * p_initial_final_merged.groupby('p3_4').score_vh.sum('sum') + p_initial_final_merged.groupby('p4_3').score_vh.sum('sum') + 1j * p_initial_final_merged.groupby('p4_4').score_vh.sum('sum'))
    p_final_prediction['m'] = pd.DataFrame(p_initial_final_merged.groupby('p3_3').score_m.sum('sum') + 1j * p_initial_final_merged.groupby('p3_4').score_m.sum('sum'))
    p_final_prediction['score'] = p_final_prediction.sum(axis=1)


    ################################
    # Reconstruct Predicted Return #
    ################################

    _x = np.linspace(-0.1, 0.1, 1000)
    _reconstructed_return = pd.DataFrame([_x, np.abs([custom_fourier(-i, p_final_prediction[['score']].reset_index().rename(columns={'p3_3': 'x', 'score': 'f(x)'})) for i in _x])]).T
    _reconstructed_return.columns = ['x', 'y']

    # smoothing via gaussian-weighted local regression
    y = gloess.gaussian_lowess(_reconstructed_return, s=0.1, order=2)
    # smoothing round 2
    predicted_return = gloess.gaussian_lowess(pd.DataFrame(np.array([_x, y]).T, columns=['x', 'y']), s=.5, order=2)

    return pd.DataFrame.from_dict({'x': _x, 'p(x)': predicted_return, 'raw': _reconstructed_return['y'], 'smoothed_1': y})
