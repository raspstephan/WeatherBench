"""
Functions for evaluating forecasts.
"""
import numpy as np
import xarray as xr
#import properscoring as ps
import xskillscore as xs

def load_test_data(path, var, years=slice('2017', '2018'), cmip=False):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    assert var in ['z', 't'], 'Test data only for Z500 and T850'
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    if cmip:
        ds['plev'] /= 100
        ds = ds.rename({'plev': 'level'})
    try:
        ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
    except ValueError:
        pass
    return ds.sel(time=years)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    if type(rmse) is xr.Dataset:
        rmse = rmse.rename({v: v + '_rmse' for v in rmse})
    else: # DataArray
        rmse.name = error.name + '_rmse' if not error.name is None else 'rmse'
    return rmse




def evaluate_iterative_forecast(da_fc, da_valid, func, mean_dims=xr.ALL_DIMS):
    rmses = []
    for f in da_fc.lead_time:
        fc = da_fc.sel(lead_time=f)
        fc['time'] = fc.time + np.timedelta64(int(f), 'h')
        rmses.append(func(fc, da_valid, mean_dims))
    return xr.concat(rmses, 'lead_time')
    # return xr.DataArray(rmses, dims=['lead_time'], coords={'lead_time': fc_iter.lead_time})


def compute_weighted_acc(da_fc, da_true):
    clim = da_true.mean('time')
    t = np.intersect1d(da_fc.time, da_true.time)
    fa = da_fc.sel(time=t) - clim
    a = da_true.sel(time=t) - clim
    
    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat
    
    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()
    
    acc = (
        np.sum(w * fa_prime * a_prime) /
        np.sqrt(
            np.sum(w * fa_prime**2) * np.sum(w * a_prime**2)
        )
    )
    return acc

def compute_weighted_meanspread(da_fc,mean_dims=xr.ALL_DIMS):
    """
    prediction: xarray. Coordinates: time, forecast_number, lat, lon. Variables: z500, t850
    time: Let there be I initial conditions
    forecast_number: For each initial condition, let there be N forecasts
    #mean variance
    #1. for each input i, for each gridpoint, find variance among all N forecasts for that single input i
    #2. for each input i, find latitude-weighted average of all the lat*lon points
    #3. find average of all I inputs. take square root
    """
    #ToDO: add assert condition to check for input size. Alternatively, if input does not have 'time' then add it as dimension
    var1=da_fc.var('member')
    weights_lat = np.cos(np.deg2rad(var1.lat))
    weights_lat /= weights_lat.mean()
    mean_spread= np.sqrt((var1*weights_lat).mean(mean_dims))
    
    if type(mean_spread) is xr.Dataset:
        mean_spread = mean_spread.rename({v: v + '_mean_spread' for v in mean_spread})
    else: # DataArray
        mean_spread.name = error.name + '_mean_spread' if not error.name is None else 'mean_spread'
    return mean_spread


def compute_weighted_crps(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    da_true=da_true.sel(time=da_fc.time)
    assert (da_true.time==da_fc.time).all #checking size.
    
    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    crps = xs.crps_ensemble(da_true, da_fc)
    crps = (crps * weights_lat).mean(mean_dims)
    return crps
# def crps_score(da_fc,da_true,member_axis,mean_dims=xr.ALL_DIMS): 
#     #check size
#     da_true=da_true.sel(time=da_fc.time)
#     assert (da_true.time==da_fc.time).all
    
#     #import properscoring as ps
#     obs = np.asarray(da_true.to_array(), dtype=np.float32).squeeze();
#     #shape: (variable,time, lat, lon)
#     pred=np.asarray(da_fc.to_array(), dtype=np.float32).squeeze();
#     #shape: (variable, member, time, lat, lon)
#     member_axis=member_axis+1 #Weird but have to do since the above line changes position of member_axis
#     if pred.ndim==4: #for single ensemble member. #ToDo: make it general
#         pred=np.expand_dims(pred,axis=member_axis)
    
#     crps=ps.crps_ensemble(obs,pred, weights=None, issorted=False,axis=member_axis) 
#     #crps.shape  #(variable, time, lat, lon)
# #     if crps.ndim==3: #for single input.#ToDo: make it general
# #         crps=np.expand_dims(crps,axis=member_axis)
#    #Converting back to xarray
#     crps_score = xr.Dataset({
#         'z500': xr.DataArray(
#         crps[0,...],
#         dims=['time', 'lat', 'lon'],
#         coords={'time': da_true.time, 'lat': da_true.lat, 'lon': da_true.lon,
#                 },
#         ),
#         't850': xr.DataArray(
#         crps[1,...],
#         dims=['time', 'lat', 'lon'],
#         coords={'time': da_true.time, 'lat': da_true.lat, 'lon': da_true.lon,
#                 },
#         )
#     })
    
#     #averaging to get single valye
#     weights_lat = np.cos(np.deg2rad(crps_score.lat))
#     weights_lat /= weights_lat.mean()
#     crps_score = (crps_score* weights_lat).mean(mean_dims)
    
#     return crps_score

def compute_weighted_mae(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the MAE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
    Returns:
        mae: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    mae = (np.abs(error) * weights_lat).mean(mean_dims)
    if type(mae) is xr.Dataset:
        mae = mae.rename({v: v + '_mae' for v in mae})
    else: # DataArray
        mae.name = error.name + '_mae' if not error.name is None else 'mae'
    return mae