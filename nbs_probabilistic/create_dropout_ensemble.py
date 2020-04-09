import fire
from fire import Fire
import xarray as xr
import numpy as np
from src.data_generator import *
from src.train import *
from src.networks import *
from src.utils import *
from tensorflow.keras import backend as K
#ToDO: 
#make it work for all networks. #(Differences: custom_objects, -can be done with an if conditon on load_model(), #output_vars, test_years, lead_time?, anything else?

#load full data instead of batches. output for full size of X.
#know how to pass optional arguments. like is_normalized, start_date, end_date
#what to do if output vars are different?? numpy-->xarray wont work
#what to do about custom_objects in get_model() function?
#save finally as .nc files -done.
def get_input(exp_id_path):
    #Getting Dictionary
    args = load_args(exp_id_path)
    var_dict=args['var_dict']
    lead_time=args['lead_time']
    batch_size=args['batch_size']
    output_vars=args['output_vars'] #ToDo: need to change numpy-->xarray function if this changes.
    # #Question: Should we input data_subsample, norm_subsample, nt_in, dt_in, test_year?
    # # data_subsample=args['data_subsample']
    # # norm_subsample=args['norm_subsample']
    # # nt_in=args['nt_in']
    # # dt_in=args['dt_in']
    # # test_years=args['test_years'] #ToDo
    
    ds = xr.merge([xr.open_mfdataset(f'../../data/WeatherBench/5.625deg/{var}/*.nc', combine='by_coords') for var in var_dict.keys()])
    
    #Question: How will this be available?
    mean = xr.open_dataarray('../../data/WeatherBench/5.625deg/13-mean.nc') #for year 2018??
    std = xr.open_dataarray('../../data/WeatherBench/5.625deg/13-std.nc')
    
    start_time='2017-01-01';end_time='2018-12-31' #Todo: want to use as optional arguments (i.e. default or if test_years specified, then that
    ds_test= ds.sel(time=slice(start_time, end_time))

    #Question: Should we input data_subsample, norm_subsample, nt_in, dt_in?
    # dg_test = DataGenerator(ds_test, var_dict, lead_time, batch_size=32, shuffle=True, load=True,
    #                  mean=None, std=None, output_vars=None, data_subsample=1, norm_subsample=1,
    #                  nt_in=1, dt_in=1 )
    dg_test = DataGenerator(
        ds_test, var_dict, lead_time, batch_size=batch_size, mean=mean, std=std,
        shuffle=False, output_vars=output_vars
    )
    return dg_test

def get_model(exp_id):
    tf.compat.v1.disable_eager_execution() #needed?
    
    #args = load_args(exp_id_path)
    #exp_id=args['exp_id']
    saved_model_path=f'../../data/WeatherBench/predictions/saved_models/{exp_id}.h5'
    
    mymodel=tf.keras.models.load_model(saved_model_path
            ,custom_objects={'PeriodicConv2D':PeriodicConv2D,'lat_mse': tf.keras.losses.mse})
    #ToDO: assert condition for string. error: file doesnt exist
    #mymodel.summary()
    return mymodel

def predict(dg_test,mymodel,number_of_forecasts,number_of_inputs):
    X,y=dg_test[0] #change to get whole data for all test_years. Till then use number_of_inputs
    time=number_of_inputs ## keep it lower for testing code. takes time.
    #number of inputs. different input times each for which an ensemble of predictions is made. 
    
    #test-time dropout
    func = K.function(mymodel.inputs + [K.learning_phase()], mymodel.outputs)
    pred_ensemble = np.array([np.asarray(func([X[:time]] + [1.]), dtype=np.float32).squeeze() for _ in range(number_of_forecasts)])
    
    #unnormalize
    pred_ensemble=(pred_ensemble * dg_test.std.isel(level=dg_test.output_idxs).values +
         dg_test.mean.isel(level=dg_test.output_idxs).values)

    #numpy --> xarray. ToDo: change if output_vars is changed.
    preds = xr.Dataset({
    'z500': xr.DataArray(pred_ensemble[...,0],
        dims=['forecast_number', 'time','lat', 'lon'],
        coords={'forecast_number': np.arange(number_of_forecasts),'time': np.arange(time), 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,},)
    ,
    't850': xr.DataArray(pred_ensemble[...,1],
        dims=['forecast_number', 'time','lat', 'lon'],
        coords={'forecast_number': np.arange(number_of_forecasts),'time': np.arange(time), 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,},)
})

    return preds

def main(exp_id_path,number_of_forecasts,number_of_inputs):
    dg_test=get_input(exp_id_path)
    
    args = load_args(exp_id_path)
    exp_id=args['exp_id']
    mymodel=get_model(exp_id)
    
    
    preds=predict(dg_test,mymodel,number_of_forecasts,number_of_inputs)
    
    preds.to_netcdf(f'../../data/WeatherBench/predictions/{exp_id}.nc')
    print('saved on disk')
    return

    

if __name__ == '__main__':
      fire.Fire(main)