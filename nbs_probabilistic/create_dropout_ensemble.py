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
#make it work for all networks. #(Differences: custom_objects, loss function,  
#load full data instead of batches. output for full size of X. sometimes stop due to RAM limits.
#should we pass optional arguments. like is_normalized, start_date, end_date
#what to do if output vars are different?? numpy-->xarray wont work

def get_input(args):
    #Get Arguments
    #essential
    exp_id=args['exp_id']
    var_dict=args['var_dict']
    batch_size=args['batch_size']
    output_vars=args['output_vars'] #ToDo: need to change numpy-->xarray function if this changes.
    test_years=args['test_years']
    #optional. 
    #Question: how to optionally input data_subsample, norm_subsample, nt_in, dt_in,test_years?
    data_subsample=args['data_subsample']
    norm_subsample=args['norm_subsample']
    nt_in=args['nt_in']
    dt_in=args['dt_in']
    #changing paths
    #model_save_dir='/../../data/WeatherBench/predictions/saved_models'
    #datadir='/../../data/WeatherBench/5.625deg'
      
    #For Model. Ques: Needed?? Since we are not building the model again, but just simply switching on dropout, i dont think we need these.
    loss=args['loss'] #How to incorporate this in model?
    filters=args['filters']
    kernels=args['kernels']
    lead_time=args['lead_time']
    lr=args['lr']
    early_stopping_patience=args['early_stopping_patience']
    reduce_lr_patience=args['reduce_lr_patience']
    data_subsample=args['data_subsample']
    norm_subsample=args['norm_subsample']
    bn_position=args['bn_position']
    dropout=args['dropout']
    l2=args['l2']
    nt_in=args['nt_in'] #Ques: Is nt and nt_in the same thing? important since it casuses problem in input size for model layer 1 sometimes.
    #nt_in=args['nt']
    activation=args['activation']
    min_es_delta=args['min_es_delta'] #not for all configs.
    
    
    #run data generator
    ds = xr.merge([xr.open_mfdataset(f'../../data/WeatherBench/5.625deg/{var}/*.nc',
                                     combine='by_coords') for var in var_dict.keys()])
    mean = xr.open_dataarray(f'../../data/WeatherBench/predictions/saved_models/{exp_id}_mean.nc') 
    std = xr.open_dataarray(f'../../data/WeatherBench/predictions/saved_models/{exp_id}_std.nc')
    
    ds_test= ds.sel(time=slice(test_years[0],test_years[-1]))

    #Question: Should we input data_subsample, norm_subsample, nt_in (or nt?), dt_in?
    #Should shuffle be False? since its testing.
    dg_test = DataGenerator(ds_test, var_dict, lead_time, batch_size=batch_size, shuffle=False,
                            load=True,mean=mean, std=std, output_vars=output_vars,
                            data_subsample=data_subsample, norm_subsample=norm_subsample,
                            nt_in=nt_in,dt_in=dt_in) 
    return dg_test

def get_model(exp_id):
    tf.compat.v1.disable_eager_execution() #needed?
    saved_model_path=f'../../data/WeatherBench/predictions/saved_models/{exp_id}.h5'
    
    #ToDo: make loss function general.
    #Break the code with error msg if not resnet or unet_google
    #Important Ques: Since we dont build again, are we passing kernel, filters, activation, dropout and other details to the network?
    if 'resnet' in exp_id:
        model=tf.keras.models.load_model(saved_model_path
                ,custom_objects={'PeriodicConv2D':PeriodicConv2D,'lat_mse': tf.keras.losses.mse})
    elif 'unet_google' in exp_id:
        model=tf.keras.models.load_model(saved_model_path, 
                                         custom_objects={'PeriodicConv2D':PeriodicConv2D})#loss fn.?
    else:
        print('Only resnet or unet_google allowed')
        #ToDO: Raise Exception
        #Ques: Does Unet (not unet_google) work?
    #model.summary()
    return model

def predict(dg_test,model,number_of_forecasts):
    X,y=dg_test[0] #currently limiting output due to RAM issues.
#     for i in range(len(dg_test)-1):
#     X_batch,y_batch=dg_test[i+1]
#     X=np.append(X,X_batch,axis=0)
#     y=np.append(y,y_batch,axis=0)
    
    #test-time dropout
    func = K.function(model.inputs + [K.learning_phase()], model.outputs)
    pred_ensemble = np.array([np.asarray(func([X] + [1.]), dtype=np.float32).squeeze() for _ in
                              range(number_of_forecasts)])
    
    #unnormalize
    observation=y
    pred_ensemble=(pred_ensemble * dg_test.std.isel(level=dg_test.output_idxs).values +
         dg_test.mean.isel(level=dg_test.output_idxs).values)
    observation=(observation * dg_test.std.isel(level=dg_test.output_idxs).values +
         dg_test.mean.isel(level=dg_test.output_idxs).values)
    
    #numpy --> xarray. ToDo: change if output_vars is changed.
    #convert from numpy to xarray
    preds = xr.Dataset({
    'z500': xr.DataArray(pred_ensemble[...,0],
        dims=['forecast_number', 'time','lat', 'lon'],
        coords={'forecast_number': np.arange(number_of_forecasts),'time': dg_test.data.time.isel(time=slice(None,X.shape[0])), 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,},)
    ,
    't850': xr.DataArray(pred_ensemble[...,1],
        dims=['forecast_number', 'time','lat', 'lon'],
        coords={'forecast_number': np.arange(number_of_forecasts),'time': dg_test.data.time.isel(time=slice(None,X.shape[0])), 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,},)
})

    observation= xr.Dataset({
    'z500': xr.DataArray(observation[...,0],
                         dims=['time','lat','lon'],
                         coords={'time':dg_test.data.time.isel(time=slice(None,X.shape[0])),'lat':dg_test.data.lat,'lon':dg_test.data.lon},)
    ,
    't850': xr.DataArray(observation[...,1],dims=['time','lat','lon'],coords={'time':dg_test.data.time.isel(time=slice(None,X.shape[0])),'lat':dg_test.data.lat,'lon':dg_test.data.lon},)          
})

    return preds

def main(exp_id_path,number_of_forecasts):
    args=load_args(exp_id_path)
    exp_id=args['exp_id']#CouldDO: assert things like 'exp_id' in args #too many asserts needed.
    
    dg_test=get_input(args)
    mymodel=get_model(exp_id)
    preds=predict(dg_test,mymodel,number_of_forecasts)
    
    preds.to_netcdf(f'../../data/WeatherBench/predictions/{exp_id}.nc')
    print(f'saved on disk in data/WeatherBench/predictions/{exp_id}.nc')
    return

    

if __name__ == '__main__':
      fire.Fire(main)