from src.score import *
from src.data_generator import *
from src.networks import *
from src.utils import *
from src.regrid import regrid
import os
import ast, re
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from configargparse import ArgParser


class LRUpdate(object):
    def __init__(self, init_lr, step, divide):
        # From goo.gl/GXQaK6
        self.init_lr = init_lr
        self.step = step
        self.drop = 1. / divide

    def __call__(self, epoch):
        lr = self.init_lr * np.power(self.drop, np.floor((epoch) / self.step))
        print(f'Learning rate = {lr}')
        return lr

def load_data(var_dict, datadir, cmip, cmip_dir, train_years, valid_years, test_years,
              lead_time, batch_size, output_vars, data_subsample, norm_subsample,
              nt_in, dt_in, **kwargs):
    # Open dataset and create data generators
    if cmip:
        # Load vars
        tmp_dict = var_dict.copy()
        constants = tmp_dict.pop('constants')
        ds = xr.merge(
            [xr.open_mfdataset(f'{cmip_dir}/{var}/*.nc', combine='by_coords')
             for var in tmp_dict.keys()] +
            [xr.open_mfdataset(f'{datadir}/constants/*.nc', combine='by_coords')]
            ,
            fill_value=0  # For the 'tisr' NaNs
        )
        ds['plev'] /= 100
        ds = ds.rename({'plev': 'level'})
    else:
        ds = xr.merge(
            [xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
             for var in var_dict.keys()],
            fill_value=0  # For the 'tisr' NaNs
        )

    ds_train = ds.sel(time=slice(*train_years))
    ds_valid = ds.sel(time=slice(*valid_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = DataGenerator(
        ds_train, var_dict, lead_time, batch_size=batch_size, output_vars=output_vars,
        data_subsample=data_subsample, norm_subsample=norm_subsample, nt_in=nt_in, dt_in=dt_in
    )
    dg_valid = DataGenerator(
        ds_valid, var_dict, lead_time, batch_size=batch_size, mean=dg_train.mean, std=dg_train.std,
        shuffle=False, output_vars=output_vars, nt_in=nt_in, dt_in=dt_in
    )
    dg_test = DataGenerator(
        ds_test, var_dict, lead_time, batch_size=batch_size, mean=dg_train.mean, std=dg_train.std,
        shuffle=False, output_vars=output_vars, nt_in=nt_in, dt_in=dt_in
    )
    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')
    return dg_train, dg_valid, dg_test


def train(datadir, var_dict, output_vars, filters, kernels, lr, batch_size, early_stopping_patience, epochs, exp_id,
         model_save_dir, pred_save_dir, train_years, valid_years, test_years, lead_time, gpu,
         norm_subsample, data_subsample, lr_step, lr_divide, network_type, restore_best_weights,
         bn_position, nt_in, dt_in, use_bias, l2, skip, dropout,
         reduce_lr_patience, reduce_lr_factor, min_lr_times, unet_layers, u_skip, loss,
         cmip, cmip_dir, pretrained_model, last_pretrained_layer, last_trainable_layer, min_es_delta, optimizer):
    print(type(var_dict))

    # os.environ["CUDA_VISIBLE_DEVICES"]=str(2)
    # # Limit TF memory usage
    # limit_mem()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in gpu])
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=[f"/gpu:{i}" for i, g in enumerate(gpu)]
    )

    # Open dataset and create data generators
    dg_train, dg_valid, dg_test = load_data(var_dict, datadir, cmip, cmip_dir, train_years, valid_years, test_years,
              lead_time, batch_size, output_vars, data_subsample, norm_subsample,
              nt_in, dt_in)

    # Build model
    if pretrained_model is not None:
        pretrained_model = keras.models.load_model(
            pretrained_model, custom_objects={'PeriodicConv2D': PeriodicConv2D})

    with mirrored_strategy.scope():
        if network_type == 'resnet':
            model = build_resnet(
                filters, kernels, input_shape=(
            len(dg_train.data.lat), len(dg_train.data.lon), len(dg_train.data.level) * nt_in
        ),
                bn_position=bn_position, use_bias=use_bias, l2=l2, skip=skip,
                dropout=dropout
            )
        elif network_type == 'unet_google':
            model = build_unet_google(
                filters,
                input_shape=(len(dg_train.data.lat), len(dg_train.data.lon),len(dg_train.data.level) * nt_in),
                output_channels=len(dg_train.output_idxs),
                dropout=dropout
                )

        if pretrained_model is not None:
            # Copy over weights
            for i, l in enumerate(pretrained_model.layers):
                model.layers[i].set_weights(l.get_weights())
                if l.name == last_pretrained_layer: break

            # Set trainable to false
            if last_trainable_layer is not None:
                for l in model.layers:
                    l.trainable = False
                    if l.name == last_trainable_layer: break

        if loss == 'lat_mse':
            loss = create_lat_mse(dg_train.data.lat)
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(lr)
        elif optimizer =='adadelta':
            opt = keras.optimizers.Adadelta(lr)
        model.compile(opt, loss, metrics=['mse'])
        print(model.summary())


    # Learning rate settings
    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
          patience=early_stopping_patience,
          verbose=1,
          min_delta=min_es_delta,
          mode='auto',
          restore_best_weights=restore_best_weights
                      ))
    if reduce_lr_patience is not None:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            verbose=1,
            min_lr=reduce_lr_factor**min_lr_times*lr,
        ))
    if lr_step is not None:
        callbacks.append(keras.callbacks.LearningRateScheduler(
            LRUpdate(lr, lr_step, lr_divide)
        ))

    # Train model
    # TODO: Learning rate schedule
    history = model.fit(dg_train, epochs=epochs, validation_data=dg_valid,
                      callbacks=callbacks
                      )
    print(f'Saving model: {model_save_dir}/{exp_id}.h5')
    model.save(f'{model_save_dir}/{exp_id}.h5')
    print(f'Saving model weights: {model_save_dir}/{exp_id}_weights.h5')
    model.save_weights(f'{model_save_dir}/{exp_id}_weights.h5')
    print(f'Saving training_history: {model_save_dir}/{exp_id}_history.pkl')
    to_pickle(history.history, f'{model_save_dir}/{exp_id}_history.pkl')
    print(f'Saving norm files: {model_save_dir}/{exp_id}_mean.nc and {model_save_dir}/{exp_id}_std.nc')
    dg_train.mean.to_netcdf(f'{model_save_dir}/{exp_id}_mean.nc')
    dg_train.std.to_netcdf(f'{model_save_dir}/{exp_id}_std.nc')

    # Create predictions
    preds = create_predictions(model, dg_test)
    if len(preds.lat) != 32:
        preds = regrid(preds, ddeg_out=5.625)
    print(f'Saving predictions: {pred_save_dir}/{exp_id}.nc')
    preds.to_netcdf(f'{pred_save_dir}/{exp_id}.nc')

    # Print score in real units
    # TODO: Make flexible for other states

    if cmip: datadir = cmip_dir
    if '5.625deg' in datadir:
        valdir = datadir
    else:
        valdir = '/'.join(datadir.split('/')[:-2] + ['5.625deg/'])
    if cmip:
        z500_valid = load_test_data(
            f'{valdir}geopotential', 'z', years=slice(test_years[0], test_years[1]))
        t850_valid = load_test_data(
            f'{valdir}temperature', 't', years=slice(test_years[0], test_years[1]))
    else:
        z500_valid = load_test_data(f'{valdir}geopotential_500', 'z')
        t850_valid = load_test_data(f'{valdir}temperature_850', 't')
    try:
        print(compute_weighted_rmse(
            preds.z.sel(lev=500) if hasattr(preds, 'level') else preds.z, z500_valid
        ).load())
    except:
        print('Z500 not found in predictions.')
    try:
        print(compute_weighted_rmse(
            preds.t.sel(lev=850) if hasattr(preds, 'level') else preds.t, t850_valid
        ).load())
    except:
        print('T850 not found in predictions.')


def load_args(my_config=None):
    p = ArgParser()
    p.add_argument('-c', '--my-config', is_config_file=True, help='config file path', default=my_config)
    p.add_argument('--datadir', type=str, required=True, help='Path to data')
    p.add_argument('--exp_id', type=str, required=True, help='Experiment identifier')
    p.add_argument('--model_save_dir', type=str, required=True, help='Path to save model')
    p.add_argument('--pred_save_dir', type=str, required=True, help='Path to save predictions')
    p.add_argument('--var_dict', required=True, help='Variables: as an ugly dictionary...')
    p.add_argument('--output_vars', nargs='+', help='Output variables. Format {var}_{level}', default=None)
    p.add_argument('--filters', type=int, nargs='+', required=True, help='Filters for each layer')
    p.add_argument('--kernels', type=int, nargs='+', default=None, help='Kernel size for each layer')
    p.add_argument('--lead_time', type=int, required=True, help='Forecast lead time')
    # p.add_argument('--iterative', type=bool, default=False, help='Is iterative forecast')
    # p.add_argument('--iterative_max_lead_time', type=int, default=5*24, help='Max lead time for iterative forecasts')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # p.add_argument('--activation', type=str, default='relu', help='Activation function')
    p.add_argument('--batch_size', type=int, default=64, help='batch_size')
    p.add_argument('--epochs', type=int, default=1000, help='epochs')
    p.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    p.add_argument('--early_stopping_patience', type=int, default=None, help='Early stopping patience')
    p.add_argument('--min_es_delta', type=float, default=0, help='Minimum improvement for early stopping')
    p.add_argument('--restore_best_weights', type=int, default=1, help='ES parameter')
    p.add_argument('--reduce_lr_patience', type=int, default=None, help='Reduce LR patience')
    p.add_argument('--reduce_lr_factor', type=float, default=0.2, help='Reduce LR factor')
    p.add_argument('--min_lr_times', type=int, default=1, help='Reduce LR patience')
    p.add_argument('--lr_step', type=int, default=None, help='LR decay step')
    p.add_argument('--lr_divide', type=int, default=None, help='LR decay division factor')
    p.add_argument('--loss', type=str, default='mse', help='Loss function')
    p.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='Start/stop years for training')
    p.add_argument('--valid_years', type=str, nargs='+', default=('2016', '2016'),
                   help='Start/stop years for validation')
    p.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='Start/stop years for testing')
    p.add_argument('--data_subsample', type=int, default=1, help='Subsampling for training data')
    p.add_argument('--norm_subsample', type=int, default=1, help='Subsampling for mean/std')
    p.add_argument('--gpu', type=int, default=0, help='Which GPU', nargs='+')
    p.add_argument('--network_type', type=str, default='resnet', help='Type')
    p.add_argument('--bn_position', type=str, default=None, help='pre, mid or post')
    p.add_argument('--nt_in', type=int, default=1, help='Number of input time steps')
    p.add_argument('--dt_in', type=int, default=1, help='Time step of intput time steps (after subsampling)')
    p.add_argument('--use_bias', type=int, default=1, help='Use bias in resnet convs')
    p.add_argument('--l2', type=float, default=0, help='Weight decay')
    p.add_argument('--dropout', type=float, default=0, help='Dropout')
    p.add_argument('--skip', type=int, default=1, help='Add skip convs in resnet builder')
    p.add_argument('--u_skip', type=int, default=1, help='Add skip convs in unet')
    p.add_argument('--unet_layers', type=int, default=5, help='Number of unet layers')
    p.add_argument('--cmip', type=int, default=0, help='Is CMIP')
    p.add_argument('--cmip_dir', type=str, default=None, help='Dir for CMIP data')
    p.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    p.add_argument('--last_pretrained_layer', type=str, default=None, help='Name of last pretrained layer')
    p.add_argument('--last_trainable_layer', type=str, default=None, help='Name of last trainable layer')

    args = p.parse_args() if my_config is None else p.parse_args(args=[])
    args.var_dict = ast.literal_eval(args.var_dict)
    return vars(args)