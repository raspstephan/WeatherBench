import re
import numpy as np
import xarray as xr
import tensorflow.keras as keras
import datetime
from src.utils import *
import pdb
import logging
import pandas as pd
import tensorflow.keras.utils as np_utils

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True,
                 mean=None, std=None, output_vars=None, data_subsample=1, norm_subsample=1,
                 nt_in=1, dt_in=1, cont_time=False, fixed_time=False, multi_dt=1, verbose=0, is_categorical=False, num_bins=20):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
            data_subsample: Only take every ith time step
            norm_subsample: Same for normalization. This is AFTER data_subsample!
            nt_in: How many time steps for input. AFTER data_subsample!
            dt_in: Interval of input time steps. AFTER data_subsample!
        """
        if verbose: print('DG start', datetime.datetime.now().time())
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.nt_in = nt_in
        self.dt_in = dt_in
        self.nt_offset = (nt_in - 1) * dt_in
        self.cont_time = cont_time
        self.fixed_time = fixed_time
        self.multi_dt = multi_dt
        self.is_categorical= is_categorical
        self.num_bins = num_bins #for categorical
        

        data = []
        level_names = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for long_var, params in var_dict.items():
            if long_var == 'constants':
                for var in params:
                    data.append(ds[var].expand_dims(
                        {'level': generic_level, 'time': ds.time}, (1, 0)
                    ))
                    level_names.append(var)
            else:
                var, levels = params
                try:
                    data.append(ds[var].sel(level=levels))
                    level_names += [f'{var}_{level}' for level in levels]
                except ValueError:
                    data.append(ds[var].expand_dims({'level': generic_level}, 1))
                    level_names.append(var)

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.data['level_names'] = xr.DataArray(
            level_names, dims=['level'], coords={'level': self.data.level})
        if output_vars is None:
            self.output_idxs = range(len(self.data.level))
        else:
            self.output_idxs = [i for i, l in enumerate(self.data.level_names.values)
                                if any([bool(re.match(o, l)) for o in output_vars])]
        self.const_idxs = [i for i, l in enumerate(self.data.level_names) if l in var_dict['constants']]
        self.not_const_idxs = [i for i, l in enumerate(self.data.level_names) if l not in var_dict['constants']]

        # Subsample
        self.data = self.data.isel(time=slice(0, None, data_subsample))
        self.dt = self.data.time.diff('time')[0].values / np.timedelta64(1, 'h')


        # Normalize
        if verbose: print('DG normalize', datetime.datetime.now().time())
        self.mean = self.data.isel(time=slice(0, None, norm_subsample)).mean(
            ('time', 'lat', 'lon')).compute() if mean is None else mean
        #         self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        self.std = self.data.isel(time=slice(0, None, norm_subsample)).std(
            ('time', 'lat', 'lon')).compute() if std is None else std
        self.data = (self.data - self.mean) / self.std

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if verbose: print('DG load', datetime.datetime.now().time())
        if load:
            if verbose: print('Loading data into RAM')
            self.data.load()
        if verbose: print('DG done', datetime.datetime.now().time())

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.idxs) / self.batch_size))
    
    @property
    def shape(self):
        return len(self.data.lat), len(self.data.lon), len(self.data.level) * self.nt_in + self.cont_time

    @property
    def nt(self):
        assert (self.lead_time / self.dt).is_integer(), "lead_time and dt not compatible."
        return int(self.lead_time / self.dt)

    @property
    def init_time(self):
        return self.data.isel(time=slice(self.nt_offset, -self.nt)).time

    @property
    def valid_time(self):
        start = self.nt+self.nt_offset
        stop = None
        if self.multi_dt > 1:
            diff = self.nt - self.nt // self.multi_dt
            start -= diff; stop = -diff
        return self.data.isel(time=slice(start, stop)).time

    @property
    def n_samples(self):
        return self.data.isel(time=slice(0, -self.nt)).shape[0]

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        if self.cont_time:
            if not self.fixed_time:
                nt = np.random.randint(1, self.nt, len(idxs))
            else:
                nt = np.ones(len(idxs), dtype='int') * self.nt
            ftime = (nt * self.dt / 100)[:, None, None] * np.ones((1, len(self.data.lat),
                                                                      len(self.data.lon)))
        else:
            nt = self.nt
        X = self.data.isel(time=idxs).values.astype('float32')
        if self.multi_dt > 1: consts = X[..., self.const_idxs]
        if self.nt_in > 1:
            X = np.concatenate([
                self.data.isel(time=idxs-nt_in*self.dt_in).values
                                   for nt_in in range(self.nt_in-1, 0, -1)
            ] + [X], axis=-1).astype('float32')
        if self.multi_dt > 1:
            X = [X[..., self.not_const_idxs], consts]
            step = self.nt // self.multi_dt
            y = [
                self.data.isel(time=idxs + nt, level=self.output_idxs).values.astype('float32')
                for nt in np.arange(step, self.nt+step, step)
            ]
        else:
            y = self.data.isel(time=idxs + nt, level=self.output_idxs).values.astype('float32')
        if self.cont_time: 
            X = np.concatenate([X, ftime[..., None]], -1).astype('float32')
        
        if self.is_categorical==True:
            #converting y to categorical
            bin_min=-5
            bin_max=5 #can also put as input
            bins=np.linspace(bin_min,bin_max,self.num_bins+1)
            bins[0]=-np.inf; bins[-1]=np.inf #for rare out-of-bound cases.can improve
            
            y1=pd.cut(y.reshape(-1), bins, labels=False)
            y1=y1.reshape(y.shape)
            y1=np_utils.to_categorical(y1, num_classes=self.num_bins)
            return X, y1
        
        else:
            return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.nt_offset, self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.idxs)


class CombinedDataGenerator(keras.utils.Sequence):

    def __init__(self, dgs, batch_size):
        self.dgs = dgs
        self.lens = np.array([len(dg.idxs) for dg in self.dgs])
        self.data = self.dgs[0].data
        self.batch_size = batch_size
        self.bss = np.round(self.lens / self.lens.sum() * batch_size)
        missing = batch_size - self.bss.sum()
        self.bss[0] += missing
        assert self.bss.sum() == batch_size, 'Batch sizes dont add up'
        print('Individual batch sizes:', self.bss)
        for dg, bs in zip(dgs, self.bss): dg.batch_size = int(bs)

    def __len__(self):
        total_samples = np.sum([len(dg.idxs) for dg in self.dgs])
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, i):
        Xs = []
        ys = []
        for dg in self.dgs:
            X, y = dg[i]
            Xs.append(X)
            ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    def on_epoch_end(self):
        for dg in self.dgs:
            dg.on_epoch_end()


def create_predictions(model, dg, multi_dt=False, parametric=False):
    """Create non-iterative predictions"""
    level_names = dg.data.isel(level=dg.output_idxs).level_names
    level = dg.data.isel(level=dg.output_idxs).level
    if parametric:
        # pdb.set_trace()
        lvl = level_names.values
        mm, ss = [], []
        for l in lvl:
            m = l.split('_'); s = l.split('_')
            m[0] += '-mean'; s[0] += '-std'
            mm.append('_'.join(m)); ss.append('_'.join(s))
        lvl = mm + ss
        level_names = xr.concat([level_names]*2, dim='level')
        level_names[:] = lvl
        level = xr.concat([level]*2, dim='level')

    preds = xr.DataArray(
        model.predict(dg)[0] if multi_dt else model.predict(dg),
        dims=['time', 'lat', 'lon', 'level'],
        coords={'time': dg.valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon,
                'level': level,
                'level_names': level_names
                },
    )
    # Unnormalize
    mean = dg.mean.isel(level=dg.output_idxs).values
    std = dg.std.isel(level=dg.output_idxs).values
    if parametric:
        mean = np.concatenate([mean, np.zeros_like(mean)])
        std = np.concatenate([std]*2)
    preds = preds * std + mean

    unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values]))

    das = []
    for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] in v]
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if not 'level' in da.dims: da = da.drop('level')
        das.append({v: da})
    return xr.merge(das)

def create_cont_predictions(model, dg, max_lead_time=120, dt=12, lead_time=None):
    dg.fixed_time = True
    lead_time = np.arange(dt, max_lead_time+dt, dt) if lead_time is None else lead_time
    lead_time = xr.DataArray(lead_time, dims={'lead_time': lead_time}, name='lead_time')
    preds = []
    for l in tqdm(lead_time):
        dg.lead_time = l.values; dg.on_epoch_end()
        p = create_predictions(model, dg)
        p['time'] = dg.init_time
        preds.append(p)
    return xr.concat(preds, lead_time)

# TODO: Outdated
# def create_iterative_predictions(model, dg, max_lead_time=5 * 24):
#     """Create iterative predictions"""
#     state = dg.data[:dg.n_samples]
#     preds = []
#     for _ in range(max_lead_time // dg.lead_time):
#         state = model.predict(state)
#         p = state * dg.std.values + dg.mean.values
#         preds.append(p)
#     preds = np.array(preds)
#
#     lead_time = np.arange(dg.lead_time, max_lead_time + dg.lead_time, dg.lead_time)
#     das = []
#     lev_idx = 0
#     for var, levels in dg.var_dict.items():
#         if levels is None:
#             das.append(xr.DataArray(
#                 preds[:, :, :, :, lev_idx],
#                 dims=['lead_time', 'time', 'lat', 'lon'],
#                 coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
#                 name=var
#             ))
#             lev_idx += 1
#         else:
#             nlevs = len(levels)
#             das.append(xr.DataArray(
#                 preds[:, :, :, :, lev_idx:lev_idx + nlevs],
#                 dims=['lead_time', 'time', 'lat', 'lon', 'level'],
#                 coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon,
#                         'level': levels},
#                 name=var
#             ))
#             lev_idx += nlevs
#     return xr.merge(das)