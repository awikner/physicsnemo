# From FourCastNet repo


# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os, sys, gc, shutil
import logging
import glob
import torch
import h5py
#import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
#from torch import Tensor
#import h5py
#import math
# import cv2
#from utils.img_utils import reshape_fields
from itertools import product
from os.path import join
import cftime
from datetime import timedelta
import xarray as xr
import warnings
import datetime
import pandas as pd

def get_data_given_path(path, variables):
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables + ['time']
        } for main_key, group in f.items() if main_key in ['input']}

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0)

def get_out_path(root_dir, year, inp_file_idx):
    # year: current year
    # inp_file_idx: file index of the input in the current year
    # steps: number of steps forward
    out_path = os.path.join(root_dir, f'{year}_{inp_file_idx:04}.h5')
    return out_path



def get_data_loader(params, files_pattern, distributed, year_start, year_end, train, num_inferences = 0, validate = False, shuffle = None):

    dataset = GetDataset(params, files_pattern, year_start, year_end, train, num_inferences, validate)
    # shuffle defaults to train unless explicitly overridden
    do_shuffle = train if shuffle is None else shuffle
    sampler = DistributedSampler(dataset, shuffle=do_shuffle) if distributed else None
    if do_shuffle and not distributed:
        sampler = torch.utils.data.RandomSampler(dataset)


    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset, sampler



def get_initialization_dates(year):
    """
    Get initialization dates (Calendar dates corresponding to 
    Mondays and Thursdays of 2024 May-July) for a given year.
    """
    # Define date range from May 1 to July 31 of 2024 (template)
    start_date = datetime.datetime(2024, 5, 1)
    end_date = datetime.datetime(2024, 7, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Find Mondays (weekday=0) and Thursdays (weekday=3)
    is_monday = date_range.weekday == 0
    is_thursday = date_range.weekday == 3
    filtered_dates = date_range[is_monday | is_thursday]
    
    # Convert to the requested year
    filtered_dates_yr = pd.to_datetime(filtered_dates.strftime(f'{year}-%m-%d'))
    return filtered_dates_yr

class GetDataset(Dataset):
    def __init__(self, params, data_dir, year_start, year_end, train, num_inferences = 0, validate = False):
        self.params = params
        self.data_dir = data_dir
        self.train = train
        if not self.train:
            self.validate = validate
        else:
            self.validate = False
        if not self.train and not self.params.forecast_lead_times:
            self.params['forecast_lead_times'] = [1]
        self.epsilon_factor = self.params.epsilon_factor
        self.parallel = False #True if params['num_data_workers'] > 1 else False
        self.num_inferences = num_inferences

        #self._get_files_stats()

        self.has_year_zero = params.has_year_zero
        if hasattr(params, 'mask_fill'):
            self.mask_fill = self.params.mask_fill
        else:
            #self.mask_fill = {'lsm': 0., 'sst': 270., 'sic': 0., 'mrso': 0.}
            self.mask_fill = {'land_sea_mask': 0., 'sea_surface_temperature': 270., 'sea_ice_cover': 0., 'volumetric_soil_water_layer_1': 0.}

        self.year_start = year_start
        self.year_end = year_end
        self.calendar = params.calendar
        self.timedelta_hours = params.timedelta_hours
        self.data_timedelta_hours = params.data_timedelta_hours
        self.datetime_class  = self.datetime_class_from_calendar(self.calendar)
        # for timedelta_hours > 24
        days, hours = divmod(self.timedelta_hours, 24)
        self.timedelta = self.datetime_class(1, 1, 1 + days, hour=hours) - self.datetime_class(1, 1, 1, hour=0)
        # self.timedelta = self.datetime_class(1, 1, 1, hour=self.timedelta_hours) - self.datetime_class(1, 1, 1, hour=0)

        self.surface_variables = params.surface_variables or []
        if hasattr(params, 'land_variables'):
            if len(params.land_variables) > 0:
                if any([land_variable in self.surface_variables for land_variable in params.land_variables]):
                    raise ValueError('land variables cannot be in surface variables.')
                self.surface_variables = self.surface_variables + params.land_variables
            self.land_variables = params.land_variables
        else:
            self.land_variables = []
            

        if hasattr(params, 'ocean_variables'):
            if len(params.land_variables) > 0:
                if any([ocean_variable in self.surface_variables for ocean_variable in params.ocean_variables]):
                    raise ValueError('ocean variables cannot be in surface variables.')
                self.ocean_variables = params.ocean_variables
            self.surface_variables = self.surface_variables + params.ocean_variables
        else:
            self.ocean_variables = []
        self.upper_air_variables = params.upper_air_variables or []

        self.constant_boundary_variables = params.constant_boundary_variables or []
        self.varying_boundary_variables = params.varying_boundary_variables or []



        # self.channel_seq = self.surface_variables + self.upper_air_variables

        # self.boundary_dss = self._load_boundary_data()
        self.dates, self.start_date, self.end_date = self._get_dates(hour_step=params.data_timedelta_hours)#(hour_step=params.timedelta_hours)

        self.constant_boundary_data, self.land_mask = self._load_constant_boundary_data()
        if torch.any(torch.isnan(self.constant_boundary_data)):
            print('Constant boundary has nan')
            sys.exit(2)
        
        max_inference_idx = len(self.dates) - max(self.params.forecast_lead_times) * self.timedelta_hours // self.data_timedelta_hours
        if self.num_inferences > 0:
            self.inference_idxs = np.linspace(0, max_inference_idx, num = num_inferences + 1, dtype = int)
        else:
            self.inference_idxs = np.arange(0, max_inference_idx)
        #print('Inference idxs:')
        #print(self.inference_idxs)
        #self.data_dss = self._load_data()
        #self.lat = torch.from_numpy(self.data_dss[0].lat.values)
        if len(params['levels']) > 0:
            self.levels = np.array(params['levels'])
        else:
            #self.levels = self.data_dss[0][self.params.lev].values
            raise ValueError('levels must now be explicitly specified in config file.')
        
        
        self.surface_mean, self.surface_std = self.load_mean_std(join(
            data_dir, params.surface_mean), join(data_dir, params.surface_std), self.surface_variables, upper_air = False)

        self.upper_air_mean, self.upper_air_std = self.load_mean_std(join(
            data_dir, params.upper_air_mean), join(data_dir, params.upper_air_std), self.upper_air_variables)

        if 'surface_ff_std' in self.params:
            _, self.surface_ff_std = self.load_mean_std(join(
                data_dir, params.surface_mean), join(data_dir, params.surface_ff_std), self.surface_variables, upper_air = False)
        if 'upper_air_ff_std' in self.params:
            _, self.upper_air_ff_std = self.load_mean_std(join(
                data_dir, params.upper_air_mean), join(data_dir, params.upper_air_ff_std), self.upper_air_variables)

        if self.params.predict_delta:
            _, self.surface_delta_std = self.load_mean_std(join(
                data_dir, params.surface_mean), join(data_dir, params.surface_delta_std), self.surface_variables, upper_air = False)
            _, self.upper_air_delta_std = self.load_mean_std(join(
                data_dir, params.upper_air_mean), join(data_dir, params.upper_air_delta_std), self.upper_air_variables)

        self.varying_boundary_mean, self.varying_boundary_std = self.load_mean_std(join(data_dir, params.boundary_mean),
                                                                                   join(data_dir, params.boundary_std),
                                                                                   self.varying_boundary_variables, upper_air = False)
        
        
        if hasattr(params, 'diagnostic_variables'):
            if len(params.diagnostic_variables) > 0:
                self.diagnostic_variables = params.diagnostic_variables
                self.diagnostic_mean, self.diagnostic_std = self.load_mean_std(join(data_dir, params.diagnostic_mean),
                                                                                    join(data_dir, params.diagnostic_std),
                                                                                    self.diagnostic_variables, upper_air = False)
            else:
                self.diagnostic_variables = []
        else:
            self.diagnostic_variables = []

        self._get_variable_list()

        #self.surface_transform = self._create_surface_transform()
        #self.boundary_transform = self._create_boundary_transform()
        #self.upper_air_transform = self._create_upper_air_transform()
        #self.surface_inv_transform = self._create_surface_inv_transform()
        #self.upper_air_inv_transform = self._create_upper_air_inv_transform()

        if self.epsilon_factor > 0.:
            torch.manual_seed(0)
            
        if self.params.sel_dates and not self.train:
            years = [2019,2020,2021,2022,2023,2024]
            self.dates_all = []
            for year in years:
                dates= get_initialization_dates(year)   
                self.dates_all.extend(dates)

    def _get_variable_list(self, level_units = '.0'):
        self.variable_list_out = []
        for variable, level in product(self.upper_air_variables, self.levels):
            self.variable_list_out.append(f'{variable}_{int(level)}{level_units}')
        self.upper_air_len = len(self.variable_list_out)
        self.variable_list_out.extend(self.surface_variables)
        self.variable_list_in = self.variable_list_out.copy()
        self.variable_list_out.extend(self.diagnostic_variables)
        self.variable_list_in.extend(self.varying_boundary_variables)

    def _reshape_and_mask_variables(self, data_array, out = False):
        upper_air = torch.from_numpy(data_array[:self.upper_air_len].reshape(len(self.upper_air_variables),
                                                            len(self.levels),
                                                            self.params.horizontal_resolution[0],
                                                            self.params.horizontal_resolution[1])).to(torch.float32)
        surface = torch.from_numpy(data_array[self.upper_air_len:self.upper_air_len+len(self.surface_variables)]\
            .reshape(len(self.surface_variables),
                     self.params.horizontal_resolution[0],
                     self.params.horizontal_resolution[1])).to(torch.float32)
        surface = self._fill_mask(surface, self.surface_variables, self.land_variables + self.ocean_variables)
        if out:
            if len(self.diagnostic_variables) > 0:
                diagnostic = torch.from_numpy(data_array[self.upper_air_len+len(self.surface_variables):\
                                        self.upper_air_len+len(self.surface_variables)+len(self.diagnostic_variables)]\
                                .reshape(len(self.diagnostic_variables),
                                self.params.horizontal_resolution[0],
                                self.params.horizontal_resolution[1])).to(torch.float32)
                diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                return upper_air, surface, diagnostic
            else:
                return upper_air, surface
        else:
            if len(self.varying_boundary_variables) > 0:
                varying_boundary = torch.from_numpy(data_array[self.upper_air_len+len(self.surface_variables):\
                                        self.upper_air_len+len(self.surface_variables)+len(self.varying_boundary_variables)]\
                                .reshape(len(self.varying_boundary_variables),
                                self.params.horizontal_resolution[0],
                                self.params.horizontal_resolution[1])).to(torch.float32)
                varying_boundary = self._fill_mask(varying_boundary, self.varying_boundary_variables)
                return upper_air, surface, varying_boundary
            else:
                return upper_air, diagnostic
            

    def _fill_mask(self, data, variables, optional_variables = None):
        if optional_variables:
            for i, var in enumerate(variables):
                if var in optional_variables:
                    nans = torch.isnan(data[i])
                    if torch.any(nans):
                        data[i] = data[i].masked_fill(nans, self.mask_fill[var])
        else:
            for i, var in enumerate(variables):
                nans = torch.isnan(data[i])
                if torch.any(nans):
                    data[i] = data[i].masked_fill(nans, self.mask_fill[var])
        return data

    def datetime_class_from_calendar(self, calendar):
        datetime_class_dict = {'standard': cftime.DatetimeGregorian,
                               'Gregorian:': cftime.DatetimeGregorian,
                               'noleap': cftime.DatetimeNoLeap,
                               '365_day': cftime.DatetimeNoLeap,
                               'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
                               'all_leap': cftime.DatetimeAllLeap,
                               '366_day': cftime.DatetimeAllLeap,
                               '360_day': cftime.Datetime360Day,
                               'julian': cftime.DatetimeJulian}
        return datetime_class_dict[calendar]

    def _load_constant_boundary_data(self):
        constant_boundary_data = torch.from_numpy(self._get_data(self.start_date, variable_list = self.constant_boundary_variables)).to(torch.float32)
        constant_boundary_data = self._fill_mask(constant_boundary_data, self.constant_boundary_variables)
        land_mask = torch.clone(constant_boundary_data[np.array(self.constant_boundary_variables) == 'land_sea_mask'].detach())
        constant_boundary_mean = torch.mean(constant_boundary_data, dim=(1,2))
        constant_boundary_std = torch.std(constant_boundary_data, dim=(1,2))
        constant_boundary_data = (constant_boundary_data - constant_boundary_mean.reshape(-1, 1, 1)) / constant_boundary_std.reshape(-1, 1, 1)
        return constant_boundary_data, land_mask

    def load_mean_std(self, mean_file, std_file, datavars, upper_air = True):
        if upper_air:
            if self.params.lev == 'lev':
                with xr.open_dataset(mean_file) as ds:
                    mean = torch.stack([torch.from_numpy(ds[var].where(xr.DataArray(data=[lev in self.levels for lev in ds["Z"].values], \
                                                                                    dims = ["Z"]), drop = True).values).to(torch.float32)\
                                                                                          for var in datavars], dim=0)
                with xr.open_dataset(std_file) as ds:
                    std = torch.stack([torch.from_numpy(ds[var].where(xr.DataArray(data=[lev in self.levels for lev in ds["Z"].values], \
                                                                                    dims = ["Z"]), drop = True).values).to(torch.float32)\
                                                                                          for var in datavars], dim=0)
            elif self.params.lev == 'plev':
                with xr.open_dataset(mean_file) as ds:
                    mean = torch.stack([torch.from_numpy(ds[var].where(xr.DataArray(data=[plev in self.levels for plev in ds["Z"].values], \
                                                                                    dims = ["Z"]), drop = True).values).to(torch.float32)\
                                                                                          for var in datavars], dim=0)
                with xr.open_dataset(std_file) as ds:
                    std = torch.stack([torch.from_numpy(ds[var].where(xr.DataArray(data=[plev in self.levels for plev in ds["Z"].values], \
                                                                                    dims = ["Z"]), drop = True).values).to(torch.float32)\
                                                                                          for var in datavars], dim=0)
        else:
            with xr.open_dataset(mean_file) as ds:
                mean = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
            with xr.open_dataset(std_file) as ds:
                std = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
        return mean, std
    
    def surface_transform(self, data):
        return (data - self.surface_mean.reshape(-1, 1, 1))/self.surface_std.reshape(-1, 1, 1)
    
    def diagnostic_transform(self, data):
        return (data - self.diagnostic_mean.reshape(-1, 1, 1))/self.diagnostic_std.reshape(-1, 1, 1)
    
    def boundary_transform(self, data):
        return (data - self.varying_boundary_mean.reshape(-1, 1, 1))/self.varying_boundary_std.reshape(-1, 1, 1)
    
    def upper_air_transform(self, data):
        return (data - self.upper_air_mean.reshape(len(self.upper_air_variables), -1, 1, 1))/ \
            self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1)
    
    def surface_inv_transform(self, data):
        return data * self.surface_std.reshape(1, -1, 1, 1) + self.surface_mean.reshape(1, -1, 1, 1)
    
    def upper_air_inv_transform(self, data):
        return data * self.upper_air_std.reshape(1, len(self.upper_air_variables), -1, 1, 1) + \
            self.upper_air_mean.reshape(1, len(self.upper_air_variables), -1, 1, 1)
    
    def diagnostic_inv_transform(self, data):
        return data * self.diagnostic_std.reshape(1, -1, 1, 1) + self.diagnostic_mean.reshape(1, -1, 1, 1)

    def surface_delta_transform(self, data):
        return data / self.surface_delta_std.reshape(-1, 1, 1)
    
    def upper_air_delta_transform(self, data):
        return data / self.upper_air_delta_std.reshape(len(self.upper_air_variables), -1, 1, 1)

    # Modification for the autoregressive parameter
    def _get_dates(self, hour_step=6.):

        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1) 

        if not self.train:
            hours = (end_date - start_date).days * 24. #- (max(self.params.forecast_lead_times)) * hour_step
        else:
            # Training mode
            hours = (end_date - start_date).days * 24.
        
        date_range = np.arange(0., hours, hour_step)
        #print(f'End data hour: {date_range[-1]}')
        return date_range, start_date, end_date

    def _get_data(self, data_datetime, out = False, variable_list = None):
        data_year = data_datetime.year
        data_idx = int((data_datetime - self.datetime_class(data_year, 1, 1, hour=0, has_year_zero=self.has_year_zero)).total_seconds())\
              // 3600 // self.data_timedelta_hours
        data_file_path = get_out_path(self.data_dir, data_year, data_idx)
        if variable_list:
            raw_data = get_data_given_path(data_file_path, variable_list)
        else:
            if out:
                raw_data = get_data_given_path(data_file_path, self.variable_list_out)
            else:
                raw_data = get_data_given_path(data_file_path, self.variable_list_in)
        return raw_data

    def __len__(self):
        if self.params.sel_dates and not self.train:
            return len(self.dates_all)
        return len(self.inference_idxs)


    def __getitem__(self, index):
        #print('Loaded Boundary Data')
        #self.dates = self._get_dates(hour_step=params.timedelta_hours)
        #self.data_dss = self._load_data(initial=False)
        #self.lat = torch.from_numpy(self.data_dss[0].lat.values)
        #self.lev = torch.from_numpy(self.data_dss[0].lev.values)
        lead_times = self.params.forecast_lead_times

        # Condition 1: Training
        if self.train:
            start_time = self.start_date + timedelta(hours=self.dates[index])
            end_time = self.start_date + timedelta(hours=self.dates[index] + self.timedelta_hours)
            data_in  = self._get_data(start_time, out = False)
            data_out = self._get_data(end_time, out = True)
            if len(self.varying_boundary_variables) > 0:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out = False)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out = False)
            if len(self.diagnostic_variables) > 0:
                upper_air_t_1, surface_t_1, diagnostic_t_1 = self._reshape_and_mask_variables(data_out, out = True)
            else:
                upper_air_t_1, surface_t_1 = self._reshape_and_mask_variables(data_out, out = True)
            
            if self.params.predict_delta:
                surface_t_1 = surface_t_1 - surface_t
                upper_air_t_1 = upper_air_t_1 - upper_air_t
                surface_t = self.surface_transform(surface_t)
                surface_t_1 = self.surface_delta_transform(surface_t_1)
                upper_air_t = self.upper_air_transform(upper_air_t)
                upper_air_t_1 = self.upper_air_delta_transform(upper_air_t_1)
            else:
                surface_t = self.surface_transform(surface_t)
                surface_t_1 = self.surface_transform(surface_t_1)
                upper_air_t = self.upper_air_transform(upper_air_t)
                upper_air_t_1 = self.upper_air_transform(upper_air_t_1)
            if len(self.diagnostic_variables) > 0:
                diagnostic_t_1 = self.diagnostic_transform(diagnostic_t_1)
            varying_boundary_data = self.boundary_transform(varying_boundary_data)
            #print('Normalized Boundary')
            if self.epsilon_factor > 0.:
                if 'surface_ff_std' in self.params:
                    surface_t_noise = torch.randn(*surface_t.shape) * (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(len(self.surface_variables), 1, 1)
                else:
                    surface_t_noise = torch.randn(*surface_t.shape) * self.epsilon_factor
                surface_t = surface_t + surface_t_noise
                if 'upper_air_ff_std' in self.params:
                    upper_air_t_noise = torch.randn(*upper_air_t.shape) * (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(len(self.upper_air_variables), len(self.levels), 1, 1)
                else:
                    upper_air_t_noise = torch.randn(*upper_air_t.shape) * self.epsilon_factor
                upper_air_t = upper_air_t + upper_air_t_noise
        
        # Condition for autoregression
        elif lead_times:
            if self.params.sel_dates and not self.train:
                start_time = self.dates_all[index]
                print("select the fixed dates")
                
                #print("start_time",start_time)
            else:
                
                start_time = self.start_date + timedelta(hours=self.dates[index])
                print("start_time",start_time)

            # Load initial conditions
            data_in = self._get_data(start_time, out = False)
            if len(self.varying_boundary_variables) > 0:
                upper_air_t, surface_t, varying_boundary_data_t = self._reshape_and_mask_variables(data_in, out = False)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out = False)

            max_lead_time = lead_times[-1]
            boundary_times = [start_time + timedelta(hours=self.timedelta_hours * lead_time) for lead_time in range(max_lead_time)]
            start_time_tensor = torch.tensor([start_time.year, start_time.month, start_time.day, start_time.hour])
            varying_boundary_data = [varying_boundary_data_t]
            varying_boundary_data.extend([self._fill_mask(\
                torch.from_numpy(self._get_data(boundary_time, variable_list = self.varying_boundary_variables)).to(torch.float32), self.varying_boundary_variables) for boundary_time in boundary_times])
            varying_boundary_data = torch.stack([self.boundary_transform(varying_boundary_data_i) for varying_boundary_data_i in varying_boundary_data], dim=0)


            if self.validate: 
                # Load targets for each time step up to the maximum lead time
                targets_surface = []
                targets_upper_air = []
                if self.params.predict_delta:
                    targets_delta_surface = []
                    targets_delta_upper_air = []
                if len(self.diagnostic_variables) > 0:
                    targets_diagnostic = []

                # Iterate over each time step up to the maximum lead time
                max_lead_time = lead_times[-1]

                for step in range(1, max_lead_time + 1):
                    target_time = start_time + timedelta(hours = self.timedelta_hours * step)
                    raw_target_data = self._get_data(target_time, out = True)

                    if len(self.diagnostic_variables) > 0:
                        upper_air_target, surface_target, diagnostic_target = self._reshape_and_mask_variables(raw_target_data, out = True)
                        targets_diagnostic.append(diagnostic_target)
                    else:
                        upper_air_target, surface_target = self._reshape_and_mask_variables(raw_target_data, out = True)
        
                    targets_surface.append(surface_target)
                    targets_upper_air.append(upper_air_target)


                for step in range(0, max_lead_time):
                    targets_surface[step] = self.surface_transform(targets_surface[step])
                    targets_upper_air[step] = self.upper_air_transform(targets_upper_air[step])
                    if len(self.diagnostic_variables) > 0:
                        targets_diagnostic[step] = self.diagnostic_transform(targets_diagnostic[step])
                
                surface_t = self.surface_transform(surface_t)
                upper_air_t = self.upper_air_transform(upper_air_t)

                targets_surface = torch.stack(targets_surface, dim=0)
                targets_upper_air = torch.stack(targets_upper_air, dim=0)
                if len(self.diagnostic_variables) > 0:
                    targets_diagnostic = torch.stack(targets_diagnostic, dim=0)

                    

        else:
            start_time = self.start_date + timedelta(hours=self.dates[index])
            data_in = self._get_data(start_time, out = False)
            if len(self.varying_boundary_variables) > 0:
                surface_t, upper_air_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
                varying_boundary_data = self.boundary_transform(varying_boundary_data).unsqueeze(0)
            else:
                surface_t, upper_air_t = self._reshape_and_mask_variables(data_in, out=False)
            surface_t = self.surface_transform(surface_t)
            upper_air_t = self.upper_air_transform(upper_air_t)
        if torch.any(torch.isnan(varying_boundary_data)):
            print('Boundary data has nan')
            sys.exit(2)
        if torch.any(torch.isnan(surface_t)):
            print('Surface t has nan')
            sys.exit(2)
        if torch.any(torch.isnan(upper_air_t)):
            print('Upper air t has nan')
            sys.exit(2)

        if self.train:
            if torch.any(torch.isnan(surface_t_1)):
                print('Surface t+1 has nan')
                sys.exit(2)
            if torch.any(torch.isnan(upper_air_t_1)):
                print('Upper air t+1 has nan')
                sys.exit(2)
            if len(self.diagnostic_variables) > 0:
                if torch.any(torch.isnan(diagnostic_t_1)):
                    print('Diagnostic has nan')
                    sys.exit(2)
            if len(self.diagnostic_variables) > 0:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, diagnostic_t_1, varying_boundary_data
            else:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data
        ### ERROR - Need to have data loader return times for validation
        elif self.validate and lead_times:
            if self.params.predict_delta:
                if len(self.diagnostic_variables) > 0:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, targets_diagnostic, targets_delta_surface, targets_delta_upper_air, \
                        varying_boundary_data, start_time_tensor
                else:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, varying_boundary_data, targets_delta_surface, targets_delta_upper_air, start_time_tensor
            else:
                if len(self.diagnostic_variables) > 0:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, targets_diagnostic, \
                        varying_boundary_data, start_time_tensor
                else:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, varying_boundary_data, start_time_tensor
        elif lead_times:
            return surface_t, upper_air_t, varying_boundary_data
        else:
            if len(self.diagnostic_variables) > 0:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, diagnostic_t_1, varying_boundary_data
            else:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data
            
def get_infer_data(params, files_pattern, distributed, year_start, year_end, step=100, num_inferences = 0, validate = False):

    dataset = GetDataset(params, files_pattern, year_start, year_end, False, num_inferences, validate)
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,  # (sampler is None),
                            sampler=None,# if train else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    return dataloader, dataset