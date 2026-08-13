
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from collections import OrderedDict
import torch
import matplotlib
import os
import matplotlib.colors as mcolors
#from dask.diagnostics import ProgressBar
import pandas as pd
import matplotlib.animation as animation
import cartopy.crs as ccrs
# sns.set_style('darkgrid')
# sns.set_context('notebook')


matplotlib.use('Agg')  # Set the backend to 'Agg'


# Assume field is your xarray DataArray with dimensions (lat, lon)
# Load or define your field DataArray here

def zonal_averaged_power_spectrum(field, time_avg=True):
    """
    This function calculates the zonal averaged power spectrum of a given field. It is designed to work with xarray DataArrays or Datasets that have 'lat', 'lon', and optionally 'time' dimensions. The function first transposes the dimensions to ensure 'lat' and 'lon' are the first two dimensions, then performs a Fast Fourier Transform (FFT) along the 'lon' axis to compute the power spectrum. The power spectrum is then averaged over 'lat' and 'time' (if present) to produce the zonal averaged power spectrum.

    Parameters:
    - field (xarray.DataArray or xarray.Dataset): The input field for which to calculate the zonal averaged power spectrum.

    Returns:
    - power_spectrum_avg (xarray.DataArray): The zonal averaged power spectrum of the input field.
    """
    # but work if i have several variables on one plev or lead_time 
    initial_field = field.copy()
   
    # if field is a xr.Dataset:
    if isinstance(field, xr.Dataset):
        vars = list(initial_field.data_vars)
        field = field.to_array(dim='var')
        print("Dataset detected. Converting to a DataArray with the first dimension being the variable.")
        print("Shape:", field.shape)
        print("Variables detected:", vars)
    else: 
        vars=None
        print("DataArray detected.")
    
    field = field.transpose('lon', 'lat', ...)
    if 'time' in field.coords:
        field = field.transpose('lon', 'lat', 'time', ...)
    dims = list(field.dims)
    print("Dimensions detected:", dims)
    if not 'lat' in dims or not 'lon' in dims:
        raise ValueError("Latitude and longitude coordinates must be present in the field.")
    # create a dict to store the coordinates
    coords = OrderedDict()
    for dim in dims:
        coords[dim] = np.array(field[dim])
    n_lon = len(coords['lon'])
    

    ###########################################################################################
    field_fft = np.fft.rfft(field, axis=0, norm='forward') # Convention used: the first Fourier coefficient is the mean of the field

    # Compute the power spectrum (squared magnitude of Fourier coefficients)
    power_spectrum = np.abs(field_fft)**2

    # Define the zonal wavenumbers
    nx = n_lon
    print("n_x =" , nx)
    k_x = np.fft.fftfreq(nx, d=1/nx)

    # Only take the positive frequencies (or the first half if using real FFT)
    k_x = k_x[:nx//2]
    power_spectrum = power_spectrum[:nx//2]
    # count the positive frequencies twice except for the first one (zero frequency), because the FFT of a real function is symmetric
    power_spectrum[1:] *= 2
    # multiply by a factor cos(pi latitude[i] / 180) in axis 1
    # C0 = 40.075*10**6 # Earth's circumference in meters
    weights = np.cos(np.pi * coords['lat']/180) # * C0
    weights = weights.reshape(1, -1, *([1] * (power_spectrum.ndim - 2)))
    power_spectrum *= weights 

    # Average the power spectrum over latitudes and time (axis 1 and 2)
    if 'time' not in coords or time_avg==False:
        power_spectrum_avg = power_spectrum.mean(axis=1)
    else:
        print("Warning: 'time' dimension detected. Averaging over the time dimension.")
        power_spectrum_avg = power_spectrum.mean(axis=(1, 2))

       # print(len(initial_field.coords))
    print("Shape after averaged FFT: ", power_spectrum_avg.shape)
    ################################################################################################

    # convert to xarray Dataset or DataArray
    new_coords = coords.copy()
    # replace lon by k_x (at same position in the ord dict)
    new_coords['lon'] = k_x
    # drop the lat dimension
    new_coords.pop('lat')
    if 'time' in new_coords and time_avg:
        new_coords.pop('time')
    # print("New coordinates:", new_coords)
    power_spectrum_avg = xr.DataArray(power_spectrum_avg, coords=new_coords.values(), dims=new_coords.keys())
    if vars is not None:
        power_spectrum_avg = power_spectrum_avg.to_dataset(dim='var')
        # rename 'lon' to 'k_x'
    power_spectrum_avg = power_spectrum_avg.rename({'lon':'k_x'})

    return k_x, power_spectrum_avg


# Amaury's code
def plot_power_spectrum(power_spectrum_avg_preds, preds_times, vars=["temperatire", "geopotential", "u_component_of_wind"],
                         plevs = [850, 500, 250], lead_times=[6, 48, 120]):
    """ Plot the power spectrum of the ground truth and the forecast
    :param power_spectrum_avg: xarray dataset, power spectrum of the ground truth
    :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
    :param preds_times: array, time values of the forecast
    :param name_fc: str, name of the forecast
    """
    # Check that len(vars) == len(plevs)
    assert len(vars) == len(plevs), 'vars and plevs must have the same length'

    # Loop through variables and pressure levels to plot
    fig, axs = plt.subplots(len(lead_times), len(vars), figsize=(18, 20))
    # k_x_gt = power_spectrum_avg_gt.k_x.values
    k_x_preds = power_spectrum_avg_preds.k_x.values
    for i, lead_time in enumerate(lead_times):
        for j, (var, plev) in enumerate(zip(vars, plevs)):
            power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time).sel(lev=plev)
            # power_spectrum_avg2 = power_spectrum_avg_gt[var].sel(time=preds_times + timedelta(hours=lead_time)).mean('time').sel(plev=plev)

            # axs[i,j].plot(k_x_gt, power_spectrum_avg2, label='Ground Truth')
            axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2)
            axs[i,j].legend()
            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
            axs[i,j].set_ylabel('Energy Spectrum')
            if i==0:
                axs[i,j].set_title(f"var = '{var}' at {int(plev)} hPa, lead time = {lead_time} hours")
            else:
                axs[i,j].set_title(f"var = '{var}' at {int(plev)} hPa, lead time = {lead_time//24} days")
            axs[i,j].grid(True)

    plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum", y = 1.01)
    plt.tight_layout() 
    plt.savefig(f"spectrum_results.png", pad_inches=0.1, bbox_inches='tight')
    return fig, axs



def plot_power_spectrum_test(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times,
                             vars=["temperatire", "geopotential", "u_component_of_wind"], 
                             plevs=[850, 500, 250]):
    """ Plot the power spectrum of the forecast and ground truth
    :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
    :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
    :param preds_times: array, time values of the forecast
    :param filename: str, path to save the plot
    :param lead_times: list, lead times in hours to plot
    :param vars: list, variables to plot (default: ["ta", "zg", "ua"])
    :param plevs: list, pressure levels in Pa to plot (default: [850*100, 500*100, 250*100])
    """
    # Filter variables that are present in the data
    available_vars = [var for var in vars if var in power_spectrum_avg_preds.data_vars]
    if not available_vars:
        raise ValueError(f"None of the specified variables {vars} are present in the data. Available variables: {list(power_spectrum_avg_preds.data_vars)}")

    # Filter pressure levels that are present in the data
    available_plevs = [plev for plev in plevs if plev in power_spectrum_avg_preds.plev.values]
    if not available_plevs:
        raise ValueError(f"None of the specified pressure levels {plevs} are present in the data. Available levels: {power_spectrum_avg_preds.plev.values}")

    # Handle lead times as specified by the user
    available_lead_times = power_spectrum_avg_preds.lead_time.values
    print(f" Available lead times: {available_lead_times}")
    plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
    if not plot_lead_times:
        raise ValueError(f"None of the specified lead times {lead_times} are present in the data. Available lead times: {available_lead_times}")

    # Create subplots
    # fig, axs = plt.subplots(len(plot_lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
    fig, axs = plt.subplots(len(available_vars), len(plot_lead_times), figsize=(18, 20), squeeze=False)

    k_x_preds = power_spectrum_avg_preds.k_x.values
    k_x_gt = power_spectrum_avg_gt.k_x.values

    for i, var in enumerate(available_vars):
        for j, lead_time in enumerate(plot_lead_times):
            for plev in available_plevs:
                try:
                    power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, plev=plev, method='nearest')
                    axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast {plev:.0f} hPa')

                    
                    # Add ground truth plot
                    power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, plev=plev, method='nearest')
                    axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth {plev:.0f} hPa')
                except KeyError as e:
                    print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, pressure level={plev}. Error: {e}")
                    continue

            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
            axs[i,j].set_ylabel('Energy Spectrum')
            axs[i,j].set_title(f"var = '{var}', lead time = {lead_time} hours")
            axs[i,j].grid(True)
            axs[i,j].legend()

    # for i, lead_time in enumerate(plot_lead_times):
    #     for j, var in enumerate(available_vars):
    #         for plev in available_plevs:
    #             try:
    #                 power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, plev=plev, method='nearest')
    #                 axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast {plev/100:.0f} hPa')

                    
    #                 # Add ground truth plot
    #                 power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, plev=plev, method='nearest')
    #                 axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth {plev/100:.0f} hPa')
    #             except KeyError as e:
    #                 print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, pressure level={plev}. Error: {e}")
    #                 continue

    #         axs[i,j].set_yscale('log')
    #         axs[i,j].set_xscale('log')
    #         axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
    #         axs[i,j].set_ylabel('Energy Spectrum')
    #         axs[i,j].set_title(f"var = '{var}', lead time = {lead_time} hours")
    #         axs[i,j].grid(True)
    #         axs[i,j].legend()
    
    plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum (Pressure Levels)", y=1.01)
    plt.tight_layout() 
    plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
    plt.close(fig)

    return fig, axs
def plot_acc_over_lead_time(acc, lead_times_hours, vars=["tas", "ta", "zg", "ua"], plevs=[None, 850, 500, 250], 
                            colors=None, fontsize_title=14):
    """
    Plot the ACC over lead time for each variable and pressure level
    :param acc: OrderedDict or xr.Dataset, ACC scores
    :param lead_times_hours: list, lead times in hours
    :param vars: list, variables to plot
    :param plevs: list, pressure levels to plot (use None for surface variables)
    :param colors: dict, colors for each model
    :param fontsize_title: int, font size for the title
    """
    if isinstance(acc, xr.Dataset):
        acc = {'Model': acc}

    if colors is None:
        colors = {'Pangu': 'blue'}

    fig, axs = plt.subplots(len(vars), 1, figsize=(12, 5*len(vars)), squeeze=False)

    for i, (var, plev) in enumerate(zip(vars, plevs)):
        ax = axs[i, 0]
        
        if plev is None:
            title = f'ACC for {var}'
        else:
            title = f'ACC for {var} at {plev:.0f} hPa'
        
        for model, ds in acc.items():
            if var in ds:
                if 'plev' in ds[var].dims and plev is not None:
                    data = ds[var].sel(plev=plev, method='nearest')
                elif 'plev' not in ds[var].dims:
                    data = ds[var]
                else:
                    print(f"Warning: {var} has unexpected dimensions. Skipping.")
                    continue
                
                ax.plot(lead_times_hours, data.values, label=model, color=colors[model], marker='o')

        ax.set_ylabel(f'{var} ACC')
        ax.set_title(title, fontsize=fontsize_title)
        ax.set_ylim(-0.3, 1.1)
        ax.axhline(0, ls='--', c='0.', lw=1)
        ax.axhline(0.6, ls='--', c='r', lw=1, label='ACC = 0.6')  # Add horizontal line at ACC = 0.6
        ax.set_xlabel('Lead time [days]')
        lead_times_ticks = np.arange(0, max(lead_times_hours)+1, 24)
        ax.set_xticks(lead_times_ticks)
        ax.set_xticklabels(['%d' % i for i in range(len(lead_times_ticks))])
        ax.legend(loc='lower left')

    plt.tight_layout()
    return fig, axs




import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import numpy as np
import time




# THIS IS THE GIF OF ANOMOLIES

from datetime import timedelta

def make_gif(combined_dataset, gt_combined_dataset, climatology, name_fc, var, output_filename, sample_index=0, plev=None):
    """
    Create a gif of the forecast anomalies for a single sample, evolving over all time steps up the maximum lead time,
    without using coastlines.
    """
    start_time = time.time()
    print(f"Starting GIF creation for {var} anomalies")

    # Data selection and setup
    if plev is not None:
        data_inference = combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
        data_gt = gt_combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
        climatology_data = climatology[var].sel(plev=plev, method='nearest')
    else:
        data_inference = combined_dataset[var].isel(time=sample_index)
        data_gt = gt_combined_dataset[var].isel(time=sample_index)
        climatology_data = climatology[var]

    print(f"Data shape - Inference: {data_inference.shape}, Ground Truth: {data_gt.shape}")
    print(f"Inference dimensions: {data_inference.dims}")
    print(f"Ground Truth dimensions: {data_gt.dims}")
    print(f"Climatology dimensions: {climatology_data.dims}")
    print(f"Lead times: {data_inference.lead_time.values}")
    
    # Get the start time for this sample from the original dataset
    start_datetime = combined_dataset.time.values[sample_index]
    print(f"Start datetime for sample {sample_index}: {start_datetime}")

    # Calculate time range for all lead times
    time_range = [start_datetime + timedelta(hours=int(lt)) for lt in data_inference.lead_time.values]

    # Prepare climatology
    if 'zsfc' in climatology_data:
        climatology_data = climatology_data.drop_vars('zsfc')
    
    # Ensure climatology has the correct spatial dimensions
    climatology_data = climatology_data.transpose('dayofyear', 'lat', 'lon')

    
    # print_info(climatology_aligned, "Aligned Climatology")
    climatology_data = climatology_data.assign_coords(lat=data_inference.lat)

    # Initialize anomalies arrays
    anomalies_inference = xr.zeros_like(data_inference)
    anomalies_gt = xr.zeros_like(data_gt)

    # Calculate anomalies for each lead time
    for i, forecast_datetime in enumerate(time_range):
        # Get the day of year for this forecast time
        forecast_doy = forecast_datetime.dayofyr
        
        # Select the corresponding climatology
        clim_for_leadtime = climatology_data.sel(dayofyear=forecast_doy)
        
        # Calculate anomalies for this lead time
        anomalies_inference[i] = data_inference[i] - clim_for_leadtime
        anomalies_gt[i] = data_gt[i] - clim_for_leadtime

        print(f"Processed lead time {data_inference.lead_time.values[i]} hours, forecast date: {forecast_datetime}")



    # vmin = float(anomalies_gt.min())
    # vmax = float(anomalies_gt.max())
    # print(f"Anomaly value range: {vmin} to {vmax}")

    max_abs_anomaly = abs(anomalies_gt).max()
    vmin = -max_abs_anomaly
    vmax = max_abs_anomaly
    print(f"Anomaly value range: {vmin} to {vmax}")

    

    # # Calculate symmetric range for anomalies
    # max_abs_anomaly = abs(anomalies_gt).max()
    # vmin = -max_abs_anomaly
    # vmax = max_abs_anomaly
    # print(f"Anomaly value range: {vmin} to {vmax}")

    # Figure setup
    fig_gif, axs = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    print("Figure created")

    # Create initial plots and colorbars
    im1 = axs[0].pcolormesh(anomalies_inference.lon, anomalies_inference.lat, 
                            anomalies_inference.isel(lead_time=0),
                            transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    im2 = axs[1].pcolormesh(anomalies_gt.lon, anomalies_gt.lat, 
                            anomalies_gt.isel(lead_time=0),
                            transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
    # Add colorbars
    plt.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.05, label='Anomaly')
    plt.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.05, label='Anomaly')

    axs[0].set_global()
    axs[1].set_global()
    axs[0].set_title(f'{name_fc} Anomaly')
    axs[1].set_title('Ground Truth Anomaly')

    frame_times = []

    def plot(i):
        frame_start = time.time()

        lead_time = data_inference.lead_time.values[i]
        current_time = time_range[i]

        forecast = anomalies_inference.isel(lead_time=i)
        truth = anomalies_gt.isel(lead_time=i)

        # Update plot data
        im1.set_array(forecast.values.ravel())
        im2.set_array(truth.values.ravel())

        var_up = f'{var}_{plev:.0f}hPa' if plev is not None else var
        title = f'{var_up} Anomaly at {current_time} (Lead time: {lead_time} hours, Sample {sample_index})'
        plt.suptitle(title, y=0.95)

        frame_end = time.time()
        frame_times.append(frame_end - frame_start)
        print(f"Frame {i} completed in {frame_end - frame_start:.2f} seconds")

    print("Starting animation creation")
    ani = animation.FuncAnimation(fig_gif, plot, frames=len(anomalies_inference.lead_time), repeat=False)
    
    print("Saving animation")
    ani.save(output_filename, writer='pillow', fps=1)
    plt.close(fig_gif)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"GIF saved as {output_filename}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average frame time: {np.mean(frame_times):.2f} seconds")
    print(f"Max frame time: {np.max(frame_times):.2f} seconds")

    return ani






# Makes GIF of th full zg field

# def make_gif_zg(combined_dataset, gt_combined_dataset, name_fc, var, output_filename, sample_index=0, plev=None):
#     """
#     Create a gif of the forecast for a single sample, evolving over lead times, without using coastlines.
#     """
#     start_time = time.time()
#     print(f"Starting GIF creation for {var}")

#     # Data selection and setup
#     if plev is not None:
#         data_inference = combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
#         data_gt = gt_combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
#     else:
#         data_inference = combined_dataset[var].isel(time=sample_index)
#         data_gt = gt_combined_dataset[var].isel(time=sample_index)

#     print(f"Data shape - Inference: {data_inference.shape}, Ground Truth: {data_gt.shape}")
#     print(f"Lead times: {data_inference.lead_time.values}")

#     vmin = float(min(data_inference.min(), data_gt.min()))
#     vmax = float(max(data_inference.max(), data_gt.max()))
#     print(f"Value range: {vmin} to {vmax}")

#     # Figure setup
#     fig_gif, axs = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})
#     print("Figure created")

#     # Create initial plots and colorbars
#     im1 = axs[0].pcolormesh(data_inference.lon, data_inference.lat, 
#                             data_inference.isel(lead_time=0),
#                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
#     im2 = axs[1].pcolormesh(data_gt.lon, data_gt.lat, 
#                             data_gt.isel(lead_time=0),
#                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
#     # Add colorbars
#     plt.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.05)
#     plt.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.05)

#     axs[0].set_global()
#     axs[1].set_global()
#     axs[0].set_title(f'{name_fc}')
#     axs[1].set_title('Ground Truth')

#     frame_times = []

#     def plot(i):
#         frame_start = time.time()

#         lead_time = data_inference.lead_time.values[i]

#         forecast = data_inference.isel(lead_time=i)
#         truth = data_gt.isel(lead_time=i)

#         # Update plot data
#         im1.set_array(forecast.values.ravel())
#         im2.set_array(truth.values.ravel())

#         var_up = f'{var}_{plev/100:.0f}hPa' if plev is not None else var
#         title = f'{var_up} at lead time {lead_time} hours (Sample {sample_index})'
#         plt.suptitle(title, y=0.95)

#         frame_end = time.time()
#         frame_times.append(frame_end - frame_start)
#         print(f"Frame {i} completed in {frame_end - frame_start:.2f} seconds")

#     print("Starting animation creation")
#     ani = animation.FuncAnimation(fig_gif, plot, frames=len(data_inference.lead_time), repeat=False)
    
#     print("Saving animation")
#     ani.save(output_filename, writer='pillow', fps=1)
#     plt.close(fig_gif)
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"GIF saved as {output_filename}")
#     print(f"Total time: {total_time:.2f} seconds")
#     print(f"Average frame time: {np.mean(frame_times):.2f} seconds")
#     print(f"Max frame time: {np.max(frame_times):.2f} seconds")

#     return ani
