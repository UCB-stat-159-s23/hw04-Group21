from ligotools import readligo as rl
from ligotools import utils as ul
import pytest
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read in data and extract parameters - code from losc tutorial
eventname = 'GW150914'
fnjson = "data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))

event = events[eventname]
fn_H1 = event['fn_H1']              # File name for H1 data
fn_L1 = event['fn_L1']              # File name for L1 data
fn_template = event['fn_template']  # File name for template waveform
fs = event['fs']                    # Set sampling rate
tevent = 1126259462.44         # Set approximate event GPS time
fband = event['fband']              # frequency band for bandpassing signal

strain_H1, time_H1, chan_dict_H1 = rl.loaddata('data/'+fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata('data/'+fn_L1, 'L1')
time = time_H1
dt = time[1] - time[0]
data = np.random.normal(1,10000,100)

# number of sample for the fast fourier transform:
NFFT = 4*fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# we will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)



f_template = h5py.File('data/'+fn_template, "r")
template_p, template_c = f_template["template"][...]
t_m1 = f_template["/meta"].attrs['m1']
t_m2 = f_template["/meta"].attrs['m2']
t_a1 = f_template["/meta"].attrs['a1']
t_a2 = f_template["/meta"].attrs['a2']
t_approx = f_template["/meta"].attrs['approx']
f_template.close()
# the template extends to roughly 16s, zero-padded to the 32s data length. The merger will be roughly 16s in.
template_offset = 16

template = (template_p + template_c*1.j)
dwindow = signal.tukey(template.size, alpha=1./8)
template_fft = np.fft.rfft(np.random.randn(4096))
# to remove effects at the beginning and end of the data stretch, window the data
# https://en.wikipedia.org/wiki/Window_function#Tukey_window
try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

# prepare the template fft.
template_fft = np.fft.fft(template*dwindow) / fs

datafreq = np.fft.fftfreq(template.size)*fs
df = np.abs(datafreq[1] - datafreq[0])

power_vec = np.interp(np.abs(datafreq), freqs, Pxx_H1)
sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
sigma = np.sqrt(np.abs(sigmasq))

optimal = np.random.rand(131072) * template_fft.conjugate() / power_vec
optimal_time = 2*np.fft.ifft(optimal)*fs

data_psd = np.random.rand(8193)
strain_whitenbp = np.random.rand(131072)
template_match = np.random.rand(131072)


SNR_complex = optimal_time/sigma

# shift the SNR vector by the template length so that the peak is at the END of the template
peaksample = int(data.size / 2)  # location of peak in the template
SNR_complex = np.roll(SNR_complex,peaksample)
SNR = abs(SNR_complex)
indmax = np.argmax(SNR)

timemax = time[indmax]
SNRmax = SNR[indmax]
d_eff = sigma / SNRmax

# tests
def test_whiten():
    ''' 
    Check the type and the length of the output array.
    '''
    white_ht = ul.whiten(strain_H1, psd_H1, dt)
    assert type(white_ht) == np.ndarray
    assert len(white_ht) == len(strain_H1)

def test_write_wavfile():
    '''
    Check the type of the output.
    '''
    assert type(data) == np.ndarray
    assert type(fs) == int
    assert type(ul.write_wavfile(eventname, fs, data)) is not None

def test_reqshift():
    '''
    Check that the output is an array, and the same length as the input array.
    '''
    assert len(ul.reqshift(data, fshift=100, sample_rate=4096)) == len(data)
    assert type(ul.reqshift(data, fshift=100, sample_rate=4096)) == np.ndarray

def test_plot_psd():
    '''
    Check to make sure the plotting function runs and creates accurate x axis label.
    '''
    timemax = 1126259462.432373
    det = 'L1'
    eventname = 'GW150914'
    time = time_H1
    tevent=5

    figure = ul.psd_plot(time, timemax, SNR, data_psd, det, eventname, fs, tevent, strain_whitenbp,template_match,template_fft, datafreq, d_eff, freqs)
    assert "frequency" in str(figure.axes)