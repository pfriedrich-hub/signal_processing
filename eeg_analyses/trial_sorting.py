import mne
import pathlib
import pickle
DIR = pathlib.Path.cwd()
eeg_DIR = '/Users/paulfriedrich/Projects/neurobio2/elevation/data'
import matplotlib
matplotlib.use('TkAgg')
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

path = DIR.cwd()
data_path = path / 'data' / 'elevation_data' / 'Leonie_1.vhdr'

# load raw elevation_data
raw = mne.io.read_raw_brainvision(data_path, preload=True)

# filter data
# multitaper
psd_data, freqs = mne.time_frequency.psd_multitaper(raw, fmin=9.5, fmax=10.5, bandwidth=None, low_bias=True,
                                            normalization='length', picks=None, proj=False, n_jobs=-1)

samplerate = raw.info['sfreq']
data = raw._data
data = psd_data


# load and name events
events = mne.events_from_annotations(raw)[0]  # load events
event_id = dict(up=1, down=2, left=3, right=4, front=5)  # assign event id's to the trigger numbers

# stim times
stim_id = 3
stim_times = events[events[:, 2] == stim_id][:, 0]

# sepoch
tmin = -0.2
tmax = 0.4
n_epochs = len(stim_times)
epoch_n_samples = int((tmax - tmin) * raw.info['sfreq'])

# initialize data arary
epoch_data = numpy.zeros([n_epochs, raw.info['nchan'], epoch_n_samples])

# epoch data
for i in range(n_epochs):
    epoch_data[i] = data[:, stim_times[i]+int(samplerate*tmin):stim_times[i]+int(samplerate*tmax)]


    # baseline
    for fi in range(num_frex):
        epoch_tf[i, fi] = epoch_tf[i, fi] / np.mean(epoch_tf[i, fi, :int(-tmin * samplerate)])
# average across epochs
evoked_tf = np.mean(epoch_tf, axis=0)







"""
# wavelet convolution


# get eeg elevation_data
folder_path = pathlib.Path('Users') / 'paulfriedrich' / 'Projects' / 'AnalyzingNeuralTimeSeries-main' / 'signal_processing' / 'sample_data'
for header_file in folder_path.glob('*.vhdr'):
    eeg = mne.io.read_raw_brainvision(header_file, preload=True)

samplerate = raw.info['sfreq']
eeg_data = raw._data[64]
eeg_time = np.arange(0, len(eeg_data) / samplerate, 1 / samplerate)  # start, stop, step size

# def tf_analysis(raw, min_freq, max_freq, num_frex):
# define frequency range
min_freq = 2
max_freq = 20
num_frex = int(max_freq - min_freq / 2)
# define wavelet parameters
time = np.arange(-1, 1, 1 / samplerate)
frex = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
# number of cycles of morlet wavelet as function of frequency (more cycles with increasing wavelet frequency)
s = np.logspace(np.log10(3), np.log10(10), num_frex) / (2 * np.pi * frex)
# define convolution parameters
n_wavelet = len(time)
n_data = len(eeg_data)
n_convolution = n_wavelet + n_data - 1
n_conv_pow2 = 2 ** (math.ceil(math.log(n_convolution, 2)))
half_of_wavelet_size = int((n_wavelet - 1) / 2)
# get FFT of elevation_data
eeg_fft = np.fft.fft(eeg_data, n_conv_pow2)
# initialize
eeg_power = np.zeros((num_frex, n_data))  # frequencies X time
# loop through frequencies and compute synchronization
for fi in range(num_frex):
    wavelet_fft = \
    np.fft.fft(np.sqrt(1 / (s[fi] * np.sqrt(np.pi)))  #todo empirical scaling factor for varying wavelet cycles
               # complex sine at different frequencies
               * np.exp(2 * 1j * np.pi * frex[fi] * time)
               # gaussian with s cycles
               * np.exp(-time ** 2 / (2 * (s[fi] ** 2))), n_conv_pow2)  # more cycles with increasing wavelet frequency
    # convolution
    eeg_conv = np.fft.ifft(wavelet_fft * eeg_fft)  # convolve
    eeg_conv = eeg_conv[:n_convolution]  # cut result to length of n_convolution
    eeg_conv = eeg_conv[half_of_wavelet_size + 1: n_convolution - half_of_wavelet_size]
    # calculate power
    temp_power = (np.abs(eeg_conv) ** 2)
    # # baseline normalization
    eeg_power[fi] = 10*np.log10(temp_power)


# get stimulus onset times (up: 1, down: 2, left: 3, right: 4, front: 5)
# select stimulus
stim_id = 3


# set epoch times
tmin=-0.2
tmax=0.4


stim_times = events[events[:, 2] == stim_id][:, 0]
for i in range(len(stim_times)):
    # get elevation_data around each event
    epoch_data[i] = eeg_power[:, stim_times[i]+int(samplerate*tmin):stim_times[i]+int(samplerate*tmax)]


# plot
epoch_time = np.arange(tmin, len(evoked_tf[1]) / samplerate+tmin, 1 / samplerate)
x, y = np.meshgrid(epoch_time, frex)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
c_ax = ax.contour(x, y, evoked_tf, linewidths=0.3, colors="k", norm=colors.Normalize())
c_ax = ax.contourf(x, y, evoked_tf, norm=colors.Normalize(), cmap=plt.cm.jet)
cbar = fig.colorbar(c_ax)
plt.show()
plt.vlines(0,ymin=frex.min(), ymax=frex.max(), colors='k', linestyles='dashed')
plt.title(list(event_id.keys())[list(event_id.values()).index(stim_id)])

# load channel name mapping and rename channels
# with open('channel_mapping.pkl', 'rb') as f:
#     mapping = pickle.load(f)
# raw.rename_channels(mapping)

# add reference channel
# raw = mne.add_reference_channels(raw, 'FCz', copy=True)

# load and apply montage
# montage_path = DIR / "AS-96_REF_c.bvef"
# montage = mne.channels.read_custom_montage(fname=montage_path)
# raw.set_montage(montage)

# # reference = ['PO9', 'PO10']  # set average of both mastoid electrodes as reference
# reference = 'average'  # alternatively use avg reference
# raw.set_eeg_reference(reference)
#
# # filter elevation_data
# raw.filter(l_freq=0.5, h_freq=40)



# plot topographic psd
epochs.plot_psd_topomap(bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
         (12, 30, 'Beta'), (30, 45, 'Gamma')], ch_type='eeg', normalize='length')

freqs = np.logspace(*np.log10([6, 35]), num=30)
n_cycles = freqs / 4  # different number of cycle per frequency
power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
             return_itc=True, decim=3, n_jobs=1)

power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
power.plot([82], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[82])
"""