import mne
import pathlib
import pickle
# define paths to current folders
DIR = pathlib.Path.cwd()
eeg_DIR = DIR / 'elevation' / "elevation_data"
import matplotlib
matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
subj_name = 'Paul'

# load raw elevation_data
raw = mne.io.read_raw_brainvision(eeg_DIR / str(subj_name + '_1.vhdr'), preload=True)

# load channel name mapping and rename channels
with open('channel_mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)
raw.rename_channels(mapping)

# add reference channel
raw = mne.add_reference_channels(raw, 'FCz', copy=True)

# load and apply montage
montage_path = DIR / "AS-96_REF_c.bvef"
montage = mne.channels.read_custom_montage(fname=montage_path)
raw.set_montage(montage)

# bandpass filter raw elevation_data
raw.filter(l_freq=1, h_freq=40)

# load and name events
events = mne.events_from_annotations(raw)[0]  # load events
event_id = dict(up=1, down=2, left=3, right=4, front=5)  # assign event id's to the trigger numbers

# set epoch times
tmin = -0.2
tmax = 0.4

# set trial rejection parameters
reject_criteria = dict(eeg=200e-6)   # 200 µV
flat_criteria = dict(eeg=2e-6)   # 2 µV

# get epoched elevation_data with applied baseline and automatic trial rejection
# (should not remove eye movements and blinks yet)
raw.info['bads'] += []
raw.interpolate_bads
raw.info['bads'] += ['FCz']  # exclude FCz channel from flat criteria
epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True)
epochs.info['bads'] = []
epochs.plot_drop_log()

# # hold on
# to avoid too many epochs being dropped due to a single channel,
# consider removing and interpolating these channels in the raw elevation_data
# raw.info['bads'] += ['FC2']
# raw.interpolate_bads()

# re-reference
# reference = ['PO9', 'PO10']  # set average of both mastoid electrodes as reference
reference = 'average'  # alternatively use avg reference
epochs.set_eeg_reference(reference)

# ICA
ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(epochs)
ica.plot_components()  # plot components
# ica_sources = ica.get_sources(epochs)
# ica_sources.plot(picks="all")  # plot time trace of components
# ica.plot_properties(epochs, picks=[0])  # take a closer look
ica.exclude = [0, 1, 3, 4]  # remove selected components
ica.apply(epochs)  # apply ICA

# ---- here we might want to save the pre-processed epochs object
epochs.save(eeg_DIR / str(subj_name + '-epo.fif'), overwrite=True)  # save preprocessed elevation_data

# read saved epochs
epochs_v = mne.read_epochs(eeg_DIR / 'vanessa-epo.fif', proj=True, preload=True, verbose=None)
epochs_l = mne.read_epochs(eeg_DIR / 'leonie-epo.fif', proj=True, preload=True, verbose=None)
epochs_s = mne.read_epochs(eeg_DIR / 'sophie-epo.fif', proj=True, preload=True, verbose=None)
