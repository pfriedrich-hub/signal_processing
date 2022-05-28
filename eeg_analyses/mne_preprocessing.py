#### EEG elevation_data processing with MNE Python - example script (EEG Practical Neuro 2, SS 2022) ####
# Import needed modules
import mne
import pathlib
import pickle
# define paths to current folders
DIR = pathlib.Path.cwd()
eeg_DIR = DIR / 'elevation' / "elevation_data"

""" prepare elevation_data """
# load eeg recordings:
raw = mne.io.read_raw_brainvision(eeg_DIR / 'vanessa_1.vhdr', preload=True)

# rename channels
raw.info["chs"]
mapping = {"1": "Fp1", "2": "Fp2", "3": "F7", "4": "F3", "5": "Fz", "6": "F4",
           "7": "F8", "8": "FC5", "9": "FC1", "10": "FC2", "11": "FC6",
           "12": "T7", "13": "C3", "14": "Cz", "15": "C4", "16": "T8", "17": "TP9",
           "18": "CP5", "19": "CP1", "20": "CP2", "21": "CP6", "22": "TP10",
           "23": "P7", "24": "P3", "25": "Pz", "26": "P4", "27": "P8", "28": "PO9",
           "29": "O1", "30": "Oz", "31": "O2", "32": "PO10", "33": "AF7", "34": "AF3",
           "35": "AF4", "36": "AF8", "37": "F5", "38": "F1", "39": "F2", "40": "F6",
           "41": "FT9", "42": "FT7", "43": "FC3", "44": "FC4", "45": "FT8", "46": "FT10",
           "47": "C5", "48": "C1", "49": "C2", "50": "C6", "51": "TP7", "52": "CP3",
           "53": "CPz", "54": "CP4", "55": "TP8", "56": "P5", "57": "P1", "58": "P2",
           "59": "P6", "60": "PO7", "61": "PO3", "62": "POz", "63": "PO4", "64": "PO8"}
# with open('channel_mapping.pkl', 'wb') as f:
#     pickle.dump(mapping, f)  # save mapping to pkl file
raw.rename_channels(mapping)


# get sensor positions for topographic analysis
montage_path = DIR / "AS-96_REF.bvef"
montage = mne.channels.read_custom_montage(fname=montage_path)
raw.set_montage(montage)
raw.plot_sensors(kind='topomap')
raw.info["dig"]  # coordinates of points on the surface of the subjects head

""" filter and epoch elevation_data """
# bandpass filter raw elevation_data
raw.filter(l_freq=0.5, h_freq=40)

events = mne.events_from_annotations(raw)[0] # load events
event_id = dict(up=1, down=2, left=3, right=4, front=5)  # assign event id's to the trigger numbers
mne.viz.plot_events(events)
tmin = -0.2
tmax = +0.4
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)  # get epochs
epochs.plot()

""" I. baseline correction """
baseline = (tmin, 0)
epochs.apply_baseline(baseline)

# II. Rejection of bad channels and segments
# In the following we will deal with two sources, blinks
# and non-physiological artifacts.
raw.plot() # look at the continuous EEG again and identify channels that either show consistently
# higher frequencies and higher amplitudes than other channels or no signal at all.

raw.info['bads'] += [] # add names of channels here to mark them as "bad"; e.g. 'EEG 001'
# Alternatively, you can also click on the channel names in the plot to mark them as "bad".

# Alternatively, you can also check for bad channels in the epoched elevation_data and indicate them there:
epochs.average().plot()
epochs.info['bads'] += ['FC2']
# A possibility to "repair" a bad channel is to interpolate its signal based on the information
# from the other channels. This can be done with this command:
epochs.interpolate_bads()

# On the other hand, there might be specific epochs that we should exclude (for example, due to
# extensive movement artifacts). This we can do manually when inspecting the elevation_data:
epochs.plot()  # click on the epochs to mark them as "bad"

# ...or automatically by selecting certain signal amplitude threshold criteria for the different types of elevation_data:
reject_criteria = dict(eeg=10-6)       # 200 µV
flat_criteria = dict(eeg=1e-6)           # 1 µV

# Note that these values are very liberal here.
epochs_auto = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True) # this is the same command for extracting epochs as used above
epochs_auto.plot_drop_log()  # summary of rejected epochs per channel

# Another completely automatized way of dealing with these artifacts is
# the AutoReject pipeline: The elevation_data is split into a train and test set.
# The test set is used to define channel specific rejection thresholds.
# Then test and train set are compared after these thresholds were applied, and
# epochs where channels have a peak-to-peak amplitude greater than that
# threshold are either repaired or rejected. The aim of the pipeline
# is to minimize the difference between test and train elevation_data. For a detailed
# description see: Jas et al. (2017): Autoreject: Automated artifact rejection
# for MEG and EEG elevation_data.
# This module can be installed as described here: http://autoreject.github.io/
# What is nice about this pipeline is that most of the parameters are set
# automatically. We can define the number of channels that are allowed to be
# interpolated (repaired). Setting the random_state makes the process
# deterministic so everyone gets the same result.

from autoreject import AutoReject # import the module
ar=AutoReject(n_interpolate=[3,6,12], random_state=42)
epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)


# Lets have a look at what AutoReject did to the elevation_data:
reject_log.plot_epochs(epochs)  # one should carefully check that not too much of the elevation_data was removed...


# However for now, let´s continue working with the not-fully automatized rejection approach
# where we set the threshold manually:
epochs = epochs_auto


## Rejection of signals of no interest using Independent Component Analysis (ICA)

# Another prominent source of noise in EEG elevation_data are blink artifacts:
# The muscles that cause our eyelids to close and open generate
# electrical dipoles that are picked up by the EEG, especially by the frontal electrodes.
# A good way to detect those artifacts is Independent Component Analysis (ICA).
# This is a procedure that finds statistically independent components in the elevation_data.
# Since The blink artifacts are generated outside of the brain, they can be
# detected by ICA very reliably.

# Blink artifacts are visible in the elevation_data as sharp peaks of high amplitude, most
# prominent in frontal EEG channels.
# --> Task: Can you detect some in the EEG elevation_data? (Use either "raw.plot()" or "epochs.plot()")


# Now, let us do the independent component analysis. The number of components will
# be selected so that the cumulative explained variance is < 0.99
ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(epochs)

# We can project the weights from the unmixing matrix on the scalp (EEG sensors)
# to get something that looks similar to the previously computed voltage
# distributions (topographies). The magnitude at each channel tells you how much the component
# affects that channel - the sign is random.
ica.plot_components()

# The first component (they are ordered by explained variance) looks like a blink artifact.
# To be sure we can also look at the time series elevation_data. This looks like the raw
# elevation_data - only with independent sources instead of channels.
ica_sources = ica.get_sources(epochs)
ica_sources.plot(picks="all")

# The time series of the first component looks like the one from the EOG
# channel. Now that we have identified it as an artifact we can remove the component
# from our elevation_data.
epochs_ica = ica.apply(epochs, exclude=[0]) # insert the index of the bad IC here

# Now the elevation_data should not contain any blink artifacts anymore. You can control this here:
epochs_ica.plot()

# In the following, we will use the ICA-corrected elevation_data:
epochs = epochs_ica

# Let´s also save the elevation_data at this point:
epochs_fname = 'D:\Lehre\EEG_practical_Neuro2\datasample_audvis_epo.fif' # enter file path and name here
epochs.save(epochs_fname, overwrite = True)


# loading the elevation_data again can be done with this command:
epochs = mne.read_epochs(epochs_fname, preload=True)



