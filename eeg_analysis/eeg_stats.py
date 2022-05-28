import mne
import pathlib
import pickle
# define paths to current folders
DIR = pathlib.Path.cwd()
eeg_DIR = DIR / 'elevation' / "elevation_data"
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# load single subject preprocessed epochs
epochs_v = mne.read_epochs(eeg_DIR / 'vanessa-epo.fif', proj=True, preload=True, verbose=None)
epochs_l = mne.read_epochs(eeg_DIR / 'leonie-epo.fif', proj=True, preload=True, verbose=None)
epochs_s = mne.read_epochs(eeg_DIR / 'sophie-epo.fif', proj=True, preload=True, verbose=None)
epochs_p = mne.read_epochs(eeg_DIR / 'paul-epo.fif', proj=True, preload=True, verbose=None)

# quick look
condition='up'
mne.viz.plot_compare_evokeds([epochs_v[condition].average(),
        epochs_l[condition].average(), epochs_s[condition].average(), epochs_p[condition].average()], picks='FCz')

# get cross subject ERPs
epochs = mne.concatenate_epochs([epochs_l, epochs_s])

# look at single conditions / channels
condition = 'front'
channel = 'FCz'
evoked = epochs.pick(channel)[condition].average()  # pick single channel of single condition
mne.viz.plot_evoked(evoked, ylim=dict(eeg=[-1, 4]))
evoked_data = evoked._data  # obtain elevation_data array of that channel


# visually compare ERP of standard to deviant conditions
deviant_condition = ['up']
evoked_deviant = epochs[deviant_condition].average()
evoked_standard = epochs['front'].average()
# plot both ERPs
mne.viz.plot_compare_evokeds([evoked_deviant, evoked_standard], picks='FCz', ylim=dict(eeg=[-1, 4]))
# plot difference waves
diff = mne.combine_evoked([evoked_deviant, evoked_standard], weights=[1, -1])

# plot distribution of difference wave
diff.plot_joint(times=[0.05, 0.1, 0.15, 0.2])

# topo maps
mne.viz.plot_evoked(diff, picks='FCz')
diff.plot_topo()
diff.plot_topomap(times=[0.05, 0.1, 0.15, 0.2])

# compare mmn rms up vs right
# plot evoked pattern on topographic map
epochs['up'].average().plot_joint()
evoked_up.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2])



