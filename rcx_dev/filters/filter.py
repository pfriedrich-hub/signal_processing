from win32com.client import Dispatch
from pathlib import Path
import numpy
import slab
import time

# Load the circuit
zb = Dispatch('ZBUS.x')
zb.ConnectZBUS("GB")
zb.ClearCOF()
zb.LoadCOF(Path.cwd() / 'rcx_dev' / 'filters' / 'filter.rcx')

# Configure the data tags
fs = zb.GetSFreq()
signal_duration = 0.2  # in s
signal_n = int(0.2 * fs)
zb.SetTagVal('signal_n', signal_n)
zb.Run()

# Compute and upload the waveform
slab.set_default_samplerate(fs)
signal = slab.Sound.chirp(duration=signal_duration, to_frequency=20000, kind='linear')
# zb.WriteTagV('signal', 0, signal.data[:, 0])
zb._oleobj_.InvokeTypes(
                    15, 0x0, 1, (3, 0), ((8, 0), (3, 0), (0x2005, 0)),
                    'signal', 0,  signal.data[:, 0])

# set filter parameters
zb.SetTagVal('gain', 1)
zb.SetTagVal('F0', 8000)
zb.SetTagVal('BW', 2000)

#play and filter signal
zb.SoftTrg(1)
while True:
    if zb.GetTagVal('running') == 0:
        last_loop = True
    else:
        last_loop = False
    if last_loop:
        break

filtered = slab.Sound(data=numpy.asarray(zb.ReadTagV('filtered', 0, signal_n)))
filtered.spectrum()




"""
from tdt import DSPProject, DSPError
from pathlib import Path
import slab
from tdt.util import connect_rpcox
# obj = connect_rpcox('zb', 'USB')

project = DSPProject()
circuit = project.load_circuit(Path.cwd() / 'filters' / 'filter.rcx', 'zb')
circuit.start()

signal_duration = 0.2  # in s
circuit.cset_tag('signal_n', signal_duration, 's', 'n')


signal_buffer = circuit.get_buffer(data_tag='signal',
                                size_tag='signal_n',
                                idx_tag='signal_i',
                                mode='w')

filtered_buffer = circuit.get_buffer(data_tag='filtered',
                                    size_tag ='signal_n',
                                    idx_tag='filtered_i',
                                     mode='r')

filtered_buffer = circuit.get_buffer('filtered', 'r')

                                     # cycle_tag='cycle')

slab.set_default_samplerate(circuit.fs)
signal = slab.Sound.chirp(duration=signal_duration, to_frequency=20000, kind='linear')
signal_buffer.write(signal.data)


circuit.trigger(1, 'pulse')

# tdtpy
# data = filtered_buffer.acquire(1, 'running', False)
data = filtered_buffer.acquire_samples(1, 1, trials=1, intertrial_interval=0, poll_interval=0.1, reset_read=True)

filtered_buffer.read(samples=50)


# activeX
size = obj.GetTagV('filtered_i')
data = onk.ReadTagV('speaker', 0, size)

# The DSPBuffer.acquire() method will block until
# the running tag becomes False then return the contents of the buffer

circuit.stop()



circuit.print_tag_info()
# circuit.set_tags(record_duration=5, play_duration=4)"""