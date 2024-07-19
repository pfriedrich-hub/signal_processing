import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tdt import DSPProject, DSPError
from pathlib import Path
import slab


signal_duration = 0.5
try:
    # Load the circuit
    project = DSPProject()
    circuit = project.load_circuit(Path.cwd() / 'filters' / 'example' / 'example.rcx',
                                   'RM1')
    circuit.start()

    # Configure the data tags
    circuit.cset_tag('record_del_n', 0, 'ms', 'n')  # recording delay
    circuit.cset_tag('record_dur_n', 0.5, 's', 'n')

    # Compute and upload the waveform
    slab.set_default_samplerate(circuit.fs)
    signal = slab.Sound.chirp(duration=0.5, to_frequency=20000, kind='linear')
    speaker_buffer = circuit.get_buffer(
    data_tag = 'speaker',
    size_tag = 'speaker_n',
    idx_tag = 'speaker_i',
    mode = 'w')

    speaker_buffer.write(signal.data)


    # # Compute and upload the waveform
    # from numpy import arange, sin, pi
    # t = arange(0, 1, circuit.fs**-1)
    # waveform = sin(2*pi*1e3*t)
    # speaker_buffer = circuit.get_buffer('speaker', 'w')
    # speaker_buffer.write(waveform)


    # Acquire the microphone data
    microphone_buffer = circuit.get_buffer(
    data_tag = 'mic',
    size_tag = 'mic_n',
    idx_tag = 'mic_i',
    mode = 'r')

    data = microphone_buffer.acquire(1, 'running', False)  # todo doesnt work yet

except DSPError:
    print("Error acquiring data")


circuit.print_tag_info()
