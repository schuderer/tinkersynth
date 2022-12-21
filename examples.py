from tinkersynth.midi import midi_note_source, NO_VELOCITY, SQUARE_ROOT, create_controllers_from_mapping, cc, \
    pitch_bend_controller
from tinkersynth.synthesizer import *

# MIDI controller mappings
arturia_keylab_essential_61 = {
    # using Analog Lab map on keyboard
    22: "part_1_btn",  # 0 or 127
    23: "part_2_btn",  # 0 or 127
    24: "part_3_btn",  # 0 or 127
    74: "cutoff",  # linear between 0 and 127 from here on
    71: "resonance",
    76: "lfo_rate",
    77: "lfo_amt",
    93: "param_1",
    18: "param_2",
    19: "param_3",
    16: "param_4",
    17: "panning",  # master_pan (turny-twisty thing above master volume slider)
    73: "attack_1",
    75: "decay_1",
    79: "sustain_1",
    72: "release_1",
    80: "attack_2",
    81: "decay_2",
    82: "sustain_2",
    83: "release_2",
    85: "volume",  # master_vol
    # control cluster at center of keyboard:
    116: "cat_btn",
    117: "preset_btn",
    28: "left_btn",
    29: "right_btn",
    114: "rotary_knob",
}

# Main functions
gate_state = HIGH


def gate_gen():
    """This is more of a test helper than anything else.
    Maybe usable for playing notes without interaction.
    midi_note_source() provides live gate generation."""
    global gate_state
    while True:
        new_state = yield gate_state
        if new_state is not None:
            gate_state = new_state


def pad_test():
    #    print(f"main: {threading.active_count(), threading.current_thread()}")
    freqs, gates = midi_note_source(0, velocity_curve=NO_VELOCITY)  # 0 = Arturia KeyLab Essential 61 MIDI In
    create_controllers_from_mapping(arturia_keylab_essential_61, overwrite=True)
    pitch_bend = transpose(freqs, scale(pitch_bend_controller(), -3.0, 3.0))
    pitch1, pitch2, pitch3, pitch4 = split(pitch_bend, 4)
    saw = sawtooth(pitch1)
    squ_frq1 = transpose(pitch2, -11.9)
    squ_frq2 = transpose(pitch3, -12.0)
    squ_frq3 = transpose(pitch4, -12.1)
    squ1 = square(squ_frq1)
    squ2 = square(squ_frq2)
    squ3 = square(squ_frq3)
    mixed = mix([saw, squ1, squ2, squ3], [1.0, 0.3, 0.3, 0.3])
    envelope = eg(mixed, gate=gates, adsr=[1.0, 0.0, 1.0, 2.0])
    lfo = scale(triangle(10), 0.8, 1.0)
    with_vibrato = gain(envelope, lfo)
    flt_cutoff = scale(cc["cutoff"], 0.0, 0.7)
    filt = low_pass_filter(with_vibrato, alpha=flt_cutoff, poles=8)
    vol = gain(filt, cc["volume"])
    player = play_live(vol)


def lead_test():
    freqs, gates = midi_note_source(0, velocity_curve=SQUARE_ROOT)  # 0 = Arturia KeyLab Essential 61 MIDI In
    create_controllers_from_mapping(arturia_keylab_essential_61, overwrite=True)
    squ = square(mult(freqs, 0.5))
    sinus = sine(freqs)
    tri = triangle(freqs)
    m = mix([squ, tri], [0.4, 1.0])  # 2 * sqrt(2) = ca. 1.4 RMS
    envelope = eg(m, gate=gates, adsr=[cc["attack_1"], cc["decay_1"], cc["sustain_1"], cc["release_1"]])
    flt = low_pass_filter(envelope, alpha=cc["cutoff"])
    drv = drive(flt, gain=cc["param_1"], mixin=0.5)
    rev = reverb_hall(drv)
    dely = delay(rev, delay=cc["param_2"], mixin=cc["param_3"], feedback=cc["param_4"])
    vol = cc["volume"]
    scaled = gain(dely, gain=vol)
    player = play_live(scaled)


def porta_test():
    freqs, gates = midi_note_source(0, velocity_curve=SQUARE_ROOT)  # 0 = Arturia KeyLab Essential 61 MIDI In  
    create_controllers_from_mapping(arturia_keylab_essential_61, overwrite=True)
    gate1, gate2 = split(gates, 2)
    porta = portamento(freqs, gate1, cc["panning"])
    osc = scale(square(5), -0.2, 0.2)
    tremolo = transpose(porta, osc)
    tri = sawtooth(tremolo)
    envelope = eg(tri, gate=gate2, adsr=[0.005, 0.1, 0.2, 0.1])
    filt = low_pass_filter(envelope, alpha=cc["cutoff"], resonance=0.0)
    scaled = gain(filt, gain=cc["volume"])
    player = play_live(scaled)


def plot_test():
    # todo: http://navjodh.com/test-and-measurement/continuously-updating-audio-waveform-display-using-python/
    squ = square(440)
    s = sine(440)
    #    filt = high_pass_filter(s, alpha=0.99, poles=32)
    filt = drive(s, gain=5.0, mixin=0.0)
    vals = []
    for i in range(int(300)):
        if i == sample_rate:
            gate_state = LOW
        vals.append(next(filt))
    plt.plot(vals)
    plt.show()


def play_notes():
    flat_adsr = [0.0, 0.0, 1.0, 0.0]
    pluck_adsr = [0.005, 0.0, 1.0, 0.3]
    crash_adsr = [0.005, 0.5, 0.0, 0.0]
    # TODO: make example for playing static notes work again
    samples = [0.0] * 50  # Add some padding
    envelope = []
    note_freqs = [220, 220 * 5 / 4, 220 * 3 / 2, 440, 440 * 3 / 2, 440, 220 * 3 / 2, 220]
    durations = [0.5] * len(note_freqs)
    durations[-1] = 0.4
    adsr_values = flat_adsr
    a = 0.2
    for freq, duration in zip(note_freqs, durations):
        # todo
        anzahl_benoetigte_samples = round(sample_rate * (duration + adsr_values[RELEASE]))
        for sample_schritt in range(anzahl_benoetigte_samples):
            aktuelle_laufzeit = sample_schritt / sample_rate
            adsr_val = adsr.send(aktuelle_laufzeit < duration)
            envelope.append(adsr_val)
            sample = (
                    mixer.send(None) *
                    adsr_val
            )
            samples.append(sample)
    play(samples)
    # plt.plot(samples[:round(anzahl_benoetigte_samples/20)])
    #    plt.plot(samples[:round(sample_rate / f) * 3])
    #    plt.plot(all_envelopes)
    plt.plot(samples[:round(anzahl_benoetigte_samples)])
    plt.show()


def main():
    print("Press Ctrl-C a couple of times to exit (for your own programs "
          "you can use synthesizer.stop_stream() to stop playing)")
    # plot_test()
    porta_test()
    # porta_test()
    # lead_test()


if __name__ == '__main__':
    # Create a thread to run the script
    run_and_reload_on_change(main)
