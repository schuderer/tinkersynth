"""
This module contains the generators/nodes for creating and manipulating wave forms and building synth patches.
See the documentation of the individual functions below for more information, and `examples.py` for examples.

The nodes are created and chained so that the audio signal flows through them. Example::

    # Connect to a MIDI device or the PC keyboard
    # `freqs` is a generator/node that gives us the frequencies of the individual played notes
    # `gates` is a generator/node providing the information whether the key is currently pressed down (and how hard)
    freqs, gates = midi_note_source()

    # Create an oscillator with a sawtooth wave form, using the played frequencies
    saw = sawtooth(freqs)

    # In the code until now, nothing has really happened. We just defined stuff.
    # Now we play our pipeline:
    play_live(saw)

As you can hear, this example generates a continuous sound, where the pitch changes when you press
different notes. The notes actually starting and stopping is handled by an `envelope_generator`.
In its default configuration, the envelope generator turns on and off the notes as we expect::

   # Replacing the last play command with these lines:
   env = envelope_generator(saw)  # you can also use the shorthand `eg`
   play_live(env)  # <-- note that we changed the input from `saw` to `env`

But an envelope generator can do much more. It models Attack, Decay, Sustain and Release (ADSR)
of a played note. For details please search the Web on adsr envelopes. By varying these values,
the sound can get very different properties -- anything between dreamy and percussive is possible.

Here's a slightly more sophisticated example::

    # Connect to a MIDI device or the PC keyboard
    # `freqs` is a generator/node that gives us the frequencies of the individual played notes
    # `gates` is a generator/node providing the information whether the key is currently pressed down (and how hard)
    freqs, gates = midi_note_source()

    # Create an oscillator with a sawtooth wave form, using the played frequencies
    saw = sawtooth(freqs)

    # Make another node which has the same frequency information, but one octave lower, and use it in
    # a square wave oscillator
    lower_freq = transpose(freqs, -12)  # could also have scaled by 0.5 or 2.0 for octaves: scale(freqs, 0.5)
    squ = square(lower_freq)

    # Now mix the two sounds together
    m = mix([saw, squ])

    # Using the `gates` info from the beginning, we can send our signal through an envelope generator
    env = eg(m, gate=gates, adsr=[0.005, 0.05, 0.35, 0.3])  # Adds a slightly percussive quality

    # Add some drive and reverb
    drv = drive(env, gain=0.8, mixin=0.5)
    rev = reverb_hall(drv)

    # Before playing, scale down a bit to be safe from clipping
    scaled = gain(rev, gain=0.8)

    # In the code until now, nothing has really happened. We just defined stuff.
    # Now we play our pipeline:
    play_live(scaled)
"""

import os
import sys
import threading
from array import array
from itertools import tee
from math import sin, pi, copysign, sqrt
from queue import SimpleQueue as Queue
from random import random
from time import sleep
from typing import Union, Optional, Iterable, Collection

import numpy as np
import miniaudio as ma

ATTACK = 0
DECAY = 1
SUSTAIN = 2
RELEASE = 3
TWELVE_EQUAL = 0
WELL_TEMPERED = 1  # support is todo
HIGH = 1
LOW = 0

temperament = TWELVE_EQUAL
channels = 1
sample_rate = 44100
sample_width = 2  # 16 bit pcm
buffersize_msec = 10
sample_max = (2 ** (8 * sample_width) / 2) - 1
stop = False


# Run management (optional):
def run_and_reload_on_change(func: callable):
    """Execute a function and monitor the executed Python script for file changes.
    This is completely optional and just to make exploration easier and faster.

    If the script file changes, the complete script will be re-run with the original parameters.

    :param func: function to execute
    :type func: callable
    """
    thread = threading.Thread(target=func)
    thread.start()
    mod_time = os.path.getmtime(sys.argv[0])
    while True:
        # Check if the script file has been modified
        if mod_time != os.path.getmtime(sys.argv[0]):
            # If the file has been modified, stop the thread and restart the script
            while thread.is_alive():
                print("Signalling audio stream to stop...")
                stop_stream()
                sleep(0.3)
            print("Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            # If the file has not been modified, sleep for a while
            sleep(0.5)


# Audio output
def play_static(samples: Iterable):
    """Play a finite iterable of audio samples.
    TODO: Adjust to be compatible with generators.
    :param samples: list/tuple/... of samples
    :type samples: Iterable
    """
    samples_np = (np.array(samples) * sample_max).astype(np.int16)
    #    print(samples_np.shape)
    #    print(samples_np[:10])
    #    print(samples_np.dtype)
    arr = array('h', samples_np)
    stream = ma.stream_raw_pcm_memory(arr, channels, sample_width)
    stream_with_callbacks = ma.stream_with_callbacks(
        stream,
        progress_callback=None,
        frame_process_method=None,
        end_callback=stop_stream,
    )
    with ma.PlaybackDevice(
            output_format=ma.SampleFormat.SIGNED16,
            nchannels=channels,
            sample_rate=sample_rate,
            buffersize_msec=buffersize_msec,
    ) as device:
        next(stream_with_callbacks)  # start the generator
        device.start(stream_with_callbacks)
        while not stop:
            sleep(0.1)


def play_live(generator: Iterable):
    """Consume the specified `generator` node as audio samples to play using system audio.
    See the constants in `synthesizer.py` for sampling rate and similar settings.

    The generator can be the last in a potentially long and complex chain (see `examples.py`).
    This might be a `gain()` node using cc["master_volume"] to set the output volume.

    :param generator: Sample generator node to play
    :type generator: generator
    """
    sample_format = {1: ma.SampleFormat.UNSIGNED8, 2: ma.SampleFormat.SIGNED16, 4: ma.SampleFormat.SIGNED32}[
        sample_width]
    stream = _my_audio_stream(generator)
    with ma.PlaybackDevice(
            output_format=sample_format,
            nchannels=channels,
            sample_rate=sample_rate,
            buffersize_msec=buffersize_msec,
    ) as device:
        next(stream)  # start the generator
        device.start(stream)
        while not stop:
            sleep(0.5)


def _my_audio_stream(data: Iterable):
    type_code = {1: "b", 2: "h", 4: "l"}[sample_width]
    did_warn = False

    def prep(x):
        nonlocal did_warn
        x_clipped = max(min(x, 1.0), -1.0)
        if x != x_clipped and not did_warn:
            print(f"Warning: Audio stream is clipping (value {x} encountered). Please scale or otherwise constrain it.")
            did_warn = True
        return int(x_clipped * sample_max)

    requested_frames = yield b""  # how many frames do we need to provide
    while not stop:
        frames = array(type_code, [prep(next(data)) for _ in range(requested_frames)])
        requested_frames = yield frames


def stop_stream():
    """Signal the audio player to stop playing"""
    global stop
    stop = True
    print("Stopping")


# Helper functions:
def gen_iter(*args):
    """Helper function to make sure the `*args` are iterable, wrapping non-iterables in a generator which
    returns the same value over and over.
    """
    arg_gens = [ensure_iterable(arg) for arg in args]
    for curr_vals in zip(*arg_gens):
        if len(curr_vals) == 1:
            yield curr_vals[0]
        else:
            yield curr_vals


def static_generator(signal):
    """Wrap a value in a generator which returns the same value over and over."""
    while True:
        yield signal


def ensure_iterable(signal: Union[Iterable, float], force=False):
    """If `signal` is not iterable, wrap it in a generator returning the same value over and over."""
    if isinstance(signal, Iterable) and not force:
        return signal
    else:
        return static_generator(signal)


# DSP nodes:
def envelope_generator(
        signal: Union[Iterable, float] = 1.0,
        gate: Union[Iterable, float] = HIGH,
        adsr: Union[Iterable[Iterable[float]], Iterable[float]] = (0.0, 0.0, 1.0, 0.0),
        auto_trigger: Union[Iterable, int] = LOW,
):
    """Create an Envelope Generator (Attack/Decay/Sustain/Release)

    Creates an amplitude scaling envelope between 0.0 and 1.0. If `signal` is provided, the result will
    be this signal scaled according to the envelope progression. If `signal` is not provided, the output
    will be the [0.0, 1.0] scaling envelope itself, which can be used as parameter for other nodes (e.g.
    `low_pass_filter` cutoff).

    Scaling and movement from one envelope step to the next is linear.

    :param signal: Optional signal to apply envelope to (default 1.0 constant)
    :type signal: Iterable or float
    :param gate: The gate signal (> 0.0 if key is held down, 0.0 (=LOW) otherwise)
    :type gate: float
    :param adsr: Tuple/list of the four values for Attack, Decay, Sustain and Release, or a generator providing a stream of tuples/lists.
    :type adsr: Union[Iterable[Iterable[float]], Iterable[float]]
    :param auto_trigger: Repeat the Attack-Decay sequence as long as `gate` is > 0.0
    :type auto_trigger: int
    :return: Yields envelope-scaled signal, or envelope signal itself if signal is not provided
    :rtype: float
    """
    adsr_phase = ATTACK
    prev_adsr = (-1, -1, -1, -1)
    prev_gate = LOW
    curr_velocity = 0
    curr_env = 0.0
    attack_step = None
    decay_step = None
    release_step = None
    max_scaled = None
    attack_step_scaled = None
    decay_step_scaled = None
    sustain_val_scaled = None
    release_step_scaled = None
    if not isinstance(adsr, list) and not isinstance(adsr, tuple):
        raise ValueError(
            f"Parameter `adsr` must be a flat list/tuple of 4 envelope values (attack, decay, sustain, release) or of 4 Iterables/Generators which provides a, d, s and r respectively. It was {adsr}, type {type(adsr)}")
    #    if not isinstance(adsr[0], Iterable):
    #        # Make flat adsr list a generator (the usual Iterable check does not suffice)
    #        adsr = static_generator(adsr)
    for sig_val, gate_val, a, d, s, r, retrig in gen_iter(signal, gate, *adsr, auto_trigger):
        adsr_val = (a, d, s, r)
        if adsr_val != prev_adsr:
            # For immediate envelope changes, we still use a change time of 10ms
            # (step size < 1.0) to avoid popping noise:
            minimum_step = 1.0 / (0.01 * sample_rate)
            attack_step = 1.0 / (adsr_val[ATTACK] * sample_rate) if adsr_val[ATTACK] > 0.0 else minimum_step
            decay_step = (1.0 - adsr_val[SUSTAIN]) / (adsr_val[DECAY] * sample_rate) if adsr_val[
                                                                                            DECAY] > 0.0 else minimum_step
            if adsr_val[DECAY] > 0.0 and adsr_val[SUSTAIN] == 0.0:
                # AD mode:
                # If A and D are set, but S is 0.0, R should be ignored 
                # (release as fast as possible = ignore release time)
                release_step = minimum_step
            elif adsr_val[DECAY] == 0.0 and adsr_val[SUSTAIN] == 0.0:
                # AR mode:
                # If A and R are set, but D and S are zero, we should immediately move from
                # the ATTACK phase to the RELEASE phase. The end value of ATTACK (= 1.0)
                # will be the RELEASE starting value:
                release_step = 1.0 / (adsr_val[RELEASE] * sample_rate) if adsr_val[RELEASE] > 0.0 else minimum_step
            else:
                # Normal ADSR mode
                release_step = adsr_val[SUSTAIN] / (adsr_val[RELEASE] * sample_rate) if adsr_val[
                                                                                            RELEASE] > 0.0 else minimum_step

        # Gate carries velocity information (don't test for HIGH, but for gate_val != LOW)
        if prev_gate == LOW and gate_val != LOW:
            # gate_val > 0 carries the velocity signal
            attack_step_scaled = attack_step * gate_val
            decay_step_scaled = decay_step * gate_val
            release_step_scaled = release_step * gate_val
            sustain_val_scaled = adsr_val[SUSTAIN] * gate_val
            max_scaled = gate_val
        if prev_gate != LOW and gate_val == LOW:
            #            print("starting release")
            adsr_phase = RELEASE
        if adsr_phase == ATTACK and gate_val != LOW:
            if curr_env < max_scaled:
                curr_env = min(curr_env + attack_step_scaled, max_scaled)
            elif adsr_val[DECAY] == 0.0 and adsr_val[SUSTAIN] == 0.0:
                # AR mode
                adsr_phase = RELEASE
            else:
                #                print("starting decay")
                adsr_phase = DECAY
        elif adsr_phase == DECAY and gate_val != LOW:
            if curr_env > sustain_val_scaled:
                curr_env = max(curr_env - decay_step_scaled, sustain_val_scaled)
            elif retrig:
                #                print("retriggering attack")
                adsr_phase = ATTACK
            else:
                #                print("starting sustain")
                adsr_phase = SUSTAIN
        if adsr_phase == RELEASE:
            if curr_env > 0.0 and gate_val == LOW:
                curr_env = max(curr_env - release_step_scaled, 0.0)
            else:
                #                print("end of release")
                #                curr_env = 0.0  # leads to popping sound
                adsr_phase = ATTACK
        #        print(prev_gate, gate_val, adsr_phase, curr_env)
        prev_gate = gate_val
        prev_adsr = adsr_val
        yield sig_val * curr_env


# Shorthand
eg = envelope_generator


def sine(f: Union[Iterable, float] = 440):
    """Create an oscillator node generating a sine wave for the given frequency `f`::

       --
     /    \\
           \\    /
             --

    Sine waves have no harmonic overtones. They are the vanilla flavour of oscillators.

    :param f: Frequency in Hz
    :type f: Iterable (generator) or constant float
    :return: Yields samples between -1.0 and 1.0
    :rtype: float.
    """
    #    print(f"sine-init: {threading.active_count(), threading.current_thread()}")
    wave = 2 * pi
    step = 0
    f_prev = -1
    prev_duration = None
    duration_in_samples = 0
    for f_val in gen_iter(f):
        #        print(f"sine_loop: {threading.active_count(), threading.current_thread()}")
        if f_val != f_prev and f_val > 0:
            duration_in_samples = round(sample_rate / f_val)
            if prev_duration is not None:
                step = round(step / prev_duration * duration_in_samples)
        yield sin(f_val * wave * step / sample_rate)
        step += 1
        if step >= duration_in_samples:
            step = 0
        f_prev = f_val
        prev_duration = duration_in_samples


def square(f: Union[Iterable, float]):
    """Create an oscillator node generating a square wave for the given frequency `f`::

      ------        -------
     |      |      |       |
            |      |
             ------

    Square waves possess a theoretically infinite cascade of harmonics. They literally have some edge to them.

    :param f: Frequency in Hz
    :type f: Iterable (generator) or constant float
    :return: Yields samples between -1.0 and 1.0
    :rtype: float.
    """
    step = 0
    amplitude = 1.0
    for f_val in gen_iter(f):
        half_duration = round(sample_rate / f_val / 2)
        if step > half_duration:
            amplitude *= -1.0
            step = 0
        else:
            step += 1
        yield amplitude


def sawtooth(f: Union[Iterable, float]):
    """Create an oscillator node generating a sawtooth wave for the given frequency `f`::

      .     .
     | \\   | \\
     |  \\  |  \\
         \\ |   \\ |
          \\|    \\|


    Sawtooth waves have a very raspy sound. Many harmonics to go around with and play with filters.
    If you like 8-bit game music, this oscillator will be important.

    :param f: Frequency in Hz
    :type f: Iterable (generator) or constant float
    :return: Yields samples between -1.0 and 1.0
    :rtype: float.
    """
    amplitude = 1.0
    amp_step = 0.0
    for f_val in gen_iter(f):
        duration = round(sample_rate / f_val)
        amp_step = copysign(2.0 / duration, amp_step)
        if amplitude == -1.0:
            amplitude = 1.0
        else:
            amplitude = max(amplitude - amp_step, -1.0)
        yield amplitude


def triangle(f: Union[Iterable, float]):
    """Create an oscillator node generating a triangle wave for the given frequency `f`::

       /\\      /\\
      /  \\    /  \\
          \\  /    \\  /
           \\/      \\/


    Triangle waves are pretty close sine waves, but possess a natural quality which is useful when
    attempting to mimic woodwind instruments or xylophones.

    :param f: Frequency in Hz
    :type f: Iterable (generator) or constant float
    :return: Yields samples between -1.0 and 1.0
    :rtype: float.
    """
    amplitude = 0.0
    amp_step = 0.0
    for f_val in gen_iter(f):
        duration = round(sample_rate / f_val)
        amp_step = copysign(2.0 / duration, amp_step)
        if amplitude >= 1.0 or amplitude == -1.0:
            amp_step = -amp_step
        if amp_step > 0:
            amplitude = min(amplitude + amp_step, 1.0)
        else:
            amplitude = max(amplitude + amp_step, -1.0)
        yield amplitude


def noise(f: Optional[Union[Iterable, float]]):
    """Create an oscillator node generating white noise, ignoring the frequency parameter `f`::

         .       .
      .     .
         .     .  .
       .    .       .
      .      .  .


    Noise is often mixed to other wave forms to add some texture. When adding
    (possibly modulated) filters, the character can change dramatically.
    Frequently combined with a suitable envelope to model percussive instruments.

    :param f: (just present for compatibility, ignored)
    :type f: (ignored)
    :return: Yields samples between -1.0 and 1.0
    :rtype: float.
    """
    for f_val in gen_iter(f):
        yield 2.0 * random() - 1.0


def low_pass_filter(  # naïve implementation
        signal: Union[Iterable, float],
        alpha: Union[Iterable, float] = 1.0,
        resonance: Union[Iterable, float] = 0.0,  # TODO: doesn't work yet
        poles: int = 4):
    """Creates a generator to attenuate (reduce) higher frequencies in the input signal.
    This removes higher harmonics, giving sounds a (literally) smoother quality.
    Can also be used for smoothing non-sound data (e.g. raw sensor values).

    `alpha` indirectly selects the cutoff frequency (alpha = Xc/sqrt(R^2 + Xc^2), where
    Xc = 1 / (2 pi f C)).
    Higher pole numbers (4 and 8 are good numbers) increase the sharpness of the cutoff.

    Currently, the undesired attenuation of lower frequencies is not compensated for (this
    is a passive IIR filter).
    Neither is there any key tracking (i.e. shift of the cutoff according to the frequency played).

    :param signal: Signal to be filtered
    :type signal: Iterable or float
    :param alpha: Select cutoff frequency. 1.0 = all-pass (default), 0.0 = no-pass
    :type alpha: Iterable or float
    :param resonance: Not implemented
    :type resonance: Iterable or float
    :param poles: Number of times the filter is applied to the signal (poles/stages/degrees).
      Every pole *should* lead to a -45 degree phase shift around the cutoff frequency and -90 above,
      but numeric results don't (yet) agree with that.
    :type poles: int
    :return: Yields the low-pass signal
    :rtype: float
    """

    last = [0.0] * poles
    filtered = [0.0] * poles
    for sig, alph, res in gen_iter(signal, alpha, resonance):
        curr_sig = sig
        for p in range(poles):
            #            filtered[p] = (1.0 - alph) * filtered[p] + alph * (curr_sig + last[p])/2.0
            filtered[p] = (1.0 - alph) * filtered[p] + alph * curr_sig
            last[p] = curr_sig
            curr_sig = filtered[p]
        curr_sig -= res * curr_sig  # inverted feedback (should work due to 4-pole filter's phase shift of -180, but doesn't)
        yield curr_sig


# Example LPF with resonance: https://www.kvraudio.com/forum/viewtopic.php?p=2090514#p2090514
# // TODO: Moog-style lowpass filter:
# //Init
# cutoff = cutoff freq in Hz
# fs = sampling frequency //(e.g. 44100Hz)
# res = resonance [0 - 1] //(minimum - maximum)
#
# f = 2 * cutoff / fs; //[0 - 1]
# k = 3.6*f - 1.6*f*f -1; //(Empirical tunning)
# p = (k+1)*0.5;
# scale = e^((1-p)*1.386249;
# r = res*scale;
# y4 = output;
#
# y1=y2=y3=y4=oldx=oldy1=oldy2=oldy3=0;
#
# //Loop
# //--Inverted feed back for corner peaking
# x = input - r*y4;
#
# //Four cascaded onepole filters (bilinear transform)
# y1=x*p + oldx*p - k*y1;
# y2=y1*p+oldy1*p - k*y2;
# y3=y2*p+oldy2*p - k*y3;
# y4=y3*p+oldy3*p - k*y4;
#
# //Clipper band limited sigmoid
# y4 = y4 - (y4^3)/6;
#
# oldx = x;
# oldy1 = y1;
# oldy2 = y2;
# oldy3 = y3;
#


def high_pass_filter(signal: Union[Iterable, float], alpha: Union[Iterable, float] = 1.0, poles: int = 2):
    """Creates a generator to attenuate (reduce) the lower frequencies in the input signal.
    This makes higher harmonics stand out more in relation.
    Can also be used for isolating momentary changes from other data (e.g. raw sensor values).

    `alpha` indirectly selects the cutoff frequency.
    Higher pole numbers (4 and 8 are good numbers) increase the sharpness of the cutoff (Not sure
    this works properly, though. It sounds weird).

    Currently, the undesired attenuation of higher frequencies is not compensated for (this
    is a passive IIR filter).
    Neither is there any key tracking (i.e. shift of the cutoff according to the frequency played).

    :param signal: Signal to be filtered
    :type signal: Iterable or float
    :param alpha: Select cutoff frequency. 1.0 = all-pass (default), 0.0 = no-pass
    :type alpha: Iterable or float
    :param poles: Number of times the filter is applied to the signal (poles/stages/degrees).
    :type poles: int
    :return: Yields the high-pass signal
    :rtype: float
    """
    last = [0.0] * poles
    filtered = [0.0] * poles
    for sig, alph in gen_iter(signal, alpha):
        curr_sig = sig
        for p in range(poles):
            filtered[p] = alph * (filtered[p] + curr_sig - last[p])
            #            filtered[p] = (1.0 - alph) * filtered[p] + alph * curr_sig
            last[p] = curr_sig
            curr_sig = filtered[p]
        yield curr_sig


def mix(signals: Union[Iterable[Collection[float]], Collection[float]],
        weights: Optional[Union[Iterable[Collection[float]], Collection[float]]] = None):
    """Creates a generator that combines several signals into one by averaging (optionally weighted averaging).

    Typically used to combine different wave forms into one. But other creative uses are
    possible, such as fusing different controller inputs (e.g. using one as coarse and the
    other as fine input for the same parameter).

    Before using `weights`, please consider the alternative of applying `gain` to the
    various input signals before mixing. This often makes the patch code easier to manage.

    Because a simple averaging would reduce the overall volume, the gain is scaled to retain
    the average (RMS) power of the signal (if no weights are provided).
    This is only correct on average, though, and might lead to occasional audio clipping.
    To make sure that the output is never clipped, consider
    outputting less than 100% gain, e.g. scale to 0.8 as the last step of your patch.

    :param signals: Tuple/list of signals to mix
    :type signals: Collection of Iterables or floats
    :param weights: Optional relative weights, overriding RMS scaling. Should sum up to 1.0
    :type weights: Optional Collection of Iterables or floats
    :return: Yields the mixed signal
    :rtype: float
    """
    n = len(signals)
    if weights is None:
        weights = [1 / sqrt(n)] * n  # e.g. 1/sqrt(2)=0.7 RMS -- 3dB per additional signal
    for vals_and_weights in gen_iter(*signals, *weights):
        vals = vals_and_weights[:n]
        weights = vals_and_weights[n:]
        yield sum([v * w for v, w in zip(vals, weights)])


def gain(signal: Union[Iterable, float], gain: Union[Iterable, float] = 1.0):
    """Creates a generator that reduces or boosts a `signal` by the factor `gain`.

    Scaling is linear (multiplication by `gain`) and unbounded.

    Typically used with `gain` values between 0.0 and 1.0, but you can get creative, of course.
    Just make sure that the audio output signal does not clip (is between -1.0 and 1.0).

    :param signal: Signal to scale.
    :type signal: Iterable or float
    :param gain: Factor to scale `signal` by, e.g. 1.0 = no change (default); 0.5 = half amplitude;
      0.0 = mute completely; -1.0 = invert
    :type gain: Iterable or float
    :return: Yields the scaled signal
    :rtype: float
    """
    for sig, g in gen_iter(signal, gain):
        yield g * sig


def scale(
        signal: Union[Iterable, float],
        target_min: Union[Iterable, float] = -1.0, target_max: Union[Iterable, float] = 1.0,
        source_min: Union[Iterable, float] = -1.0, source_max: Union[Iterable, float] = 1.0
):
    """Creates a generator that linearly resizes a `signal` between two values.

    Please note that this LERP transformation is linear and unbounded. To limit it to not exceed `target_min` and
    `target_max`, please combine it with the `constrain` generator function.

    The parameters `target_min` and `target_max` come first so that scaling normal [-1.0, 1.0] signals
    does not require much typing::

     osc = square(440)
     scaled = scale(osc, 0.0, 1.0)  # create a DC square wave signal

    :param signal: Signal to be scaled
    :type signal: Iterable or float
    :param target_min: New lower bound (default -1.0)
    :type target_min: Iterable or float
    :param target_max: New upper bound (default 1.0)
    :type target_max: Iterable or float
    :param source_min: Original lower bound (default -1.0)
    :type source_min: Iterable or float
    :param source_max: Original upper bound (default 1.0)
    :type source_max: Iterable or float
    :return: Yield the scaled signal
    :rtype: float
    """
    for sig, t_min, t_max, s_min, s_max in gen_iter(signal, target_min, target_max, source_min, source_max):
        scale = (t_max - t_min) / (s_max - s_min)
        yield t_min + (sig - s_min) * scale


def transpose_factor_12eq(s: float):
    """Get the transposition factor to multiply with any given frequency (equal temperament).

    :param s: Semitones to transpose up (positive s) or down (negative s). Need not be integer.
    :type s: float
    :return: Transposition factor
    :rtype: float
    """
    return 2 ** (s / 12)


def transpose(freq: Union[Iterable, float], semitones: Union[Iterable, float] = 0.0):
    """Create a generator that transposes a frequency `freq` by a given number of `semitones`.

    :param freq: Frequency signal (one frequency value, like 440.0, or an Iterable/node providing frequency values)
    :type freq: Iterable or float
    :param semitones: Semitones (>0.0 for transposing up, <0.0 for transposing down), non-integers allowed.
    :type semitones: Iterable or float
    :return: Yields the transposed frequency value
    :rtype: float
    """
    if temperament != TWELVE_EQUAL:
        raise NotImplementedError("Only 12 Equal Temperament is supported")
    prev_s = None
    prev_f = None
    factor = None
    transposed_f = None
    for f, s in gen_iter(freq, semitones):
        if s != prev_s:
            factor = transpose_factor_12eq(s)
            transposed_f = f * factor
        if f != prev_f:
            transposed_f = f * factor
        yield transposed_f
        prev_f = f
        prev_s = s


def split(signal: Union[Iterable, float], n: int):
    """Creates `n` generators from one single `signal`.

    Necessary if the same signal has to be used more than once, because e.g. using an oscillator
    in two places would consume it at double speed, increasing its pitch by an octave.

    Particularly useful for using control signals in several places
    (e.g. multiple oscillators with `freq`, or using gate signals in `eg` and in `portamento`).
    But also commonly used in effects like `reverb`.

    Be advised that if the n generators are not consumed evenly,
    the size of the required buffer can grow extremely large.

    Example 1::

      freq1, freq2, freq3 = split(orig_freq, 3)

    Example 2::

      detuned_saws = [sawtooth(transpose(f, random()-0.5)) for f in split(orig_freq, 10)]
      output = mix(detuned_saws)

    :param signal: Signal to split up into n signals
    :type signal: Iterable or float
    :param n: Number of signals to create
    :type n: int
    :return: Returns a tuple with `n` separately usable copies of the original signal
    :rtype: tuple of Iterables or floats
    """
    return tuple(tee(ensure_iterable(signal), n))


def constrain(signal: Union[Iterable, float], minimum: float = -1.0, maximum: float = 1.0):
    """Creates a generator to limit a signal to hard minimum and maximum boundaries.
    This clips the signal at those boundaries. For less audible clipping, see `drive`.

    :param signal: Signal to constrain
    :type signal: Iterable or float
    :param minimum: Lower bound
    :type minimum: float
    :param maximum: Upper bound
    :type maximum: float
    :return: Yields clipped signal
    :rtype: float
    """
    for s in signal:
        yield max(min(s, maximum), minimum)


def if_else(
        bool_exp: Union[Iterable, float],  # >=0.5 = true, else false
        if_true: Iterable,  # Generator (/chain) to use as signal if true
        if_false: Iterable,  # Generator (/chain) to use as signal if false
):
    """Creates a generator which selects one or the other upstream signals to pass through.

    Can for example be used to turn on/off effects (and save CPU cycles)::

      plain = square(440)
      with_reverb = reverb_hall(plain)
      out = if_else(cc["some_button"], with_reverb, plain)

    Keep in mind that you can create your own Python generators for the `bool_exp` part, e.g.::

      def on_off():
        '''Needs to be initialized with paranthesis: if_else(on_off(), ...)'''
        global some_variable
        while True:
          if some_variable is True:
            yield 1.0
          else:
            yield 0.0

      # or with generator expressions (assuming global scope for this example)
      on_off = (float(v) for v in repeat(some_variable))

    Note that the non-used signal node (`if_true` or `if_false`) is not running while not selected. This
    might sometimes have unintended consequences. For example, if one of these or both have a `split`
    somewhere upstream, you might encounter large memory consumption/buffer overflow in the inactive
    branch.

    To be completely safe (at the cost of wasting CPU cycles), you can use a combination of `gain`
    and `mix` to achieve the same. The basic arithmetic generator functions `add`, `sub`, `mult` and `div`,
    as well as the all-purpose `apply`, can be useful building blocks as well::

      plain = square(440)
      with_reverb = reverb_hall(plain)
      out = mix(
        gain(with_reverb, cc["some_button"])
        gain(plain, sub(1.0, cc["some_button"]))
      )

    :param bool_exp: Expression to use for the choice. >=0.5 means True and <0.5 means False.
    :type bool_exp: float
    :param if_true: Stream to pass through if `bool_exp` is greater than or equal to 0.5
    :type if_true: Iterable
    :param if_false: Stream to pass through if `bool_exp` is less than 0.5
    :type if_false: Iterable
    :return: Yields selected stream's data
    :rtype: Probably float, but hard to say, really
    """
    for b in gen_iter(bool_exp):
        if b >= 0.5:
            yield next(if_true)
        else:
            yield next(if_false)


def apply(signal: Union[Iterable, float], func: callable):
    """Create a generator that applies a function to the signal value and yields its output

    This helper exists to give you greater flexibility when creating your patches,
    so there's no need to define custom generators for every little unsupported transformation.

    :param signal: Signal whose values to modify
    :type signal: Iterable or float
    :param func: Function to apply to the signal
    :type func: callable
    :return: Yield value-by-value output of `func` applied to `signal`
    :rtype: float
    """
    for s in signal:
        yield func(s)


def mult(signal: Union[Iterable, float], other_val: Union[Iterable, float]):
    """Create a generator that multiplies one signal with another.

    This helper exists to give you greater flexibility when creating your patches,
    so there's no need to define custom generators for every little unsupported operation.

    :param signal: Signal whose values to modify
    :type signal: Iterable or float
    :param other_val: Other value of the arithmetic operation
    :type other_val: Iterable or float
    :return: Yield value-by-value output of `signal` multiplied by `other_val`
    :rtype: float
    """
    for s, v in gen_iter(signal, other_val):
        yield s * v


def div(signal: Union[Iterable, float], other_val: Union[Iterable, float]):
    """Create a generator that divides one signal by another.

    This helper exists to give you greater flexibility when creating your patches,
    so there's no need to define custom generators for every little unsupported operation.

    :param signal: Signal whose values to modify
    :type signal: Iterable or float
    :param other_val: Other value of the arithmetic operation
    :type other_val: Iterable or float
    :return: Yield value-by-value output of `signal` divided by `other_val`
    :rtype: float
    """
    for s, v in gen_iter(signal, other_val):
        yield s / v


def add(signal: Union[Iterable, float], other_val: Union[Iterable, float]):
    """Create a generator that adds one signal to another.

    This helper exists to give you greater flexibility when creating your patches,
    so there's no need to define custom generators for every little unsupported operation.

    :param signal: Signal whose values to modify
    :type signal: Iterable or float
    :param other_val: Other value of the arithmetic operation
    :type other_val: Iterable or float
    :return: Yield value-by-value output of `signal` added to `other_val`
    :rtype: float
    """
    for s, v in gen_iter(signal, other_val):
        yield s + v


def sub(signal: Union[Iterable, float], other_val: Union[Iterable, float]):
    """Create a generator that subtracts one signal from another.

    This helper exists to give you greater flexibility when creating your patches,
    so there's no need to define custom generators for every little unsupported operation.

    :param signal: Signal whose values to modify
    :type signal: Iterable or float
    :param other_val: Other value of the arithmetic operation
    :type other_val: Iterable or float
    :return: Yield value-by-value output of `signal` minus `other_val`
    :rtype: float
    """
    for s, v in gen_iter(signal, other_val):
        yield s - v


def drive(
        signal: Union[Iterable, float],
        gain: Union[Iterable, float] = 2.0,
        mixin: Union[Iterable, float] = 0.0):
    """Creates a generator boosting a signal to make it sound 'fatter'. Depending on the parameters,
    This can range from a clean boost, via overdrive (if you mix in undriven signal),
    to distortion (without any simulated physical tube amp effects, though).

    :param signal: Source signal iterable/generator
    :param gain: amplitude scaling factor (default 2.0). Higher gains lead to more drastic cutoffs. A gain value of 0.0 turns this node into a linear pass-through.
    :param mixin: mix this much (0.0 to 1.0) of the unmodified signal back in (default 0.0)
    """
    for s, g, m in gen_iter(signal, gain, mixin):
        # Apply gain, then apply smoothing function x/(1+|x|), then scale result back to [-1, 1]
        # s_ = s * g
        # smoothed = s_ / (1 + abs(s_))
        # yield smoothed * (g + 1)/g
        # save one multiplication and division, do it in one step:
        driven = s / (1 + abs(s * g)) * (g + 1)
        #        print(g, s, driven)
        #        yield m * s + (1.0 - m) * driven  # results in attenuation (e.g. at 0.5, the two parts are not suddenly half as loud only because they are going to be mixed)
        yield driven + m * s * 0.75  # attenuate to avoid clipping


def drive_deprecated(signal: Union[Iterable, float], gain: float = 1.5, soften: float = 0.5, exp: int = 3):
    """Signal boosting generator to make it sound 'fatter'. Depending on the parameters,
    This can range from a clean boost, via overdrive (if you mix in undriven signal afterwards),
    to distortion (without any simulated physical tube amp effects, though).

    :param signal: source signal iterable/generator
    :param gain: linear amplitude scaling factor (default 1.5)
    :param soften: part of the saturation amplitude range which is rounded off towards the peak (default 0.5 as 1.5-0.5 = 1.0 which results in an unclipped signal)
    :param exp: steepness of the rounding-off (default 3). 1 would result in no effect, 5 results in sharper rounding, but overshoots and requires reducing both the `boost` and `softing` parameters, e.g. to 1.25 and 0.25 respectively.
    https://www.wolframalpha.com/input?i=plot+1.25x+-+0.25*x%5E5+between+-1+and+1
    """
    if exp % 2 == 0:
        raise ValueError("exp must be an uneven integer number")
    max_val = abs(gain - soften)
    if max_val != 0.0:
        print("Warning: assuming a [+1,-1] signal, your drive settings might "
              f"lead to {'clipping' if max_val > 1.0 else 'attenuation'} "
              f"(gain {gain} - soften {soften} = max {max_val})")
    for s in signal:
        yield gain * s - soften * s ** exp


def delay(signal: Union[Iterable, float],
          delay: Union[Iterable, float] = 1.0,  # in seconds
          mixin: Union[Iterable, float] = 0.6,  # mixin
          feedback: Union[Iterable, float] = 0.5,  # feedback mixin
          ):
    """Creates a generator which adds a delay effect to the `signal`.

    This echo effect can be used for some interesting synthesizer patches (just don't overdo it).
    It is also the basis of the `reverb_room/hall/church` effects.
    If you're technical, you might be able to use this for phase-shifting.

    Limitations:
     - Changing the delay's time while sound is playing creates audible popping noises
       due to dropped samples.
     - To avoid clipping, the original signal is attenuated (reduced) by ca. -3dB. This should
       adapt to the mixin and feedback, but currently does not.

    :param signal: Signal to process
    :type signal: Iterable or float
    :param delay: Delay in seconds (default 1.0)
    :type delay: Iterable or float
    :param mixin: Gain with which to mix the delayed signal back into the original signal (default 0.6)
    :type mixin: Iterable or float
    :param feedback: Gain with which the result (including delay) will be, in turn, used as the source
      signal for the coming delay (default 0.5).
    :type feedback: Iterable or float
    :return: Signal with added delay effect
    :rtype: float
    """
    q = Queue()
    prev_delay = None
    #    prev_mixin = None
    #    prev_feedback = None
    delay_in_samples = None
    for s, d, m, f in gen_iter(signal, delay, mixin, feedback):
        if d != prev_delay:
            delay_in_samples = int(d * sample_rate)
        #        if m != prev_mixin:
        #            # At mixin == 0.5, both weights are 1.0.
        #            # With mixin -> 1.0, m_ds stays at 1.0, m_s decreases towards 0.0
        #            # With mixin -> 0.0, m_ds decreases towards 0.0, m_s stays at 1.0
        #            m_ds = min(m * 2.0, 1.0)
        #            m_s = min(max(2.0 - m * 2.0, 0.0), 1.0)
        #        if f != prev_feedback:
        #            f_out = min(f * 2.0, 1.0)
        #            f_s = min(max(2.0 - f * 2.0, 0.0), 1.0)
        mismatch = q.qsize() - delay_in_samples
        output = s
        if mismatch >= 0:
            if mismatch >= 2:
                delayed_sample = (q.get(block=False) + q.get(block=False)) / 2.0
            else:
                delayed_sample = q.get(block=False)
            if m != 0.0:  # attempt to reduce CPU cycles used
                #                output = m * delayed_sample + (1.0 - m) * s
                #                output = m_ds * delayed_sample + m_s * s
                output = m * delayed_sample + s * 0.75  # attenuate to avoid clipping
                if f != 0.0:  # attempt to reduce CPU cycles used
                    #                    s = f * output + (1.0 - f) * s
                    #                    s = f_out * output + f_s * s
                    s = f * output + s * 0.75  # attenuate to avoid clipping
        else:
            output = s
        yield output
        q.put(s)
        prev_delay = d


#        prev_mixin = m
#        prev_feedback = f

def reverb_room(signal: Union[Iterable, float]):
    """Create a generator with a prefab "room reverb" effect. Nothing fancy, feel free to improve!

    :param signal: Signal to add reverb to
    :type signal: Iterable or float
    :return: Yields signal with reverb effect
    :rtype: float
    """
    # use double the delay everywhere and mixin = 0.3 for church-like reverb
    d1 = delay(signal, 0.05, 0.3, 0.2)
    d2 = delay(d1, 0.08, 0.2, 0.2)
    yield from d2


def reverb_hall(signal: Union[Iterable, float]):
    """Create a generator with a prefab "hall reverb" effect. Nothing fancy, feel free to improve!

    :param signal: Signal to add reverb to
    :type signal: Iterable or float
    :return: Yields signal with reverb effect
    :rtype: float
    """
    # use double the delay everywhere and mixin = 0.3 for church-like reverb
    d1 = delay(signal, 0.05, 0.3, 0.2)
    d2 = delay(d1, 0.08, 0.2, 0.2)
    d3 = delay(d2, 0.1, 0.2, 0.2)
    d31, d32 = split(d3, 2)
    d3f = low_pass_filter(d31, alpha=0.85)
    d4 = delay(d3f, 0.115, 0.28, 0.2)
    d5 = delay(d4, 0.15, 0.15, 0.2)
    d5f = low_pass_filter(d5, alpha=0.6)
    m = mix([d5f, d32], [1.0, 0.4])
    yield from m


def reverb_church(signal: Union[Iterable, float]):
    """Create a generator with a prefab "church reverb" effect. Nothing fancy, feel free to improve!

    Sounds currently a bit too echo-y, particularly for percussive instruments. Works rather well for
    floating soundscapes, though.

    :param signal: Signal to add reverb to
    :type signal: Iterable or float
    :return: Yields signal with reverb effect
    :rtype: float
    """
    # use double the delay everywhere and mixin = 0.3 for church-like reverb
    d1 = delay(signal, 0.1, 0.3, 0.2)
    d2 = delay(d1, 0.16, 0.3, 0.2)
    d3 = delay(d2, 0.2, 0.3, 0.2)
    d31, d32 = split(d3, 2)
    d3f = low_pass_filter(d31, alpha=0.85)
    d4 = delay(d3f, 0.23, 0.3, 0.2)
    d5 = delay(d4, 0.3, 0.3, 0.2)
    d5f = low_pass_filter(d5, alpha=0.6)
    m = mix([d5f, d32], [1.0, 0.2])
    yield from m


def reverb_todo(signal: Union[Iterable, float],
                spread: Union[Iterable, float] = 0.5,  # 0.5 = default delay lengths
                mixin: Union[Iterable, float] = 0.3,
                # mixin % of total signal (>0.5 is higher than original signal. For 1.0, nothing of the original signal will be left -- can be used for phase shifting)
                feedback: Union[Iterable, int] = 0.2,  # feedback mixed back in
                room_hall: str = "room",
                ):
    """TODO: This should become a reverb that can be easily modified by turning  one "time" and one "intensity" knob
    """
    s1, s2, s3, s4 = split(signal, 4)
    d1 = delay(signal, mult(spread, 0.05 / 0.5), mixin, feedback)
    d2 = delay(d1, mult(spread, 0.08 / 0.5), mixin, feedback)
    if room_hall.lower() == "room":
        yield from d2
    elif room_hall.lower() == "hall":
        d3 = delay(d2, mult(spread, 0.1 / 0.5), mixin, feedback)
        d31, d32 = split(d3, 2)
        d3f = low_pass_filter(d31, alpha=0.85)
        d4 = delay(d3f, mult(spread, 0.115 / 0.5), mixin, feedback)
        d5 = delay(d4, mult(spread, 0.15 / 0.5), mixin, feedback)
        d5f = low_pass_filter(d5, alpha=0.6)
        m = mix([d5f, d2], [0.8, 0.2])
        yield from m


def portamento(
        freq: Union[Iterable, float],  # Hz
        gate: Union[Iterable, float],  # 0 == low, >0 == high
        duration: Union[Iterable, float] = 0.1,  # in seconds
):
    """Creates a generator to add portamento between notes.

    Portamento "carries" the frequency from one note to the other, adding a smooth, sweeping quality.

    Note that, because it needs to be able to influence frequency values, it acts on the frequency values
    (upstream of oscillators), while many other effects act on sample values (downstream of oscillators).

    This particular implementation uses legato portamento. This means than only notes that are played
    in a connected way will have portamento added between them (the new one pressed before the old one is released).
    You can use this to more precisely control the sound in the same piece: Use a more normal or stakkato-like playing
    style to get non-portamento transitions, and a legato style for portamento transitions. This distinction
    is what the gate signal is needed for.

    TODO: Add a parameter to change styles from legato to always.

    :param freq: Frequency signal to carry between notes, in Hz
    :type freq: Iterable or float
    :param gate: Gate signal
    :type gate: Iterable or float
    :param duration: Transition time between two notes, in seconds
    :type duration: Iterable or float
    :return: Yields the modified frequency
    :rtype: float
    """
    curr_freq = 440.0
    target_freq = 440.0
    step = 0.0
    prev_gate = 0
    for f, g, d in gen_iter(freq, gate, duration):
        if prev_gate == 0.0 and g > 0.0:
            curr_freq = f
            target_freq = f
            step = 0.0
        if g > 0:
            if f != target_freq:
                # Starting new slide
                target_freq = f
                num_steps = max(d * sample_rate, 1)
                step = (target_freq - curr_freq) / num_steps
            elif curr_freq != target_freq:
                # Continuing slide
                if step > 0:
                    curr_freq = min(curr_freq + step, target_freq)
                elif step < 0:
                    # Continuing downwards slide
                    curr_freq = max(curr_freq + step, target_freq)
        prev_gate = g
        yield curr_freq
