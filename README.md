# TinkerSynth

An amateurish library of synthesizer components/effects based on Python's [generators](https://wiki.python.org/moin/Generators).

Supports MIDI devices and sound output (currently no stereo and no polyphony).

This started out as a fun little exercise, but turned out to just not stop growing.
I found it surprisingly intuitive to use Python generators as self-contained components for this.

The full documentation (including a description of all functionality) can be found in
the [documentation](https://schuderer.github.io/tinkersynth/) (also in the directory `docs`).

## Quick start

Clone (or download) this repository, install the requirements,
and, if you wish, run the example script:

```bash
cd somewhere_where_you_keep_code
git clone https://github.com/schuderer/tinkersynth.git
pip install -r requirements.txt
python examples.py
```
(If the installation of python-rtmidi fails, you might need to install some of its
[requirements](https://github.com/SpotlightKid/python-rtmidi/blob/master/INSTALL.rst#requirements))

If no MIDI device is connected, you can use the PC keyboard as input (US layout).

Look at [examples.py](https://schuderer.github.io/tinkersynth/examples.html) to see how synth pipelines (also called "patches") can be defined and play around with
them. The example code automatically restarts the synthesizer if you change the code
to make it easier to just play around and tweak stuff.

The terms `node` and `generator` are used interchangeably. The term `signal` usually refers
to another generator's output which is processed by the current generator (sometimes also called `stream`).
But it can in principle also be a constant value. It does not have to be sound
(for example, `sine`/`square`/`sawtooth`/`triangle` oscillators consume a `frequency` signal
and output a `samples` signal). Pretty much all signals are just float values and
can be used as parameters as well as "real" sound data (e.g. when using 
oscillators with low frequencies for controlling filter or transpose parameters, 
cf. LFOs, modulation).

For a complete list of all available nodes, see the [documentation](https://schuderer.github.io/tinkersynth/)
or the subdirectory `docs`.

## What is a synthesizer?

A music synthesizer creates simple basic wave forms (e.g. sine, square, sawtooth)
using oscillators and modifies these by passing them through various effect stages
(also called "nodes"). For example, creating half a dozen or so sawtooth waves,
slightly detuning each of them, and then mixing them together again creates a
familiar Trance/Techno-like synthesizer sound. Each stage or node is implemented
here as a Python generator.

## What are Python generators and how are they used here?

In Python, a [generator](https://wiki.python.org/moin/Generators) is like a mixture between a function and a list. It creates
one value after another. The nicety is that it does not create all of them at once,
but only when e.g. a for-loop asks it for the next value. Generator code usually looks
like an ordinary function with a loop inside. And inside the loop, they use the keyword
`yield` to deliver a value instead of `return`.

A generator can also consume values from another generator at the same time as it
produces its own values. That way, generators can process and manipulate data streams,
one value after another. For a simple example, see the `gain` node in 
[synthesizer.py](https://schuderer.github.io/tinkersynth/synthesizer.html).
It consumes a stream of samples (and gain parameters) and produces a stream of scaled
samples.

Each node/generator here is rather simple and can be used and understood in
isolation. What makes all of them powerful is the way they can be combined easily and
in a literally infinite number of ways, creating unheard of effects and sounds.

I also highly recommend trying to create your own generators. For ideas on common effects,
look for example here: https://www.musicdsp.org/en/latest/

Some nice Python tools for common generator-related problems can be found
in Python's itertools: https://docs.python.org/3/library/itertools.html

## Could you add functionality xyz?

It might already be there! Look at the [documentation](https://schuderer.github.io/tinkersynth/) (also in directory `docs`)
of the `synthesizer.py` module.

If it's not there, it might still be there, but in different shape. ;) As this project is meant as a playground 
with reusable building blocks
for your own Python experiments, you'll find that, for example:
 - LFOs (low-frequency-oscillators) are just oscillators with lower frequencies
 - AM modulation is just a `gain` node with a `scaled` oscillator as `gain` parameter (cf. vibrato)
 - FM modulation is just a `transpose` node with a `scaled` oscillator as `s` parameter
 - An envelope-driven cutoff is just an (possibly `scaled`) envelope generator output used as `alpha` parameter for `low_pass_filter`

There are also a lot of helper functions to help you achiev what you want,
such as basic arithmetic functions, scaling, constraints, if_else, or applying
arbitrary custom functions to the signal.

Of course, you can also still create your own nodes. Making a Python generator for
signal processing is surprisingly straightforward. Look at the 
[synthesizer](https://schuderer.github.io/tinkersynth/synthesizer.html) source code
for inspiration (don't fret if there are a few functions look a bit unwieldy -- most effects aren't).

If you read this far - okay, I admit, there is still a lot to do here, such as:
 - Resonance for our `low_pass_filter`
 - Porting (and understanding) that nice Moog filter code I found somewhere
 - A better flexible, `reverb`
 - Arpeggiators, which will require a refactor of our `midi_note_source`
 - Saving output to file
 - Playing a pre-determined melody
 - Polyphony!
 - Oscilloscope!
 - Stereo!
 - and many more

Who knows, maybe you have gotten one of them to work or made something you want to show off?
Create a GitHub issue and let us know.
