"""
This module is needed to connect to a MIDI device.

Use it to create generators (nodes) for your various MIDI inputs: note frequencies, gate signal, controllers (knobs
and dials), etc.

For examples, see `examples.py`
"""

from math import sqrt
from threading import RLock
from time import sleep
from typing import Optional, Union

try:
    import rtmidi
except ImportError:
    rtmidi = None
    print("Note: MIDI library `python-rtmidi` not found. Falling back to computer keyboard.")

try:
    from pynput import keyboard
except ImportError:
    if rtmidi is None:
        print("Error: Neither MIDI library `python-rtmidi` nor keyboard library `pynput` found. Exiting.")
        exit(1)
    else:
        keyboard = None

CHANNEL_MASK = 0b00001111
EVENT_MASK = 0b11110000
NOTE_ON = 0b10010000
NOTE_OFF = 0b10000000
PITCH_BEND = 0b11100000
PROG_CHANGE = 0b11000000  # data byte contains instruments 0-127. Drums are on 9 (10 when counting from 1)
SET_CONTROLLER = 0b10110000  # 123 = panic
# The most common controller numbers are the following:
#
# 0 = Sound bank selection (MSB)
# 1 = Modulation wheel, often assigned to a vibrato or tremolo effect.
# 7 = Volume level of the instrument
# 10 = Panoramic (0 = left; 64 = center; 127 = right)
# 11 = Expression (sometimes used also for volume control or similar, depending on the synthesizer)
# 32 = Sound bank selection (LSB)
# 64 = Sustain pedal (0 = no pedal; >= 64 => pedal ON)
# 121 = All controllers off (this message clears all the controller values for this channel,
#       back to their default values)
# 123 = All notes off (this message stops all the notes that are currently playing)

# Velocity scaling variants
NO_VELOCITY = 0
LINEAR = 1
SQUARED = 2
CUBIC = 3
SQUARE_ROOT = 4
LOW = 0
HIGH = 1

STANDARD_PITCH = 440  # tuning of A4

# Contains the currently pressed MIDI notes
notes = {}

# Last known CC values
controller_vals = {
    "pitch_bend": 0.5,  # special case, not a "real" controller, hence not in mappings
    "modulation": 0.0,
    "volume": 0.5,
    "expression": 0.0,
    "sustain": 0.0,
}

# Will contain the CC value Generators to be used by synthesizer
controllers = {}
cc = controllers  # shorthand

# Mapping from MIDI CC numbers to human-readable names
controller_mappings = {
    1: "modulation",
    7: "volume",
    11: "expression",
    64: "sustain",
}

# Values for PC keyboard input mode
pc_key_velocity = 0.6
pc_key_c = 60
pc_key_notes = {
    "a": 0,
    "w": 1,
    "s": 2,
    "e": 3,
    "d": 4,
    "f": 5,
    "t": 6,
    "g": 7,
    "y": 8,
    "h": 9,
    "u": 10,
    "j": 11,
    "k": 12,
    "o": 13,
    "l": 14,
    "p": 15,
    ";": 16,
    "'": 17,
    "]": 18,
    "\\": 19,
}

# Thread control
notes_lock = RLock()
controllers_lock = RLock()
stop = False


def get_freq(midi_pitch: int):
    """
    Calculate frequency in Hz from MIDI pitch/note number (60 = C3) with
    relation to A4 as defined in STANDARD_PITCH.

    Uses equal temperament.

    :param midi_pitch: MIDI pitch/note number
    :type midi_pitch: int
    :return: frequency in Hz
    :rtype: float
    """
    factor = 2 ** ((midi_pitch - 69) / 12)
    return STANDARD_PITCH * factor


def init_midi_in(
        port_or_device_name: Optional[Union[int, str]] = None,
        rtmidi_api: int = rtmidi.API_UNSPECIFIED,
        velocity_curve: int = SQUARE_ROOT,
        # passing this through to the handler is a necessary evil for performance reasons
):
    """Initializes and returns Midi input.
    You normally should not have to use this function. To get frequency and gate generators which are already
    hooked up to a MIDI device, please use `midi_note_source`.

    If things don't work, try different ports for your device, or mash a few
    MIDI-related buttons on your device. You might also need to specify
    a different rtmidi_api: https://spotlightkid.github.io/python-rtmidi/rtmidi.html#low-level-apis

    :param port_or_device_name: Optional integer port index or substring to find in device listing.
      The default is to ask the user interactively.
    :type port_or_device_name: int or str or None
    :param rtmidi_api: Optional rtmidi low-level API choice (default should work in most cases)
    :type rtmidi_api: enum from rtmidi._rtmidi (int)
    :param velocity_curve: Velocity curve to use (default midi.SQUARE_ROOT). One of NO_VELOCITY, LINEAR, SQUARED,
      CUBIC, SQUARE_ROOT.
    :type velocity_curve: enum (int)
    """
    available_ports = []
    pc_key_port = "Computer keyboard (A row: white, Q row: black, Z/X: octave, C/V: velocity)"
    port = 0
    midi_in = None
    if keyboard is not None:
        available_ports = [pc_key_port]
    if rtmidi is not None:
        midi_in = rtmidi.MidiIn(rtapi=rtmidi_api)
        available_ports = midi_in.get_ports() + available_ports
    if len(available_ports) == 0:
        print("No input devices available. Exiting.")
        exit(2)
    if port_or_device_name is None:
        print("The following MIDI In ports have been found:")
        for i, port_name in enumerate(available_ports):
            print(f" - [{i}] {port_name}")
        port = input("Enter the number of the port you want to connect to (default: 0, Ctrl-C to abort): ")
        port = int(port) if port != '' else 0
    elif isinstance(port_or_device_name, str):
        search_hits = [port_or_device_name.lower() in n.lower() for n in available_ports]
        for i, port_name, found in zip(range(len(search_hits)), available_ports, search_hits):
            print(f"  {'-->' if found else '   '} [{i}] {port_name}")
            port = i if found else port
        if not any(search_hits):
            raise ValueError(f"Could not find '{port_or_device_name}' in any device name.")
    else:
        port = port_or_device_name
    port_name = available_ports[port]

    if rtmidi is not None and port_name != pc_key_port:
        print(f"Opening MIDI port [{port}]: {available_ports[port]}")
        midi_in.open_port(port)

        #    print(f"init_midi: {threading.active_count(), threading.current_thread()}")
        # Sometimes, we get a TypeError along the lines of dict/tuple/... is not callable.
        # This is probably due to rtmidi's pyx code keeping a C pointer to the
        # callback, and it being either cleaned up or moved by Python.
        # Wrapping it in the same scope as the creation of MidiIn seems to fix this:
        def wrap_handler(message, data=None):
            #        print(f"midi_callback: {threading.active_count(), threading.current_thread()}")
            _handle_midi_message(message, data=data)

        midi_in.set_callback(wrap_handler, data=velocity_curve)
        input_device = midi_in
    else:
        input_device = start_keyboard_listener()
    return input_device


def _handle_midi_message(message, data=None):
    """Callback function to be called whenever a MIDI message is received."""
    # print(message)
    global notes
    velocity_curve = data
    (event, byte1, byte2), time = message
    # http://www.music-software-development.com/midi-tutorial.html
    # channel = event & CHANNEL_MASK
    event_type = event & EVENT_MASK
    if event_type == NOTE_ON:
        #        print("note on")
        pitch = get_freq(byte1)
        velocity = byte2 / 128
        with notes_lock:
            if pitch not in notes:
                notes[pitch] = _scale_velocity(max(velocity, 0.0), velocity_curve)
    elif event_type == NOTE_OFF:
        #        print("note off")
        pitch = get_freq(byte1)
        # velocity = byte2  # ignored
        # print(f"removing {pitch} from {notes}...")
        with notes_lock:
            if pitch in notes:
                del (notes[pitch])
    # print(f"new dict: {notes}")
    elif event_type == SET_CONTROLLER:
        controller_number = byte1
        value = byte2
        if controller_number == 123:
            print("All Notes Off")
            with notes_lock:
                notes.clear()  # All notes off
        elif controller_number in controller_mappings:
            #            with controllers_lock:
            name = controller_mappings[controller_number]
            controller_vals[name] = value / 127
        else:
            print(f"controller {controller_number} value {value}")
    elif event_type == PITCH_BEND:
        # print("pitch bend")
        # with controllers_lock:
        value = (byte2 << 7) + byte1  # 14 bit value
        controller_vals["pitch_bend"] = 2.0 * (value - 8192) / 16384


def midi_note_source(
        port_or_device_name: Optional[Union[int, str]] = None,
        rtmidi_api: int = rtmidi.API_UNSPECIFIED,
        velocity_curve: int = SQUARE_ROOT,
        retrigger_on_changed_note: bool = False,
):
    """Returns a tuple of two generators (`freqs`, `gates`), where `freqs` yields
    the frequency of the last note. `gates` yields
    LOW (0) if no key is pressed, or a value larger than 0 containing the velocity value otherwise.
    Does currently not support polyphony.
    `freqs()` provides an initial frequency of 440 Hz (as set by `STANDARD_PITCH`) to avoid
    division-by-zero issues in any downstream oscillators.
    `gates()` is LOW initially (if the note buffer is empty).

    The PC keyboard tries to emulate that of Ableton Live. If PC keyboard is selected as device,
    the middle and upper letter row (on US layout) are the white and black keys, respectively.
    Z/X change the octave, and C/V change the velocity.

    :param port_or_device_name: Optional integer port index or substring to find in device listing.
      The default is to ask the user interactively.
    :type port_or_device_name: int or str or None
    :param rtmidi_api: Optional rtmidi low-level API choice (default should work in most cases)
    :type rtmidi_api: enum from rtmidi._rtmidi (int)
    :param velocity_curve: Velocity curve to use (default midi.SQUARE_ROOT). One of NO_VELOCITY, LINEAR, SQUARED,
      CUBIC, SQUARE_ROOT.
    :type velocity_curve: enum (int)
    :param retrigger_on_changed_note: Retrigger the gate if the note changes. This makes percussive AD/AR-Envelopes
      retrigger on legato playing, but makes portamento impossible (default False).
    :return: A tuple of two generators (`freqs`, `gates`)
    :rtype: tuple of generators
    """
    midi_in = None

    def ensure_midi():
        nonlocal midi_in
        if midi_in is None:
            with notes_lock:
                midi_in = init_midi_in(port_or_device_name, rtmidi_api=rtmidi_api, velocity_curve=velocity_curve)

    def freqs():
        ensure_midi()
        global stop
        stop = False
        freq = STANDARD_PITCH
        while True:
            #            print(f"key_source-freqs: {threading.active_count(), threading.current_thread()}")
            with notes_lock:
                for n, vel in notes.items():
                    #                    print(f"found note {n}")
                    freq = n
            yield freq

    def gates():
        ensure_midi()
        newest_note = -1
        curr_velocity = -1
        prev_note = -1
        prev_gate = -1
        while True:
            #            print(f"key_source-gates: {threading.active_count(), threading.current_thread()}")
            with notes_lock:
                curr_gate = int(len(notes) > 0)
                if curr_gate:
                    newest_note = list(notes)[-1]
                    curr_velocity = notes[newest_note]
            #            if curr_gate != prev_gate:
            if retrigger_on_changed_note and curr_gate and prev_gate and newest_note != prev_note:
                # print(f"Note has changed but gate is still HIGH. "
                #       f"Setting gate to LOW momentarily to trigger it as a new note.")
                curr_gate = LOW  # Have to disable for legato portamento to work
            elif curr_gate != prev_gate:
                #                print(f"gate changed to {curr_gate}")
                prev_gate = curr_gate
            prev_note = newest_note
            yield curr_gate * curr_velocity

    # Register some standard midi controllers
    # For more info: http://midi.teragonaudio.com/tech/midispec/ctllist.htm
    create_controllers_from_mapping(controller_mappings, overwrite=True)

    return freqs(), gates()


def _scale_velocity(x: float, kind: int):
    """Transfer function of the linear velocity reading to scale amplitude"""
    if kind == NO_VELOCITY:
        return 1.0
    if kind == LINEAR:
        return x
    if kind == SQUARED:
        return x ** 2 * 0.75 + 0.25
    if kind == CUBIC:
        return x ** 3 * 0.75 + 0.25
    if kind == SQUARE_ROOT:
        return sqrt(x)


def pitch_bend_controller():
    """Generator which delivers the current value of the pitch bend wheel of your MIDI device.

    :return: Yields pitch-bend values between -1 and +1
    :rtype: float
    """
    cc_name = "pitch_bend"
    prev_val = None
    while True:
        #        with controllers_lock:
        val = controller_vals[cc_name]
        if val != prev_val:
            print(f"{cc_name} = {val:2.2f}")
        yield val
        prev_val = val


def make_controller(number: int, name: str, default=0.5, overwrite=False):
    """Returns a generator for the given MIDI controller number and
    registers it under the name of the second argument.

    In most cases, you might prefer to pass a dictionary of mappings to the convenience
    function `create_controllers_from_mapping` instead of using `make_controller`.

    Values will be scaled between 0 and 1.0
    You can choose to use this to override any other default MIDI controllers
    by using their name here. You can also use the same name several times.
    The last controller input will always override the previous one.
    If you don't want this behaviour, please don't register duplicates and avoid
    the following names: volume, modulation, expression, sustain, pitch_bend.

    For information on MIDI controllers, see for example http://midi.teragonaudio.com/tech/midispec/ctllist.htm

    :param number: MIDI CC number
    :type number: int
    :param name: Human-readable name to use when using the controller from the `controllers` (or `cc`) dict.
    :type name: str
    :param default: Default value
    :type default: float
    :param overwrite: Allow overwriting existing controller numbers (default is False)
    :type overwrite: bool
    :return: Controller generator
    :rtype: generator
    """
    if not overwrite and number in controller_mappings:
        raise ValueError(f"Attempt to register controller {number} more than once (as '{name}').")
    controller_mappings[number] = name
    controller_vals[name] = default  # what value is smart to choose?
    prev_val = None

    def gen():
        nonlocal prev_val, name
        while True:
            #        with controllers_lock:
            val = controller_vals[name]
            if val != prev_val:
                print(f"{name} = {val:2.2f}")
            yield val
            prev_val = val

    return gen()


def create_controllers_from_mapping(mapping: dict, default=0.5, overwrite=False):
    """Creates generators for the given `dict` of MIDI controller numbers and names,
    and registers them in the `midi.controllers` dict (shorthand: `midi.cc`).
    The generators are pre-initialized and can be used like any generator, e.g.::

      from midi import cc
      co, res = cc["cutoff"], cc["resonance"]
      bla = low_pass_filter(my_signal, alpha=co, resonance=res)
      # or simply:
      bla = low_pass_filter(my_signal, alpha=cc["cutoff"], resonance=cc["resonance"])

    :param mapping: CC numbers and names
    :type mapping: dict
    :param default: Default value to use before any input is made (0.5 by default)
    :type default: float
    :param overwrite: Whether to overwrite existing controllers with the same name (default False).
    :type overwrite: bool
    :return: Dict of the created generators (no need to use it, you can use `midi.cc["my_cc_name"]` instead.
    :rtype: dict
    """
    generators = {}
    for num, name in mapping.items():
        generator = make_controller(num, name, default, overwrite)
        if name in controllers:
            if not overwrite:
                raise ValueError(f"Controller generator with name '{name}' already exists.")
            print(f"Replacing previous '{name}' controller")
        controllers[name] = generator
        generators[name] = generator
        print(f"Generator for MIDI controller number {num} ({name}) available as midi.cc['{name}']")
    return generators


# Keyboard fallback
def _on_key_press(key):
    global pc_key_c, notes, pc_key_velocity
    try:
        #        print('alphanumeric key {0} pressed'.format(key.char))
        if key.char in pc_key_notes:
            #            print("note on")
            pitch = get_freq(pc_key_c + pc_key_notes[key.char])
            velocity = pc_key_velocity
            with notes_lock:
                if pitch not in notes:
                    notes[pitch] = velocity
        elif key.char == "z":
            pc_key_c = max(pc_key_c - 12, 0)
            print(f"Keyboard now spans MIDI notes {pc_key_c} to {pc_key_c + 12} (60 is C3)")
        elif key.char == "x":
            pc_key_c = min(pc_key_c + 12, 108)
            print(f"Keyboard now spans MIDI notes {pc_key_c} to {pc_key_c + 12} (60 is C3)")
        elif key.char == "c":
            pc_key_velocity = max(pc_key_velocity - 0.1, 0.1)
            print(f"You are playing with velocity {pc_key_velocity}")
        elif key.char == "v":
            pc_key_velocity = min(pc_key_velocity + 0.1, 1.0)
            print(f"You are playing with velocity {pc_key_velocity}")
    except AttributeError:
        #        print('special key {0} pressed'.format(key))
        ...


def _on_key_release(key):
    global pc_key_c, notes
    #    print('{0} released'.format(key))
    try:
        if key.char in pc_key_notes:
            pitch = get_freq(pc_key_c + pc_key_notes[key.char])
            #            print("note off")
            # print(f"removing {pitch} from {notes}...")
            with notes_lock:
                if pitch in notes:
                    del (notes[pitch])
    except AttributeError:
        ...


def start_keyboard_listener():
    print("Keybard input active")
    listener = keyboard.Listener(
        on_press=_on_key_press,
        on_release=_on_key_release)
    listener.start()
    return listener


if __name__ == "__main__":
    print("Test mode. Select your MIDI device, then press some keys to see whether it works.")
    for f, g in midi_note_source():
        print(f"freq: {f}, gate: {g}")
        sleep(1)
