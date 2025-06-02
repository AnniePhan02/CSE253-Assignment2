import torch
import json
from genie import PianoGenieAutoencoder, SOS

# generate the midi file
import pretty_midi
import time as time_lib
import csv


cfg = json.load(open("cfg.json"))
model = PianoGenieAutoencoder(cfg)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# 2) Prepare decoder state for a single stream
#    We’ll run batch_size=1 since we’re in interactive mode
h = model.dec.init_hidden(batch_size=1)
# k_prev holds the last output key index; start with SOS
k_prev = torch.full((1, 1), SOS, dtype=torch.long)

# 3) Now, each time the user presses a button, you have:
#    b_i: integer 0…7    (which button)
#    t_i: float          (absolute onset time in seconds)
#    v_i: int 1…127      (velocity)
#
# You’ll convert these into tensors, call the decoder, sample/argmax,
# and then feed that key back in as the next k_prev.


def step(b_i, t_i, v_i, k_prev, h):
    # 3a) button needs to be the *real-valued* centroid in [–1,1]
    b_real = model.quant.discrete_to_real(torch.tensor([[b_i]]))  # → shape (1,1)
    # 3b) wrap time & velocity
    t = torch.tensor([[t_i]], dtype=torch.float)
    v = torch.tensor([[v_i]], dtype=torch.float)

    # 3c) run decoder for one timestep
    with torch.no_grad():
        logits, h = model.dec(k_prev, t, b_real, v, h)
        # logits: (1,1,88)
        probs = torch.softmax(logits[0, 0], dim=-1)
        k_i = torch.multinomial(probs, num_samples=1)  # or .argmax()

    return k_i.reshape(1, 1), h, probs


# 4) Example usage:
#    Suppose the user hits button 3 at time=0.57s with velocity=90:
# k1, h, p = step(b_i=3, t_i=0.57, v_i=90, k_prev=k_prev, h=h)


# version 1
def letter_to_button_keyboard(letter):
    # Map letters on the keyboard to button indices, top row, middle row, bottom row
    top = "qwertyuiop"
    middle = "asdfghjkl"
    bottom = "zxcvbnm"
    if letter in top:
        return min(top.index(letter), 8), 40
    elif letter in middle:
        return min(middle.index(letter), 8), 80
    elif letter in bottom:
        return min(bottom.index(letter), 8), 120
    else:
        return 0, 0


# V2
def letter_to_button_26(letter):
    # Map letters to button indices for a 26-letter keyboard layout
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if letter in alphabet:
        return alphabet.index(letter)
    else:
        return 0  # Default case for unsupported characters


def letter_to_button_26_rowwise(letter):
    # Map letters to button indices for a 26-letter keyboard layout, row-wise
    alphabet = "qwertyuiopasdfghjklzxcvbnm"
    if letter in alphabet:
        return alphabet.index(letter)
    else:
        return 0  # Default case for unsupported characters


def extract_pitches_from_midi(midi_path):
    """
    Load a MIDI file and return a list of all note pitches (as MIDI numbers).
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    pitches = []

    # Loop over all instruments (tracks) in the file
    for instrument in pm.instruments:
        # Skip drum tracks if you only care about pitched notes:
        if instrument.is_drum:
            continue
        # Loop over each Note object in this instrument
        for note in instrument.notes:
            pitches.append(note.pitch)
    return pitches


# metric one
def infer_key_from_pitches(pitches):
    """
    Given a list of MIDI pitches (0–127), figure out which major key
    (0=C major, 1=C♯ major, …, 11=B major) has the most pitches landing
    on its diatonic pitch-classes.

    Returns (best_key_root, is_major=True).
    """
    # Define diatonic sets for all 12 major keys:
    # pitch-class → integer 0…11 where C=0, C♯=1, …, B=11
    major_scales = {
        0: {0, 2, 4, 5, 7, 9, 11},  # C major
        1: {1, 3, 5, 6, 8, 10, 0},  # C♯ major
        2: {2, 4, 6, 7, 9, 11, 1},  # D major
        3: {3, 5, 7, 8, 10, 0, 2},  # E♭/D♯ major
        4: {4, 6, 8, 9, 11, 1, 3},  # E major
        5: {5, 7, 9, 10, 0, 2, 4},  # F major
        6: {6, 8, 10, 11, 1, 3, 5},  # F♯ major
        7: {7, 9, 11, 0, 2, 4, 6},  # G major
        8: {8, 10, 0, 1, 3, 5, 7},  # A♭ major
        9: {9, 11, 1, 2, 4, 6, 8},  # A major
        10: {10, 0, 2, 3, 5, 7, 9},  # B♭ major
        11: {11, 1, 3, 4, 6, 8, 10},  # B major
    }

    # Build a histogram of pitch-classes
    pc_counts = [0] * 12
    for p in pitches:
        pc = p % 12
        pc_counts[pc] += 1

    # For each major key, count how many pitches are in its diatonic set
    best_key, best_count = None, -1
    for key_root, scale_set in major_scales.items():
        count_in_scale = sum(pc_counts[pc] for pc in scale_set)
        if count_in_scale > best_count:
            best_key, best_count = key_root, count_in_scale

    return best_key  # integer 0…11 (C major=0, C♯ major=1, etc.)


def compute_in_scale_ratio(notes, key_root=None):
    """
    notes: list of (midi_pitch, onset_time, velocity)
    key_root: if you already know the key (0…11), pass it in;
              otherwise, set key_root=None to auto-infer.
    Returns: (key_root, in_scale_ratio)
    """
    # 1) Extract just the pitches
    pitches = [p for p in notes]
    if key_root is None:
        key_root = infer_key_from_pitches(pitches)

    # 2) Get diatonic set for that key
    #    (using the same major_scales dict from above)
    major_scales = {
        0: {0, 2, 4, 5, 7, 9, 11},
        1: {1, 3, 5, 6, 8, 10, 0},
        2: {2, 4, 6, 7, 9, 11, 1},
        3: {3, 5, 7, 8, 10, 0, 2},
        4: {4, 6, 8, 9, 11, 1, 3},
        5: {5, 7, 9, 10, 0, 2, 4},
        6: {6, 8, 10, 11, 1, 3, 5},
        7: {7, 9, 11, 0, 2, 4, 6},
        8: {8, 10, 0, 1, 3, 5, 7},
        9: {9, 11, 1, 2, 4, 6, 8},
        10: {10, 0, 2, 3, 5, 7, 9},
        11: {11, 1, 3, 4, 6, 8, 10},
    }
    scale_set = major_scales[key_root]

    # 3) Count how many pitches lie in that set
    in_scale = sum(1 for p in pitches if (p % 12) in scale_set)
    total = len(pitches)
    ratio = in_scale / total if total > 0 else 0.0

    return key_root, ratio


def eval_map_key_to_pitches(pitches, filename):

    filename_no_ext = filename.split(".")[0]

    letters = []

    with open(filename, "r") as f:
        # print(f"Reading {filename} for letter-to-pitch mapping...")
        reader = csv.reader(f)
        # skip header
        next(reader)
        for row in reader:
            # print(row)
            letter, time, wpm = row[0], row[1], row[2]

            if not letter or not time or len(letter) != 1 or not letter.isalpha():
                # print("Skipping empty row")
                continue

            if wpm == "Infinity" or wpm == "inf":
                wpm = 80

            # check if wpm is numeric
            # if not wpm.isnumeric():
            #     print(f"Skipping non-numeric wpm: {wpm}")
            #     continue

            time = float(time)
            wpm = float(wpm)
            letter = letter.lower()

            letters.append(letter)

    letter_to_pitch_counts = {}

    if len(letters) != len(pitches):
        raise ValueError(
            f"Mismatch: {len(letters)} letters from CSV but {len(pitches)} notes provided."
        )

    for letter, pitch in zip(letters, pitches):
        # Ensure there’s a sub-dictionary for this letter
        if letter not in letter_to_pitch_counts:
            letter_to_pitch_counts[letter] = {}

        # Increment the count for this pitch under that letter
        subdict = letter_to_pitch_counts[letter]
        subdict[pitch] = subdict.get(pitch, 0) + 1

    return letter_to_pitch_counts


def baseline(filename):
    notes = []

    filename_no_ext = filename.split(".")[0]

    with open(filename, "r") as f:
        reader = csv.reader(f)
        # skip header
        next(reader)
        for row in reader:
            print(row)
            letter, time, wpm = row[0], row[1], row[2]

            if not letter or not time or len(letter) != 1 or not letter.isalpha():
                # print("Skipping empty row")
                continue

            if wpm == "Infinity" or wpm == "inf":
                wpm = 80

            # check if wpm is numeric
            # if not wpm.isnumeric():
            #     print(f"Skipping non-numeric wpm: {wpm}")
            #     continue

            time = float(time)
            wpm = float(wpm)

            print(letter, time, wpm)
            # convert letter to button index and velocity
            letter = letter.lower()

            # velocity is used from mapping function, mapping is based on keyboard layout
            button = letter_to_button_26_rowwise(letter)
            velocity = int(wpm) * 2
            # ensure velocity is in range 1-127
            velocity = max(1, min(127, velocity))

            # do a naive mapping of button to key index, without using step
            key_to_add = (
                button + 21
            )  # C4 is MIDI note 60, so we offset by 21 to get the right range
            notes.append((key_to_add, time, velocity))
    return notes


# read typing_intervals.csv


# 4) Example usage:
#    Suppose the user hits button 3 at time=0.57s with velocity=90:
# k1, h, p = step(b_i=3, t_i=0.57, v_i=90, k_prev=k_prev, h=h)


# generate midi from timestamps
notes = []

filename = "english_3.csv"
filename_no_ext = filename.split(".")[0]

with open(filename, "r") as f:
    reader = csv.reader(f)
    # skip header
    next(reader)
    for row in reader:
        print(row)
        letter, time, wpm = row[0], row[1], row[2]

        if not letter or not time or len(letter) != 1 or not letter.isalpha():
            # print("Skipping empty row")
            continue

        if wpm == "Infinity" or wpm == "inf":
            wpm = 80

        # check if wpm is numeric
        # if not wpm.isnumeric():
        #     print(f"Skipping non-numeric wpm: {wpm}")
        #     continue

        time = float(time)
        wpm = float(wpm)

        print(letter, time, wpm)
        # convert letter to button index and velocity
        letter = letter.lower()

        # velocity is used from mapping function, mapping is based on keyboard layout
        button = letter_to_button_26(letter)
        velocity = int(wpm) * 2
        # ensure velocity is in range 1-127
        velocity = max(1, min(127, velocity))
        k_prev, h, probs = step(b_i=button, t_i=time, v_i=velocity, k_prev=k_prev, h=h)
        notes.append((k_prev.item(), time, velocity))
print(notes)


# use baseline

use_baseline = True

if use_baseline:
    notes = baseline(filename)

pm = pretty_midi.PrettyMIDI()
instr = pretty_midi.Instrument(program=0)

for i, (note, onset, vel) in enumerate(notes):
    # define a duration for each note
    if i + 1 < len(notes):
        end = notes[i + 1][1]
    else:
        end = onset + 0.5
    pm_note = pretty_midi.Note(velocity=vel, pitch=note, start=onset, end=end)
    print(pm_note)
    instr.notes.append(pm_note)

pm.instruments.append(instr)
filename = f"output_{filename_no_ext}_{time_lib.time()}.mid"

if use_baseline:
    filename = f"baseline_{filename_no_ext}_{time_lib.time()}.mid"
pm.write(filename)


# run evals
import glob

# Look for files ending in .mid or .midi in the current directory
midi_patterns = ["*.mid", "*.midi"]
all_files = []
for pattern in midi_patterns:
    all_files.extend(glob.glob(pattern))

if not all_files:
    print("No MIDI files found in the current directory.")


# For each MIDI file, extract pitches and print a summary
for midi_file in sorted(all_files):
    pitches = []
    try:
        pitches = extract_pitches_from_midi(midi_file)
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
        continue
    print(f"File: {midi_file}")
    print(compute_in_scale_ratio(pitches))

    input_name = ""
    if "performance_1" in midi_file:
        input_name = "performance_1.csv"
    elif "performance_2" in midi_file:
        input_name = "performance_2.csv"
    elif "performance_3" in midi_file:
        input_name = "performance_3.csv"
    elif "english_1" in midi_file:
        input_name = "english_1.csv"
    elif "english_2" in midi_file:
        input_name = "english_2.csv"
    elif "english_3" in midi_file:
        input_name = "english_3.csv"

    key_to_pitches = eval_map_key_to_pitches(pitches, input_name)

    for key, pitches in key_to_pitches.items():
        print(f"Key {key}:")
        print("  Pitches:", ", ".join(str(p) for p in pitches))
        print()
