import torch
import json
from genie import PianoGenieAutoencoder, SOS


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


def letter_to_button_26(letter):
    # Map letters to button indices for a 26-letter keyboard layout
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if letter in alphabet:
        return alphabet.index(letter)
    else:
        return 0  # Default case for unsupported characters


# read typing_intervals.csv
import csv

# 4) Example usage:
#    Suppose the user hits button 3 at time=0.57s with velocity=90:
# k1, h, p = step(b_i=3, t_i=0.57, v_i=90, k_prev=k_prev, h=h)

notes = []

filename = "typing_wpm_timestamps.csv"
filename_no_ext = filename.split(".")[0]

with open(filename, "r") as f:
    reader = csv.reader(f)
    # skip header
    next(reader)
    for row in reader:
        print(row)
        letter, time, wpm = row[0], row[1], row[2]

        if not letter or not time:
            print("Skipping empty row")
            continue

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
        velocity = int(wpm * 2)
        # ensure velocity is in range 1-127
        velocity = max(1, min(127, velocity))
        k_prev, h, probs = step(b_i=button, t_i=time, v_i=velocity, k_prev=k_prev, h=h)
        notes.append((k_prev.item(), time, velocity))
print(notes)
# generate the midi file
import pretty_midi
import time

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
filename = f"output_{time.time()}_{filename_no_ext}.mid"
pm.write(filename)
