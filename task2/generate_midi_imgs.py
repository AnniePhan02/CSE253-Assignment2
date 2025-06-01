import os
import mido
import matplotlib.pyplot as plt

from mido import MidiFile


def midi_to_note_events(mid):
    events = []
    time = 0
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((abs_time, msg.note))
    return events


def plot_piano_roll(events, filename):
    if not events:
        print(f"No note events found in {filename}")
        return
    times, notes = zip(*events)
    plt.figure(figsize=(10, 4))
    plt.scatter(times, notes, s=5)
    plt.title(f"Piano Roll: {filename}")
    plt.xlabel("Time (ticks)")
    plt.ylabel("MIDI Note")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.splitext(filename)[0] + ".png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def process_directory(directory):
    for file in os.listdir(directory):
        if file.lower().endswith(".mid") or file.lower().endswith(".midi"):
            path = os.path.join(directory, file)
            print(f"Processing {file}")
            try:
                midi = MidiFile(path)
                events = midi_to_note_events(midi)
                plot_piano_roll(events, os.path.join(directory, file))
            except Exception as e:
                print(f"Failed to process {file}: {e}")


# Replace with your folder path
midi_folder = "./26bin_5-31"
process_directory(midi_folder)
