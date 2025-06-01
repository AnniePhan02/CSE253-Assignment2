# %% [markdown]
# # Assignment 2 Music Generation

# %% [markdown]
# ## Task 1: Symbolic Unconditioned Generation
# 
# ### Train a RNN LSTM on 4 melodies (p(x)) and sample new sequences.

# %% [markdown]
# ### Installation & Imports

# %%
# Install required libraries
# !pip install torch pretty_midi matplotlib midi2audio librosa

# %%
# Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pretty_midi
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import librosa
import collections
import math

from torch.utils.data import Dataset, DataLoader
from midi2audio import FluidSynth
from IPython.display import Audio, display
from matplotlib.ticker import MaxNLocator

# %% [markdown]
# ### Data Loading

# %%
# Load the pre‐serialized JSB Chorales dataset
with open("JSB-Chorales-dataset-master/jsb-chorales-quarter.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# We’ll work with the training split here, you can also access 'valid' and 'test'
chorales = data["train"]
print(f"Loaded {len(chorales)} training chorales.")
print("Sample:", chorales[0][:5])

# %% [markdown]
# 
# ### Dataset Context
# The JSB Chorales dataset consists of 382 four-part harmonized chorales by J.S. Bach. It is widely used in symbolic music modeling and has been curated to support machine learning tasks. We use the version released by [Zhuang et al.](https://github.com/czhuang/JSB-Chorales-dataset), which is represented as a sequence of four‐voice chord events (soprano, alto, tenor, bass), quantized to quarter‐note durations. 
# 
# Instead of modeling only the soprano line, we now build a **polyphonic** model that learns full four‐voice chorales in parallel. At each time step, the model will predict an entire 4‐tuple of MIDI pitches (or rests) for all voices simultaneously.
# 
# ### Preprocessing Steps
# 
# 1. **Extract four‐voice chord tuples**  
#    - For each chorale, read each 4‐element chord event (one MIDI pitch per voice).  
#    - Skip any chord where all four voices are rests (`-1, -1, -1, -1`).  
#    - Drop any chorale that has fewer than 10 valid chords.
# 
# 2. **Build a chord vocabulary**  
#    - Collect the set of all unique 4‐tuples (soprano, alto, tenor, bass) across the training split.  
#    - Map each unique chord‐tuple to a distinct integer index.
# 
# 3. **Tokenize each chorale as a sequence of chord‐indices**  
#    - Convert each 4‐tuple in a chorale to its index in the chord vocabulary.  
#    - Discard any chord not found in the vocabulary (e.g., if it only appeared in validation/test).
# 
# 4. **Prepare sequence‐to‐sequence training pairs**  
#    - Slide a fixed‐length window (e.g., 32 chords) over each tokenized chord sequence.  
#    - For each window, the input is the first 32 chord‐indices, and the target is the next 32 chord‐indices (shifted by one).
# 
# 5. **Build `ChordSequenceDataset` and `DataLoader`**  
#    - Wrap the tokenized sequences of indices in a PyTorch `Dataset` that returns `(input_seq, target_seq)` pairs.  
#    - Use a `DataLoader` with a suitable batch size (e.g., 64) to feed the LSTM.
# 
# After these steps, we feed full four‐voice chord sequences into our `MusicRNN` model so that at each step it learns to predict a 4‐voice chord rather than a single monophonic melody.

# %%
# We build a sequence of 4‐tuples for all 4 harmonies: soprano, alto, tenor, bass.
# We skip any chord that is all rests (-1 in every voice), and drop very short chorales.
chord_seqs = []

for chorale in chorales:
    chord_list = []
    for chord in chorale:
        # Chord is either a list/tuple of length 4, or -1 for a complete rest
        if isinstance(chord, (list, tuple)) and len(chord) == 4:
            # Convert any numpy types to int and keep the 4‐tuple as is:
            chord_tuple = (int(chord[0]), int(chord[1]), int(chord[2]), int(chord[3]))
            # If the chord is NOT four rests, we keep it.  (If all four voices are -1, skip.)
            if chord_tuple != (-1, -1, -1, -1):
                chord_list.append(chord_tuple)
    # Only keep chorales longer than 10 chords
    if len(chord_list) > 10:
        chord_seqs.append(chord_list)

print(f"Extracted {len(chord_seqs)} four‐voice sequences.")
print("Example chord‐sequence (first 5 chords):", chord_seqs[0][:5])

# %% [markdown]
# ### Vocabulary & Tokenization

# %%
# Build a set of all unique 4‐tuples (chords) in the training split.

all_chords = sorted({tuple(chord) for seq in chord_seqs for chord in seq})
# Map each chord‐tuple to a unique integer index
chord_to_idx = {chord: i for i, chord in enumerate(all_chords)}
idx_to_chord = {i: chord for chord, i in chord_to_idx.items()}
vocab_size = len(chord_to_idx)

# Convert each chord‐tuple sequence into a list of indices
tokenized_chord_seqs = [[chord_to_idx[ch] for ch in seq] for seq in chord_seqs]

print("Four‐voice chord vocabulary size:", vocab_size)
print("Tokenized example (first 10 chord‐indices):", tokenized_chord_seqs[0][:10])

# %% [markdown]
# ### Dataset Class

# %%
# Create Dataset class for LSTM training. 
# Takes tokenized melody sequences and splits into
# fixed-length input-output pairs.
class ChordSequenceDataset(Dataset):
    def __init__(self, token_chord_seqs, seq_len=32):
        super().__init__()
        self.samples = []
        # Slide a window of length seq_len over each chord‐token sequence
        for seq in token_chord_seqs:
            for i in range(len(seq) - seq_len):
                x = seq[i : i + seq_len]           # input: a sequence of chord‐indices
                y = seq[i + 1 : i + seq_len + 1]   # target: next‐chord at each step
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # Return LongTensors of shape (seq_len,) of chord‐indices
        return torch.tensor(x, dtype=torch.long), \
               torch.tensor(y, dtype=torch.long)

# %% [markdown]
# ### DataLoader Preparation

# %%
# Create batches of (input, target) pairs for training.
seq_len  = 32 # length of each input sequence (tries to predict 32 next notes)
batch_size = 64 # number of sequences per batch (process 64 input-output pairs at a time)

# Create dataset and dataloader
dataset = ChordSequenceDataset(tokenized_chord_seqs, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Total training chord‐sequences: {len(dataset)}")

# %% [markdown]
# ### Training Model

# %%
class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, seq_len=32):
        super(MusicRNN, self).__init__()
        # Embedding now maps each chord‐index to a dense vector
        self.embedding      = nn.Embedding(vocab_size, embedding_dim)
        # Positional embeddings add information about each timestep's position
        self.position_embed = nn.Embedding(seq_len, embedding_dim)

         # LSTM stack: processes the embedded sequence, with dropout between layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # input/output tensors have shape (batch, seq, feature)
            dropout=0.2 # dropout on outputs of all layers except the last
        )

        self.norm    = nn.LayerNorm(hidden_dim) # LayerNorm stabilizes the activations before the final layers
        self.dropout = nn.Dropout(0.3) # Dropout after LSTM to reduce overfitting
        self.fc      = nn.Linear(hidden_dim, vocab_size) # Final linear layer maps hidden states to vocabulary logits

    def forward(self, x):
        batch_size, seqlen = x.size()
        # Create a tensor of positions [0, 1, ..., seq_len-1] for each example
        positions = (torch.arange(seqlen, device=x.device)
                        .unsqueeze(0)
                        .expand(batch_size, seqlen))
        embeddings = self.embedding(x) + self.position_embed(positions)

        out, _     = self.lstm(embeddings)
        out        = self.norm(out)
        out        = self.dropout(out)
        
        logits     = self.fc(out)  # shape: (batch_size, seqlen, vocab_size)
        return logits

# %%
def train_rnn(model, dataloader, vocab_size, num_epochs=10, lr=0.001,
              device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train the MusicRNN on the provided dataloader.

    model: instance of MusicRNN
    dataloader: yields (input_batch, target_batch) pairs
    vocab_size: size of the token vocabulary for loss calculation
    """
    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    # Scheduler reduces LR by 0.5 if validation loss hasn't improved for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            # Forward pass: get logits of shape (batch, seq_len, vocab_size)
            logits = model(xb)

            # Compute cross-entropy loss across all timesteps
            loss = loss_fn(
                logits.view(-1, vocab_size),   # (batch*seq_len, vocab_size)
                yb.view(-1)                     # (batch*seq_len,)
            )

            # Backward pass and gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model parameters
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        # Step the scheduler with the average training loss
        scheduler.step(avg_loss)

# %%
# Trains the Model for 10 epochs
model = MusicRNN(vocab_size=vocab_size, seq_len=32)
train_rnn(model, dataloader, vocab_size, num_epochs=10)

# %% [markdown]
# ### Sampling from the trained LSTM

# %%
# 3 samples: (A) a random 4-note prefix, (B) a single-note "cold" start, or (C) a very short seed.

def sample_diverse(
    model,
    tokenized_seqs,
    max_length=64,
    prefix_type="random_short",  # "random_short", "single", or "fixed"
    fixed_prefix=None,           # only used if prefix_type=="fixed"
    prefix_len=4, 
    first_steps_temp=2.0,        # high temp for initial steps
    normal_temp=1.0,
    top_k=5,
    top_p=0.8,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    prefix_type:
      - "fixed": uses fixed_prefix (list of IDs)
      - "random_short": picks a random melody and takes prefix_len tokens
      - "single": starts from 1 random token
    """
    model.eval().to(device)
    
    # Pick our seed
    if prefix_type == "fixed":
        assert fixed_prefix is not None
        prefix = fixed_prefix
    elif prefix_type == "random_short":
        seq = random.choice(tokenized_seqs)
        prefix = seq[:prefix_len]
    elif prefix_type == "single":
        prefix = [ random.choice(tokenized_seqs)[0] ]
    else:
        raise ValueError("bad prefix_type")

    generated = prefix[:]
    input_seq = torch.tensor([generated], device=device)
    
    def filter_logits(logits):
        from torch.nn.functional import softmax
        logits = logits.clone()
        # Top-k
        if top_k>0:
            kth = torch.topk(logits, top_k)[0][-1]
            logits[logits <  kth] = -1e9
        # Top-p
        if top_p>0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cum > top_p
            mask[...,1:] = mask[...,:-1].clone()
            mask[...,0]  = False
            logits[ sorted_idx[mask] ] = -1e9
        return logits

    for i in range(max_length - len(prefix)):
        # Choose temperature
        temp = first_steps_temp if i < len(prefix) else normal_temp
        
        seq_len = model.position_embed.num_embeddings
        inp = input_seq[:, -seq_len:]
        logits = model(inp)[0, -1, :] / temp
        filt   = filter_logits(logits)
        probs  = F.softmax(filt, dim=-1)
        nxt    = torch.multinomial(probs, 1).item()

        generated.append(nxt)
        input_seq = torch.tensor([generated], device=device)

    return generated

# Try all three strategies:
gens = {}
gens["A_random4"] = sample_diverse(
    model,
    tokenized_chord_seqs,       
    prefix_type="random_short",
    prefix_len=4
)
gens["B_single"] = sample_diverse(
    model,
    tokenized_chord_seqs,
    prefix_type="single"
)
gens["C_fixed4"] = sample_diverse(
    model,
    tokenized_chord_seqs,
    prefix_type="fixed",
    fixed_prefix=tokenized_chord_seqs[0][:4]  # first 4 chords of the first chorale
)

# Now map each generated chord-index sequence back to actual 4-tuples
chord_sequences = {
    name: [idx_to_chord[idx] for idx in seq]
    for name, seq in gens.items()
}

# 3 different generated strategies
generated_chords   = chord_sequences["A_random4"]
generated_chords2  = chord_sequences["B_single"]
generated_chords3  = chord_sequences["C_fixed4"]

# %% [markdown]
# ### Save original & generated as MIDI and convert to WAV for listening

# %%
# Helper function to write a list of MIDI pitches to a .mid file
# with all four voice‐notes in parallel at each time step.
def save_four_voice_midi(chord_seq, filename="polyphonic_output.mid", note_duration=0.5):
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0)  # single piano instrument
    current_time = 0.0

    for item in chord_seq:
        if isinstance(item, tuple):
            chord_tuple = item
        else:
            # assume 'item' is an index
            chord_tuple = idx_to_chord[item]

        for pitch in chord_tuple:
            if pitch != -1:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=current_time,
                    end=current_time + note_duration
                )
                instr.notes.append(note)

        current_time += note_duration

    pm.instruments.append(instr)
    pm.write(filename)

# %%
# Convert chord-indices → write a four-voice MIDI
save_four_voice_midi(generated_chords,  filename="generated_chords_A.mid")
save_four_voice_midi(generated_chords2, filename="generated_chords_B.mid")
save_four_voice_midi(generated_chords3, filename="generated_chords_C.mid")

# Convert original 4-voice (first 64 chords) and each generated version to WAV
save_four_voice_midi(tokenized_chord_seqs[0][:64], filename="original_chords.mid")
fs = FluidSynth("FluidR3_GM.sf2")
fs.midi_to_audio("original_chords.mid",      "original_chords.wav")
fs.midi_to_audio("generated_chords_A.mid",   "generated_A.wav")
fs.midi_to_audio("generated_chords_B.mid",   "generated_B.wav")
fs.midi_to_audio("generated_chords_C.mid",   "generated_C.wav")

# Play back and display audio in notebook
print("🎹 Original four-voice:")
display(Audio("original_chords.wav"))

print("🎹 Generated (A: random4):")
display(Audio("generated_A.wav"))

print("🎹 Generated (B: single-chord cold start):")
display(Audio("generated_B.wav"))

print("🎹 Generated (C: fixed4 prefix):")
display(Audio("generated_C.wav"))

# %% [markdown]
# ### Extract pitches back from files for plotting

# %%
def extract_midi_pitches(midi_file):
    """Load MIDI and return a list of (start_time, pitch) sorted by time."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    pitches = []
    for note in midi_data.instruments[0].notes:
        pitches.append((note.start, note.pitch))
    # Sort by start time and return pitch only
    pitches.sort()
    
    return [p[1] for p in pitches]

def extract_pitch_from_wav(wav_file):
    """Use librosa’s pitch tracker to extract a MIDI‐rounded pitch contour."""
    y, sr = librosa.load(wav_file)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_track = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        pitch_track.append(pitch if pitch > 0 else np.nan)
    # Convert Hz to MIDI pitch (round)
    midi_pitches = [int(round(librosa.hz_to_midi(p))) if not np.isnan(p) else np.nan for p in pitch_track]
    return midi_pitches

def plot_waveform(wav_file, ax, title="Waveform"):
    """Load a WAV file and plot its waveform on the given Axes."""
    y, sr = librosa.load(wav_file, sr=None)  # preserve native sample rate
    times = np.arange(len(y)) / sr
    ax.plot(times, y, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_spectrogram(wav_file, ax, title="Spectrogram"):
    """Load a WAV file and plot its log-power spectrogram on the given Axes."""
    y, sr = librosa.load(wav_file, sr=None)
    # Compute short-time Fourier transform
    D = librosa.stft(y, n_fft=1024, hop_length=512)
    # Convert to decibels
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='hz',
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    # Add a colorbar on the right of this axis
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

# %% [markdown]
# ### Compare Original vs Generated A (random4): Waveform & Spectrogram

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

# Original waveform (top-left)
plot_waveform("original.wav",     axes[0, 0], title="Original Waveform")
# Generated A waveform (top-right)
plot_waveform("generated_A.wav",   axes[0, 1], title="Generated A Waveform")

# Original spectrogram (bottom-left)
plot_spectrogram("original.wav",   axes[1, 0], title="Original Spectrogram")
# Generated A spectrogram (bottom-right)
plot_spectrogram("generated_A.wav", axes[1, 1], title="Generated A Spectrogram")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Compare Original vs Generated B (single‐note): Waveform & Spectrogram

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

# Original waveform (top-left)
plot_waveform("original.wav",      axes[0, 0], title="Original Waveform")
# Generated B waveform (top-right)
plot_waveform("generated_B.wav",    axes[0, 1], title="Generated B Waveform")

# Original spectrogram (bottom-left)
plot_spectrogram("original.wav",    axes[1, 0], title="Original Spectrogram")
# Generated B spectrogram (bottom-right)
plot_spectrogram("generated_B.wav",  axes[1, 1], title="Generated B Spectrogram")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Compare Original vs Generated C (fixed4‐prefix): Waveform & Spectrogram

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

# Original waveform (top-left)
plot_waveform("original.wav",      axes[0, 0], title="Original Waveform")
# Generated C waveform (top-right)
plot_waveform("generated_C.wav",    axes[0, 1], title="Generated C Waveform")

# Original spectrogram (bottom-left)
plot_spectrogram("original.wav",    axes[1, 0], title="Original Spectrogram")
# Generated C spectrogram (bottom-right)
plot_spectrogram("generated_C.wav",  axes[1, 1], title="Generated C Spectrogram")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Evaluation (Four‐Voice Chord Model)

# %% [markdown]
# ### 1. Chord‐Level Cross‐Entropy Loss and Perplexity
# 
# We evaluate the four‐voice model on held‐out splits:
# 
# 1. Preprocess **validation** and **test** chorales into chord‐tuples.
# 2. Tokenize each chord tuple with `chord_to_idx`.
# 3. Build `ChordSequenceDataset` / `DataLoader` for each split.
# 4. Use our `evaluate()` helper to compute chord‐level average cross‐entropy loss and perplexity.
# 5. Print validation & test metrics.

# %%
# 1. Preprocess valid & test splits into four‐voice chord sequences
def extract_chord_seqs(chorales, min_len=10):
    seqs = []
    for chorale in chorales:
        chord_list = []
        for chord in chorale:
            # Chord is a 4‐tuple or -1; keep only valid 4‐tuples
            if isinstance(chord, (list, tuple)) and len(chord) == 4:
                chord_tuple = (int(chord[0]), int(chord[1]), int(chord[2]), int(chord[3]))
                if chord_tuple != (-1, -1, -1, -1):
                    chord_list.append(chord_tuple)
        if len(chord_list) > min_len:
            seqs.append(chord_list)
    return seqs

valid_chord_seqs = extract_chord_seqs(data["valid"])
test_chord_seqs  = extract_chord_seqs(data["test"])

# %%
# 2. Tokenize using chord_to_idx (drops unseen chords)
valid_chord_tokens = [
    [chord_to_idx[ch] for ch in seq if ch in chord_to_idx]
    for seq in valid_chord_seqs
]
test_chord_tokens = [
    [chord_to_idx[ch] for ch in seq if ch in chord_to_idx]
    for seq in test_chord_seqs
]

# Build chord‐level datasets & dataloaders
val_dataset_ch  = ChordSequenceDataset(valid_chord_tokens, seq_len=seq_len)
test_dataset_ch = ChordSequenceDataset(test_chord_tokens,  seq_len=seq_len)
val_loader_ch   = DataLoader(val_dataset_ch,  batch_size=batch_size)
test_loader_ch  = DataLoader(test_dataset_ch, batch_size=batch_size)

print(f"Validation chord samples: {len(val_dataset_ch)},  Test chord samples: {len(test_dataset_ch)}")

# %%
# 3. Evaluation helper
def evaluate(model, loader, vocab_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (batch, seq_len, vocab_size)
            
            loss = loss_fn(
                logits.view(-1, vocab_size),
                yb.view(-1)
            )
            total_loss += loss.item()
            total_tokens += yb.numel()
    
    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    return avg_loss, ppl

# 4. Run chord‐level evaluation
val_loss_ch, val_ppl_ch   = evaluate(model, val_loader_ch,  vocab_size)
test_loss_ch, test_ppl_ch = evaluate(model, test_loader_ch, vocab_size)

print(f"Validation (chords)  —  Loss: {val_loss_ch:.4f},  Perplexity: {val_ppl_ch:.2f}")
print(f"Test (chords)        —  Loss: {test_loss_ch:.4f},  Perplexity: {test_ppl_ch:.2f}")

# %% [markdown]
# #### Interpretation (Chord Level)
# 
# - **Validation loss = 7.3067** (perplexity ≈ 1490.21)  
# - **Test loss = 7.2839** (perplexity ≈ 1456.60)  
# 
# A chord‐level perplexity of ~1457–1490 indicates the model is effectively choosing among ~1,450 equally likely four‐voice chord tokens at each step. The small gap between validation and test perplexities suggests the model generalizes reasonably well at the chord level, though absolute perplexity remains high (likely because the chord vocabulary is large).

# %% [markdown]
# ### 2. Voice‐Specific Pitch Statistics
# Next, we inspect each individual voice (soprano, alto, tenor, bass) to see whether the model captures their marginal pitch distributions. For each voice:
# 
# 1. Flatten all pitches of that voice in the test set.
# 2. Flatten all pitches of that voice in one generated sample (we’ll use “Generated A (random4)”).
# 3. Compute mean & standard deviation for both.
# 4. Plot histograms side by side (test vs generated).

# %%
# Flatten test set pitches for each of the 4 voices
# test_chord_tokens is a list of chord‐index sequences; convert back to chord_tuples
test_voice_pitches = {i: [] for i in range(4)}  # 0=soprano, 1=alto, 2=tenor, 3=bass

for seq in test_chord_tokens:
    for chord_idx in seq:
        chord_tuple = idx_to_chord[chord_idx]  # (soprano, alto, tenor, bass)
        for voice_idx in range(4):
            test_voice_pitches[voice_idx].append(chord_tuple[voice_idx])

# Flatten generated sample A (“random4”) pitches for each voice
gen_chord_ids_A = gens["A_random4"]  # list of chord‐indices
gen_voice_pitches_A = {i: [] for i in range(4)}

for chord_idx in gen_chord_ids_A:
    chord_tuple = idx_to_chord[chord_idx]
    for voice_idx in range(4):
        gen_voice_pitches_A[voice_idx].append(chord_tuple[voice_idx])

# Compute mean & std for each voice (test vs generated A)
for v in range(4):
    mean_test = np.mean(test_voice_pitches[v])
    std_test  = np.std(test_voice_pitches[v])
    mean_gen  = np.mean(gen_voice_pitches_A[v])
    std_gen   = np.std(gen_voice_pitches_A[v])
    voice_name = ["Soprano","Alto","Tenor","Bass"][v]
    print(f"{voice_name}:")
    print(f"  Test   mean = {mean_test:.2f}, std = {std_test:.2f}")
    print(f"  Gen A  mean = {mean_gen:.2f}, std = {std_gen:.2f}\n")

# %% [markdown]
# #### Interpretation (Voice Marginals)
# 
# - **Soprano**: Test (63.11 ± 5.36), Gen A (66.30 ± 4.60). The generated soprano is shifted ~3 semitones higher on average and slightly less variable.  
# - **Alto**: Test (71.99 ± 4.58), Gen A (75.64 ± 3.62). The generated alto is ~3.65 semitones higher.  
# - **Tenor**: Test (77.66 ± 4.59), Gen A (81.00 ± 3.28). The generated tenor is ~3.34 semitones higher.  
# - **Bass**: Test (83.13 ± 4.99), Gen A (85.67 ± 2.97). The generated bass is ~2.54 semitones higher.  
# 
# All four voices in the generated sample are biased toward higher pitches compared to the test set, and their variances are slightly reduced. This indicates the model has learned pitch ranges but drifts upward in all voices.

# %%
# Plot histograms (Test vs Generated A) for each voice
voice_labels = ["Soprano", "Alto", "Tenor", "Bass"]
fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharey=False)

for v in range(4):
    # Test histogram (left column)
    axes[v, 0].hist(
        test_voice_pitches[v],
        bins=range(min(test_voice_pitches[v]), max(test_voice_pitches[v]) + 2),
        color='blue',
        alpha=0.7
    )
    axes[v, 0].set_title(f"{voice_labels[v]} Test Pitch Dist.")
    axes[v, 0].set_xlabel("MIDI Pitch")
    axes[v, 0].set_ylabel("Count")
    axes[v, 0].grid(True, linestyle='--', alpha=0.5)

    # Generated A histogram (right column)
    axes[v, 1].hist(
        gen_voice_pitches_A[v],
        bins=range(min(gen_voice_pitches_A[v]), max(gen_voice_pitches_A[v]) + 2),
        color='orange',
        alpha=0.7
    )
    axes[v, 1].set_title(f"{voice_labels[v]} Gen A Pitch Dist.")
    axes[v, 1].set_xlabel("MIDI Pitch")
    axes[v, 1].set_ylabel("Count")
    axes[v, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Voice‐Transition (Bigram) Analysis
# To see if the model learns realistic voice‐step transitions, we compare bigram frequencies (consecutive‐note pairs) in the test set vs. the generated sample for each voice. We will:
# 
# 1. Build bigram counts for each voice in test set: count all pairs (pitch_t, pitch_{t+1}).
# 2. Build bigram counts for each voice in one generated sample (A).
# 3. Convert these counts to conditional distributions P(next_pitch | current_pitch) and measure KL divergence from test to generated, per voice.

# %%
def compute_bigram_probs(pitch_sequence):
    """Given a list of pitches, returns dict: { current_pitch: { next_pitch: prob } }."""
    counts = {}
    for (p1, p2) in zip(pitch_sequence[:-1], pitch_sequence[1:]):
        counts.setdefault(p1, collections.Counter())
        counts[p1][p2] += 1
    # Normalize to probabilities
    bigram_probs = {}
    for p1, counter in counts.items():
        total = sum(counter.values())
        bigram_probs[p1] = {p2: count / total for p2, count in counter.items()}
    return bigram_probs

def kl_divergence(p_dist, q_dist):
    """
    Compute KL(P || Q) where P and Q are dicts of { symbol: prob }.
    Missing symbols in Q receive a small epsilon probability.
    """
    epsilon = 1e-8
    kl = 0.0
    for symbol, p_val in p_dist.items():
        q_val = q_dist.get(symbol, epsilon)
        kl += p_val * math.log(p_val / q_val)
    return kl

# Build bigram distributions for each voice in test set
test_bigram = {}
for v in range(4):
    test_bigram[v] = compute_bigram_probs(test_voice_pitches[v])

# Build bigram distributions for each voice in generated A
gen_bigram_A = {}
for v in range(4):
    gen_bigram_A[v] = compute_bigram_probs(gen_voice_pitches_A[v])

# Compute average KL divergence across all current_pitch contexts, per voice
kl_results = {}
for v in range(4):
    kl_sum, count = 0.0, 0
    for p1, p_dist in test_bigram[v].items():
        q_dist = gen_bigram_A[v].get(p1, {})
        kl_sum += kl_divergence(p_dist, q_dist)
        count += 1
    kl_results[v] = kl_sum / count if count > 0 else float('nan')

# Print KL divergence for each voice
for v in range(4):
    print(f"{voice_labels[v]} KL divergence (Test || Gen A) = {kl_results[v]:.4f}")

# %% [markdown]
# #### Interpretation (Voice Bigram KL)
# 
# - Each voice’s KL divergence is around 12–14. This large value means the generated voice‐leading transitions differ substantially from the test distribution. In other words, the model’s step‐to‐step pitch choices for each voice don’t closely match the test chorales.

# %% [markdown]
# ### 4. Chord‐Transition (Bigram) Analysis
# Finally, we check whether the model’s predicted chord transitions (4‐voice bigrams) match those in the test set:
# 
# 1. Build test‐set bigram counts on chord‐indices.
# 2. Build generated sample A bigram counts on chord‐indices.
# 3. Compute KL divergence of chord‐transition distributions.

# %%
# Build bigram distributions on chord‐indices (test set)
test_chords_flat = [idx for seq in test_chord_tokens for idx in seq]
test_chord_bigram = compute_bigram_probs(test_chords_flat)

# Build bigram distributions on chord‐indices (generated A)
gen_chord_bigram_A = compute_bigram_probs(gen_chord_ids_A)

# Compute KL divergence for each chord context
kl_sum, count = 0.0, 0
for c1, p_dist in test_chord_bigram.items():
    q_dist = gen_chord_bigram_A.get(c1, {})
    kl_sum += kl_divergence(p_dist, q_dist)
    count += 1
kl_chord = kl_sum / count if count > 0 else float('nan')

print(f"Chord‐level KL divergence (Test || Gen A) = {kl_chord:.4f}")

# %% [markdown]
# #### Interpretation (Chord Bigram KL Divergence)
# 
# - A chord‐level KL of ~17.65 is very large, indicating the generated chord‐to‐chord transitions deviate greatly from the test distribution. The model is not capturing four‐voice harmonic progressions as faithfully as expected.

# %% [markdown]
# ## Overall Evaluation Summary
# 
# 1. **Chord‐Level Perplexity**:  
#    - Validation ≈ 1490.21, Test ≈ 1456.60  
#    This high perplexity reflects the large chord vocabulary (~1,500 distinct chords) and shows the model still struggles to narrow down its predictions reliably.
# 
# 2. **Voice Marginals**:  
#    - Generated voices (Gen A) skew higher in pitch (≈ +2–4 semitones) relative to the test distributions, with slightly lower variance.
# 
# 3. **Voice Bigram KL Divergences** (Test‖Gen A):  
#    - Soprano ≈ 13.97  
#    - Alto    ≈ 12.36  
#    - Tenor   ≈ 13.44  
#    - Bass    ≈ 13.40  
#    These large KL values indicate the model’s stepwise pitch transitions do not match the test set well for any voice.
# 
# 4. **Chord Bigram KL Divergence**:  
#    - ≈ 17.65  
#    The four‐voice harmonic transitions in generated chords differ significantly from those in the held‐out chorales.
# 
# Taken together, the evaluation shows that, although the model has learned to generate plausible chord shapes (it still produces four‐voice chords), its distributions—both marginal and sequential—drift noticeably from the test data. There is ample room for improvement (e.g., more training data, larger/deeper architectures, attention to balancing pitch ranges).

# %% [markdown]
# 


