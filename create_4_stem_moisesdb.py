import os
import soundfile as sf
from moisesdb.dataset import MoisesDB
from moisesdb.defaults import mix_4_stems

# Input and Output directories
input_base_dir = '/home/kaim/scratch/MOISESDB/moisesdb/'
output_base_dir = '/home/kaim/scratch/moisesdb-4stem/'

# Initialize MoisesDB
db = MoisesDB(
    data_path=input_base_dir,
    sample_rate=44100
)

# Process only the first 3 tracks in the dataset
for i, track in enumerate(db):
    
    # Get the mixed stems
    stems = track.mix_stems(mix_4_stems)
    
    # Create a folder for each song
    song_dir = os.path.join(output_base_dir, track.name)
    os.makedirs(song_dir, exist_ok=True)
    
    # Save individual stems
    for stem_name, audio in stems.items():
        if audio is not None:
            stem_file = os.path.join(song_dir, f"{stem_name}.wav")
            print(f"Saving {stem_name} to {stem_file}")
            sf.write(stem_file, audio.T, 44100)
    
    # Save the mixture
    mixture = track.audio  # This is the full mixture
    mixture_file = os.path.join(song_dir, "mixture.wav")
    print(f"Saving mixture to {mixture_file}")
    sf.write(mixture_file, mixture.T, 44100)

    print(f"Finished processing {track.name}")
