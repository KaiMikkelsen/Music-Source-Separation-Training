from moisesdb.dataset import MoisesDB

db = MoisesDB(data_path="/home/kaim/scratch/MOISESDB/moisesdb/",sample_rate=44100)

from moisesdb.dataset import MoisesDB
from moisesdb.defaults import mix_4_stems


for track in db:
    print(track.name)
    stems = track.mix_stems(mix_4_stems)
    for stem_name, samples in stems.items():
        print(stem_name, samples.shape)
