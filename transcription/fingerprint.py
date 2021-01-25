from dejavu.dejavu import Dejavu
# Audio Fingerprinting based on FFT and Spectogram

# TODO add more audio samples. Setup SQLITE or similiar database to setup/run automatically

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "database": "dejavu",
    }
}

djv = Dejavu(config)
djv.fingerprint_directory("dejavu/mp3", [".mp3"], 3)
