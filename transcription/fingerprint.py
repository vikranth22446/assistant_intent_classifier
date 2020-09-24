from dejavu.dejavu import Dejavu

config = {
     "database": {
         "host": "127.0.0.1",
         "user": "root",
         "database": 'dejavu',
     }
 }

djv = Dejavu(config)

djv.fingerprint_directory("dejavu/mp3", [".mp3"], 3)