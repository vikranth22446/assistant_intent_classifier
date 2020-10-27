import os
from abc import ABC, abstractmethod 
class Transcription(ABC):

    @abstractmethod
    def transcript(self, wave_path: str) -> str:
        pass

class DeepspeechTranscription(Transcription):

    def transcript(self, wave_path: str) -> str:
        # We have the code for this. We split wav file into frames
        pass

class Wav2LetterTranscription(Transcription):

    def transcript(self, wave_path: str) -> str:
        # process wave frame by frame at a sample rate
        # run through the docker
        # parse the docker output
        pass

# Coding Tasks
    # -> Read each wav file in folder
    # -> split into chunks 
    # -> transcription on each chunk 
    # -> save each transcription to file in folder

    # Finish everything
        # -> apply noise filter on each chunk
        # -> Integrate shopping skill (next week)

# Non Coding Tasks
    # Collect a sample of audio clips from meatwad
    # Record more samples of audio clips saying the shopping sentences and save to drive

# sudo docker run --rm -v $PWD:/root/host/ -it --ipc=host --name w2l -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c "cat /root/host/recording1.wav | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/wav2lettermodel"
