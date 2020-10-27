import os
from abc import ABC, abstractmethod 
from typing import List
import click

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

class VadDetector:
    def chunk_recording(self, wave_path: str, temp_folder_name: str) -> List[str]:
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

@click.command()
@click.option('-v', default=True, help='Not verbose')
@click.option('--vad_agg', default=1,
              help='Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive.')
@click.option("-f", help="Read from .wav file instead of microphone")
def cli(vad_agg, f, v=True):
    """Simple program that greets NAME for a total of COUNT times."""
    pass


if __name__ == '__main__':
    cli()
