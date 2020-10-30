import os
import re
import shutil
import subprocess
import time
import wave
from abc import ABC, abstractmethod
from typing import List, AnyStr, Optional
import click
import humanfriendly as humanfriendly
import psutil
from memory_profiler import profile
import deepspeech

import numpy as np
import glob

from webrtcvad_example import create_chunks


class Transcription(ABC):
    model_name = "default"
    use_vad = False

    @abstractmethod
    def transcript(self, wave_path: str) -> str:
        pass

    def transcript_all(self, wave_paths: List[str], recording_path: Optional[str], save_results_dir: str) -> List[str]:
        results = []
        for wave_path in wave_paths:
            results.append(self.transcript(wave_path))
        if save_results_dir is not None:
            f_name, f_ext = os.path.splitext(os.path.basename(recording_path))
            if not os.path.exists(save_results_dir):
                os.mkdir(save_results_dir)
            full_path = os.path.join(save_results_dir, f_name + self.model_name + ".txt")
            print("Writing transcript to ", full_path)
            with open(full_path, 'w') as f:
                f.write("\n".join(results))
        return results


class DeepspeechTranscription(Transcription):
    model_name = "DeepSpeech"
    use_vad = True

    def __init__(self, model_dir="deepspeech-0.7.3-models"):
        super().__init__()
        self.model_path = os.path.join(model_dir, 'deepspeech-0.7.3-models.pbmm')
        self.model = deepspeech.Model(self.model_path)

    # @profile
    def transcript(self, wave_path: str) -> str:
        # We have the code for this. We split wav file into frames
        print(f"Running transcript on {wave_path}")
        start = time.time()
        w = wave.open(wave_path, 'r')
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        if rate != self.model.sampleRate():
            print("WARNING!: The sample rate of the wav is different than model sample rate")
        data16 = np.frombuffer(buffer, dtype=np.int16)
        text = self.model.stt(data16)
        end = time.time()
        print("Time Elapsed:", humanfriendly.format_timespan(end - start))
        return text


class Wav2LetterTranscription(Transcription):
    model_name = "Wav2Letter"
    use_vad = False

    def profile_docker_mem_usage(self, proc):
        SLICE_IN_SECONDS = 3
        while proc.poll() is None:
            p = psutil.Process(proc.pid)
            mem_status = "RSS {},  VMS: {}".format(humanfriendly.format_size(p.memory_info().rss),
                                                   humanfriendly.format_size(p.memory_info().vms))
            time.sleep(SLICE_IN_SECONDS)
            print(mem_status)

    def transcript(self, wave_path: str) -> str:
        # process wave frame by frame at a sample rate
        # run through the docker
        # parse the docker output
        print(f"Running transcript on {wave_path}")
        volume_mount_dir = "$PWD"
        wav2_letter_model_path = 'wav2lettermodel'
        # Handle synchronously on purpose

        cmd = f"""docker run --rm -v {volume_mount_dir}:/root/host/ -i --ipc=host -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c 'cat /root/host/{wave_path} | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/{wav2_letter_model_path}'"""
        start = time.time()

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

        self.profile_docker_mem_usage(proc)
        end = time.time()
        print("Time Elapsed:", humanfriendly.format_timespan(end - start))

        read_lines = proc.stdout.readlines()
        # parsing can also be done via | awk -F, '{print $3}' but didn't want to loose timing information if needed in the future
        result = []
        for line in read_lines:
            if match := re.search(r'\d+,\d+,(.*)', line.decode("utf-8")):
                result.append(match.group(1))

        return '\n'.join(result)


class VadDetector:
    def chunk_recording(self, wave_path: str, chunk_folder_name: Optional[str]) -> List[str]:
        if chunk_folder_name is None:
            chunk_folder_name = wave_path + "_chunks"
        if os.path.exists(chunk_folder_name):
            shutil.rmtree(chunk_folder_name)
        os.mkdir(chunk_folder_name)
        try:
            segments, full_paths = create_chunks(vad_agg=1, wave_path=wave_path, folder_name=chunk_folder_name)
        except Exception as e:
            print("Warning: Invalid wav file at ", wave_path)
            print(e)
            return []
        # added segment since its possible to direct pass it through
        return full_paths


def find_wav_files(dir_path: str) -> List[str]:
    print(f"Searching ", os.path.join(dir_path, "*.wav"))
    wav_files = []
    for file in glob.glob(os.path.join(dir_path, "*.wav"), recursive=False):
        wav_files.append(file)
    return wav_files

def process_pipeline(recordings_folder, transcription_folder, transcriptionModel: Transcription):
    for file in find_wav_files(recordings_folder):
        recording_name = file
        if transcriptionModel.use_vad:
            transcriptions = VadDetector().chunk_recording(recording_name, None)
        else:
            transcriptions = [recording_name]
        result = transcriptionModel.transcript_all(transcriptions, recording_name, transcription_folder)
        print("\n".join(result))

# Finish everything
# -> apply noise filter on each chunk
# -> Integrate shopping skill (next week)

# Non Coding Tasks
# Collect a sample of audio clips from meatwad
# Record more samples of audio clips saying the shopping sentences and save to drive

@click.command()
@click.option('-recordings_folder')
@click.option('-transcription_folder')
def cli(recordings_folder, transcription_folder):
    """Simple program that greets NAME for a total of COUNT times."""
    process_pipeline(recordings_folder, transcription_folder, Wav2LetterTranscription())


if __name__ == '__main__':
    recordings_folder = "samples/"
    transcript_folder = "samples/transcripts/"
    process_pipeline(recordings_folder, transcript_folder, Wav2LetterTranscription())

# sudo docker run --rm -v $PWD:/root/host/ -it --ipc=host --name w2l -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c "cat /root/host/samples/recording1.wav | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/wav2lettermodel"
# awk -F, '{print $3}'
