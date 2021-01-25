import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # cleanup TF error message items

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

from tensorflow.python.training.tracking import base

from .webrtcvad_example import create_chunks
basedir = os.path.abspath(os.path.dirname(__file__))


class Transcription(ABC):
    model_name = "default"

    def __init__(self):
        self.use_vad = False

    # @abstractmethod
    def transcript(self, wave_path: str) -> str:
        pass

    def transcript_all(
        self,
        wave_paths: List[str],
        recording_path: Optional[str],
        save_results_dir: str,
    ) -> List[str]:
        results = []
        for wave_path in wave_paths:
            results.append(self.transcript(wave_path))
        if save_results_dir is not None:
            f_name, f_ext = os.path.splitext(os.path.basename(recording_path))
            uses_vad = "_with_vad" if self.use_vad else ""
            save_results_dir = os.path.join(
                save_results_dir, self.model_name + uses_vad
            )
            if not os.path.exists(save_results_dir):
                os.makedirs(save_results_dir)
            full_path = os.path.join(save_results_dir, f_name + ".txt")
            print("Writing transcript to ", full_path)
            with open(full_path, "w") as f:
                f.write("\n".join(results))
        return results


class DeepspeechTranscription(Transcription):
    model_name = "DeepSpeech"

    def __init__(self, model_dir="deepspeech-0.7.3-models", use_vad=True):
        super().__init__()
        self.use_vad = use_vad
        self.model_path = os.path.join(basedir, model_dir, "deepspeech-0.7.3-models.pbmm")
        self.model = deepspeech.Model(self.model_path)

    # @profile
    def transcript(self, wave_path: str) -> str:
        # We have the code for this. We split wav file into frames
        print(f"Running transcript on {wave_path}")
        start = time.time()
        w = wave.open(wave_path, "r")
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        if rate != self.model.sampleRate():
            print(rate)
            print(
                "WARNING!: The sample rate of the wav is different than model sample rate"
            )
        data16 = np.frombuffer(buffer, dtype=np.int16)
        text = self.model.stt(data16)
        end = time.time()
        print("Time Elapsed:", humanfriendly.format_timespan(end - start))
        return text


class Wav2LetterTranscription(Transcription):
    model_name = "Wav2Letter"

    def __init__(self, use_vad=False):
        super().__init__()
        self.use_vad = use_vad

    def profile_docker_mem_usage(self, proc):
        SLICE_IN_SECONDS = 3
        while proc.poll() is None:
            p = psutil.Process(proc.pid)
            # mem_status = "Mem used: {}".format(humanfriendly.format(p.memory_percent()))
            time.sleep(SLICE_IN_SECONDS)
            # print(mem_status)

    def transcript(self, wave_path: str) -> str:
        """
        # process wave frame by frame at a sample rate
        # run through the docker
        # parse the docker output
        """
        print(f"Running transcript on {wave_path} with vad: {self.use_vad}")
        # volume_mount_dir = "$PWD"
        volume_mount_dir = basedir
        wav2_letter_model_path = "wav2lettermodel"
        # Handle synchronously on purpose
        
        cmd = f"""docker run --rm -v {volume_mount_dir}:/root/host/ -i --ipc=host -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c 'cat /root/host/{wave_path} | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/{wav2_letter_model_path}'"""
        start = time.time()

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True
        )
        self.profile_docker_mem_usage(proc)
        read_lines = proc.stdout.readlines()

        end = time.time()
        print("Time Elapsed:", humanfriendly.format_timespan(end - start))
        # parsing can also be done via | awk -F, '{print $3}' but didn't want to loose timing information if needed in the future
        result = []
        for line in read_lines:
            if match := re.search(r"\d+,\d+,(.*)", line.decode("utf-8")):
                result.append(match.group(1))
        print("\n".join(result))
        return "\n".join(result)



class VadDetector:
    def chunk_recording(
        self, wave_path: str, chunk_folder_name: Optional[str]
    ) -> List[str]:
        if chunk_folder_name is None:
            chunk_folder_name = wave_path + "_chunks"
        if os.path.exists(chunk_folder_name):
            shutil.rmtree(chunk_folder_name)
        os.mkdir(chunk_folder_name)
        try:
            segments, full_paths = create_chunks(
                vad_agg=1, wave_path=wave_path, folder_name=chunk_folder_name
            )
        except Exception as e:
            print("Warning: Invalid wav file at ", wave_path)
            print(e)
            return []
        print("chunk recording")
        # added segment since its possible to direct pass it through
        return full_paths


def find_wav_and_ts_files(dir_path: str) -> List[str]:
    print(f"Searching ", os.path.join(dir_path, "*.ts/.wav"))
    wav_files = []
    for file in glob.glob(os.path.join(dir_path, "*.wav"), recursive=False):
        wav_files.append(file)
    for file in glob.glob(os.path.join(dir_path, "*.ts"), recursive=False):
        wav_files.append(file)
    return wav_files


def convert_channel_sample_rate(file_name, channel_size=1, sample_rate=16000):
    f_name, f_ext = os.path.splitext(file_name)
    if f_ext != ".wav":
        new_path = f_name + ".wav"
        if not os.path.exists(new_path):
            proc = subprocess.Popen(
                f"ffmpeg -y -i {file_name} -vn {new_path}",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                shell=True,
            ).wait()
        final_path = f_name + "_channel_1_16000" + ".wav"
        if not os.path.exists(final_path):
            proc = subprocess.Popen(
                f"sox {new_path} -c {channel_size} -r {sample_rate} {final_path}",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                shell=True,
            ).wait()
        return final_path

    w = wave.open(file_name, "r")
    rate = w.getframerate()
    wave_num_channels = w.getnchannels()
    print("Current rate", rate, "Num channels", wave_num_channels)
    if rate != sample_rate or wave_num_channels != channel_size:
        new_path = f_name + "_channel_1_16000" + ".wav"
        proc = subprocess.Popen(
            f"sox {file_name} -c {channel_size} -r {sample_rate} {new_path}",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
        )
        proc.wait()
        file_name = new_path
    return file_name

def get_transcript_full_path(recording_name: str, transcription_folder: str, transcription_model: Transcription):
    # TODO remove duplication by combining this with other process
    uses_vad = "_with_vad" if transcription_model.use_vad else ""
    f_name, f_ext = os.path.splitext(os.path.basename(recording_name))
    save_results_dir = os.path.join(transcription_folder, transcription_model.model_name + uses_vad)
    full_path = os.path.join(save_results_dir, f_name + ".txt")
    return full_path

def process_pipeline_entry(
    recording_name: str, transcription_folder: str, transcription_model: Transcription
):
    recording_name = convert_channel_sample_rate(recording_name)
    if transcription_model.use_vad:
        transcriptions = VadDetector().chunk_recording(recording_name, None)
    else:
        transcriptions = [recording_name]
    result = transcription_model.transcript_all(
        transcriptions, recording_name, transcription_folder
    )
    print("\n".join(result))
    return get_transcript_full_path(recording_name, transcription_folder, transcription_model)

def process_pipeline(
    recordings_folder, transcription_folder, transcription_model: Transcription
):
    for file in find_wav_and_ts_files(recordings_folder):
        process_pipeline_entry(file, transcription_folder, transcription_model)


# Finish everything
# -> apply noise filter on each chunk
# -> Integrate shopping skill (next week)

@click.command()
@click.option("-recordings_folder")
@click.option("-transcription_folder")
def cli(recordings_folder, transcription_folder):
    """Simple program that greets NAME for a total of COUNT times."""
    process_pipeline(
        recordings_folder, transcription_folder, Wav2LetterTranscription(use_vad=False)
    )


if __name__ == "__main__":
    recordings_folder = "wetransfer-b86082/"
    # process_pipeline_entry("test.wav", "test_transcript", Wav2LetterTranscription())
    process_pipeline_entry(
        "save_audio/savewav_2020-09-09_19-04-59_750112.wav",
        "test_transcript",
        DeepspeechTranscription(),
    )
