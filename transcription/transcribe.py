import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import requests
import json
import click
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disables Tf warnings
log = logging.getLogger(__name__)
DEFAULT_SAMPLE_RATE = 16000


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            # pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech
        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        log.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


def default_save_audio_path_func(vad_audio, wav_data, wav_dir="save_audio_old/"):
    vad_audio.write_wav(os.path.join(wav_dir, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
    os.makedirs(wav_dir, exist_ok=True)


def default_input_detected_callback(text):
    if not text:
        return
    print("Recognized: %s" % text)
    with open("test.txt", 'a') as f:
        f.write(text + "\n")
    # if text:
    #     r = requests.get("http://localhost:8000/classify/"+text)
    #     print("Json Response", json.dumps(r.json(), indent = 4))
    #     r = requests.get("http://localhost:8000/classify_record/"+text)
    #     print("Json Response", json.dumps(r.json(), indent = 4))


def handle_audio(save_audio_path_func, input_detected_callback,
                 deepspeech_model_dir='deepspeech-0.7.3-models',
                 verbose=True,
                 vad_aggressiveness=1,
                 input_file=None):
    model_path = os.path.join(deepspeech_model_dir, 'deepspeech-0.7.3-models.pbmm')
    log.info('Initializing model...')
    model = deepspeech.Model(model_path)
    # Starting audio with VAD
    vad_audio = VADAudio(aggressiveness=vad_aggressiveness,
                         device=None,
                         input_rate=DEFAULT_SAMPLE_RATE,
                         file=input_file)
    log.info("Listening (ctrl-C to exit)...")
    spinner = None
    if verbose:
        spinner = Halo(spinner='line')
    frames = vad_audio.vad_collector()
    stream_context = model.createStream()
    wav_data = bytearray()

    for frame in frames:
        if frame is not None:

            if spinner: spinner.start()

            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))

            if True: wav_data.extend(frame)

        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            if True:
                vad_audio.write_wav(
                    os.path.join('save_audio_old', datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")),
                    wav_data)
                wav_data = bytearray()
            text = stream_context.finishStream()
            text = text.strip()
            default_input_detected_callback(text)
            # print("Recognized: %s" % text)
            stream_context = model.createStream()


@click.command()
@click.option('-v', default=True, help='Not verbose')
@click.option('--vad_agg', default=1,
              help='Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive.')
@click.option("-f", help="Read from .wav file instead of microphone")
def cli(vad_agg, f, v=True):
    """Simple program that greets NAME for a total of COUNT times."""
    handle_audio(default_save_audio_path_func,
                 default_input_detected_callback,
                 verbose=v,
                 vad_aggressiveness=vad_agg)


if __name__ == '__main__':
    cli()
