# Transcription

## Setup
### Deepspeech
Download acoustic deepspeech-0.7.3-models from https://github.com/mozilla/DeepSpeech/releases/tag/v0.7.3

In the current directory run:
```
mkdir deepspeech-0.7.3-models
cd deepspeech-0.7.3-models
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.3/deepspeech-0.7.3-models.pbmm
```
### Wav2Letter



# Results

To view the sample transcription, run `process_pipeline.py`. This will transcript with deepspeech or wav2letter. 

To view a streaming example, check out `transcribe.py`

This will allow testing for Deepspeech or Wav2Letter transcription with and without VAD. 
VAD is used to break up audio samples into chunks

Deepspeech - it is mostly accurate but struggles with faster sentences or non ideal situations. Performs much better with VAD

Wav2Letter - most accurate but more painful to setup/run. It is difficult to tell if performance is better with or without VAD

LibreASR - fast but struggled to get accurate phrases. It also has a smaller community than the other projects

vosk - seems to struggle with non ideal voice and perform worse than deepspeech. 