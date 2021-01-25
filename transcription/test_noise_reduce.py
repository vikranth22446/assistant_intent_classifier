import librosa
from scipy.io import wavfile
import noisereduce as nr

# # load data
# from pydub.effects import normalize
# from scipy.io import wavfile
# import matplotlib.pyplot as plt
import numpy as np

#
# import numpy as np
rate, data = wavfile.read(
    "samples_v2_transcripts/mic_1604597998_16000_s16le_channel_0.wav"
)
# # select section of data that is noise
data = data / 32768
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=data, verbose=False)
reduced_total_noisy_frames_npint16 = np.ndarray.astype(
    np.iinfo(np.int16).max * reduced_noise, dtype=np.int16
)
# # reduced_noise = np.array(reduced_noise)

wavfile.write("test.wav", rate, reduced_total_noisy_frames_npint16)

# # sox one.wav -r 16000 sr.wav
# # sox stereo.wav -c 1 mono.wav
# from pydub import AudioSegment
#
# data = AudioSegment.from_wav("samples_v2_transcripts/mic_1604597998_16000_s16le_channel_0.wav")
# # boost volume by 5 dB
# channels = data.split_to_mono()
# data = channels[0]

# data = normalize(data)
# data = data + 8
# spec, frequencies = data.filter_bank(nfilters=5)


# def visualize(spect, frequencies, title=""):
#     # Visualize the result of calling seg.filter_bank() for any number of filters
#     i = 0
#     for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
#         plt.subplot(spect.shape[0], 1, index + 1)
#         if i == 0:
#             plt.title(title)
#             i += 1
#         plt.ylabel("{0:.0f}".format(freq))
#         plt.plot(row)
#     plt.show()
#
#
# visualize(spec, frequencies)
#
# data.export("louder_song.wav", format='wav')
import numpy as np
