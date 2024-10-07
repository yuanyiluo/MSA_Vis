import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

import librosa

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

import config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_false',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--text_num", default=1,
                        help="How real time the recording is in seconds.", type=int)
    parser.add_argument("--a_inputdir", default='audios/',
                        help="", type=str)
    parser.add_argument("--t_inputdir", default='texts/',
                        help="", type=str)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    features = []
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    # Represents whether the energy level threshold(see recognizer_instance.energy_threshold) for sounds should be automatically adjusted based on the currently ambient noise level while listening.

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(device_index=index)
                    break
    else:
        source = sr.Microphone()

    # Load / Download model
    model = args.model
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    args.audio_index = 0

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        # args.audio_index = args.audio_index + 1
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        # data = audio.get_raw_data()
        data_queue.put(audio.data)
        # curr_audio_path = args.a_inputdir + f"audio{args.text_num}/"
        # if not os.path.exists(curr_audio_path):
        #     os.mkdir(curr_audio_path)
        # # 将录音数据写入.wav格式文件
        # with open(curr_audio_path + f"{args.audio_index}.wav", "wb") as f:
        #     # audio.get_wav_data()获得wav格式的音频二进制数据
        #     f.write(audio.get_wav_data())
        #     print(args.audio_index)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback)  # phrase_time_limit持续监测时间

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    print("begin.\n")

    while True:
        try:
            now = datetime.now()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():  # 当听不到声音后，开始transform
                print('start transform')
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                audio_data = data_queue.get()
                data_queue.queue.clear()

                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                if len(audio_data) % 2 != 0:
                    audio_data = audio_data[:-1]  # 截断最后一个字节
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # using librosa to get audio features (f0, mfcc, cqt)
                hop_length = 512  # hop_length smaller, seq_len larger
                f0 = librosa.feature.zero_crossing_rate(audio_np, hop_length=hop_length).T  # (seq_len, 1)
                mfcc = librosa.feature.mfcc(y=audio_np, sr=16000, hop_length=hop_length, htk=True).T  # (seq_len, 20)
                cqt = librosa.feature.chroma_cqt(y=audio_np, sr=16000, hop_length=hop_length).T  # (seq_len, 12)
                tem_feature = np.concatenate([f0, mfcc, cqt], axis=-1)
                features.append(tem_feature)
                if len(features) > 1:
                    feature = np.concatenate(features).mean(0).reshape(1, -1)
                else:
                    feature = tem_feature

        #             # Read the transcription.
        #             result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        #             text = result['text'].strip()
        #
        #             # If we detected a pause between recordings, add a new item to our transcription.
        #             # Otherwise edit the existing one.
        #             if phrase_complete:
        #                 transcription.append(text)
        #             else:
        #                 transcription[-1] = text
        #
        #             # Clear the console to reprint the updated transcription.
        #             os.system('cls' if os.name == 'nt' else 'clear')
        #             for line in transcription:
        #                 print(line)
        #             with open(args.t_inputdir + f'{args.text_num}.txt', 'w', encoding='utf-8') as f:
        #                 f.write(transcription[-1])  # 写入文本
        #                 args.text_num = args.text_num + 1
        #             # Flush stdout.
        #             print('', end='', flush=True)
        #
        #             # Infinite loops are bad for processors, must sleep.
        #             sleep(0.25)
        except KeyboardInterrupt:
            break
    #
    # print("\n\nTranscription:")
    # for line in transcription:
    #     print(line)


if __name__ == "__main__":
    main()
