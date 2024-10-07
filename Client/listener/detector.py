"""
vdetector() used for vision detection and process
adetector() used for audio detection and check the speaker sound
"""
from datetime import datetime

import cv2
import pyaudio
import numpy as np
import logging
import speech_recognition as sr
from queue import Queue
import argparse
import torch

import os

import config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Detector:
    def __init__(self):
        # for video
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.ERROR('Cannot open camera')

        # for audio
        self.CHUNK = 1024
        RATE = 16000
        self.SILENCE_TIME = 2  # check the sounds end
        self.data_queue = Queue()

        p = pyaudio.PyAudio()
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def vdetector(self):
        """
        used for capture the video through local camera
        """
        # print("starting to capture video...")
        ret, frame = self.cap.read()
        if not ret:
            logging.ERROR("Can't receive frame (stream end?). Exiting ...")
        return frame

    def energy_detector(self):
        """
        used for capture the audio through local microphone
        """
        data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.int16)  # audio energy
        return data

    def audio_detector(self):
        print('Audio detector started')
        parser = argparse.ArgumentParser()
        parser.add_argument("--energy_threshold", default=1000,
                            help="Energy level for mic to detect.", type=int)
        parser.add_argument("--a_inputdir", default='audios/',
                            help="", type=str)
        parser.add_argument("--t_inputdir", default='texts/',
                            help="", type=str)
        parser.add_argument("--phrase_timeout", default=3,
                            help="How much empty space between recordings before we "
                                 "consider it a new line in the transcription.", type=float)
        args = parser.parse_args()

        # The last time a recording was retrieved from the queue.
        phrase_time = None
        # Thread safe Queue for passing data from the threaded recording callback.

        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        recorder = sr.Recognizer(language="zh-CN")
        recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False
        # Represents whether the energy level threshold(see recognizer_instance.energy_threshold) for sounds should be automatically adjusted based on the currently ambient noise level while listening.

        source = sr.Microphone()

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            text = recorder.recognize(audio)
            self.data_queue.put((text, audio.data))
            if not self.data_queue.empty():
                print('get sound')
                config.WITH_AUDIO = 1

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
        print("begin.\n")


if __name__ == '__main__':
    detector = Detector()
    detector.audio_detector()
