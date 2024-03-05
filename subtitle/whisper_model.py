from typing import Literal, Union, List, Any, TypedDict
from time import time
import logging


class WhisperModel:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.device = None

    # 加载whisper模型，默认使用small模型
    def load(self, model_name: Literal["tiny", "base", "small", "medium", "large", "large-v2"] = "small",
             device: Union[Literal["cpu", "cuda"], None] = None):
        self.device = device
        import whisper
        # 默认model保存在.cache文件夹下
        self.whisper_model = whisper.load_model(model_name, device)

    # 将音频信息转化为文本信息
    def transcribe(self, audio, lang):
        tic = time.time()

        res = self.whisper_model.transcribe(
            audio,
            task="transcribe",
            language=lang,
            verbose=True,
            word_timestamps=True
        )

        logging.info(f"Done transcription in {time.time() - tic:.1f} sec")
        return res

    # 将音频信息转化为文本信息
    def translate(self, audio, lang):
        tic = time.time()

        res = self.whisper_model.transcribe(
            audio,
            task="translate",
            language=lang,
            verbose=True,
            word_timestamps=True
        )

        logging.info(f"Done translate in {time.time() - tic:.1f} sec")
        return res
