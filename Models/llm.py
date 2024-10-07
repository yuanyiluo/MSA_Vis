from PyQt5.QtCore import QThread, pyqtSignal
import ollama
import logging


class ModelThread(QThread):
    response_signal = pyqtSignal(str)

    def __init__(self, text: str, sentiment: str, mysql):
        super().__init__()
        self.text = text
        self.sentiment = sentiment
        self.mysql = mysql

    def run(self):
        history_text = self.mysql.check_table()

        modelfile = f'''
        FROM llama3.1
        SYSTEM 你是一个能理解他人情感的优质对话系统，当前你理解到我的情绪为{self.sentiment}。这是我们的历史对话{history_text}，其中user表示我，machine表示你。
        '''

        ollama.create(model='llama3.1', modelfile=modelfile)

        response = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": self.text,
                },
            ],
        )
        self.response_signal.emit(response["message"]["content"])
        self.mysql.insert_table(self.text, response["message"]["content"])


if __name__ == '__main__':
    llm = ModelThread()
    llm.run(0)
