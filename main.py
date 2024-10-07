from Control.controller import MyApp
from PyQt5.QtWidgets import QApplication
import sys
import threading

def program1():
    app = QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    sys.exit(app.exec_())

def program2():



# 创建三个线程，分别运行两个程序
thread1 = threading.Thread(target=program1)
thread2 = threading.Thread(target=program2)
thread3 = threading.Thread(target=program3)

# 启动3个线程
thread1.start()
thread2.start()
thread3.start()

# 等待3个线程执行完毕
thread1.join()
thread2.join()
thread3.join()


