import threading
import time
from threading import  Thread

thnum=0
f_flag = 0

class MyThread(threading.Thread):
    def run(self):
        for i in range(1000):
            global thnum
            thnum += 1
        global f_flag
        f_flag=1
        print(thnum)


def test():
    global thnum
    while True:
        if f_flag != 0:
            for i in range(1000):
                thnum += 1
            break

    print(thnum)


if __name__=="__main__":
    t = MyThread()
    t.start()
