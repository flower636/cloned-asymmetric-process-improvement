import threading
import time

import ctypes
import os

# NOTE: Be aware that getpid() is inside the POSIX standard, but gettid() is explicit of linux (not in the standard...) so...
# We have to do a syscall from python! That's fun!

libc = ctypes.CDLL('libc.so.6')
SYS_gettid = 186  # syscall number for gettid on x86_64
def gettid():
    return libc.syscall(SYS_gettid)

def thread_task(name):
    print(f"[{name}]({gettid()}) Running iteration...")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    print(f"[Main Process]({os.getpid()}) Creating child process...")
    t1 = threading.Thread(target=thread_task, args=("Thread one",))
    t1.start()
    t1.join()
