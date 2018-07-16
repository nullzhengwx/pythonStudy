import os
import random

from multiprocessing import Process, Queue

import time
from multiprocessing.pool import Pool

import multiprocessing


def fork_test():
    """
    fork方法是调用一次，返回两次，原因在于操作系统将当前进程（父进程）
    复制出一份进程（子进程），这两个进程几乎完全相同，于是fork方法分别在父进程
    和子进程中返回。子进程中永远返回0,父进程中返回的是子进程的ID。
    :return:
    """
    print('current Process (%s) start ...' % (os.getpid()))
    pid = os.fork()

    if pid < 0:
        print('error in fork')
    elif pid == 0:
        print('I am child process(%s) and my parent process is (%s)'
              % (os.getpid(), os.getppid()))
    else :
        print('I(%s) created a child process (%s).' % (os.getpid(), pid))

def run_proc(name):
    print('Chilk process %s (%s) Running ...' % (name, os.getpid()))

def process_demo():
    print('Parent process %s.' % os.getpid())
    for i in range(5):
        p = Process(target=run_proc, args=(str(i),))
        print('Process will start.')
        p.start()
    p.join()
    print('Process end.')

def run_task(name):
     print('Task %s (pid = %s) is running ..,' % (name, os.getpid()))
     time.sleep(random.random() * 3)
     print('Task %s end.' % name)

def process_pool_demo():
    """
    Pool对象调用join()方法会等待所有子进程执行完毕，调用join()之前必须先调用close(),
    调用close()之后就不能继续添加新的Process了。
    :return:
    """
    print('Current process %s.' % os.getpid())
    p = Pool(processes=3)
    for i in range(5):
        p.apply_async(run_task, args=(i, ))
    print('Waiting for all subprocesses done ...')
    p.close()
    p.join()
    print('All subprocesses done.')

def proc_write(q ,urls):
    """
    写数据进程执行的代码
    :param q:       Queue
    :param urls:
    :return:
    """
    print('Process(%s) is writing ...' % os.getpid())
    for url in urls:
        q.put(url)
        print('Put %s to queue ...' % url)
        time.sleep(random.random())

def proc_read(q):
    """
    读数据进程执行的代码
    :param q:       Queue
    :return:
    """
    print('Process(%s) is reading...' % os.getpid())
    while True:
        url = q.get(True)
        print('Get %s from queue.' % url)

def proc_queue_demo():
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    proc_writer1 = Process(target=proc_write, args=(q, ['url1', 'url2', 'url3']))
    proc_writer2 = Process(target=proc_write, args=(q, ['url4', 'url5', 'url6']))
    proc_reader = Process(target=proc_read, args=(q, ))

    # 启动子进程proc_writer， 写入：
    proc_writer1.start()
    proc_writer2.start()

    # 启动子进程proc_reader, 读取：
    proc_reader.start()

    # 等待proc_writer 结束：
    proc_writer1.join()
    proc_writer2.join()

    # proc_reader 进程里是死循环， 无法等待其结束，只能强行终止：
    proc_reader.terminate()

def proc_pipe_send(pipe, urls):
    for url in urls:
        print("Process(%s) send: %s" % (os.getpid(), url))
        pipe.send(url)
        time.sleep(random.random())

def proc_pipe_recv(pipe):
    while True:
        print("Process(%s) rev: %s" % (os.getpid(), pipe.recv()))
        time.sleep(random.random())

def proc_pipe_demo():
    """
    Pipe有个参数duplex，默认为True，就表示pipe是全双工模式。
    如果为false，第一个conn1只负责接受信息，第二个conn2只负责发送信息
    如果没有消息可接受，recv方法会一直阻塞，如果管道已经被关闭，那么recv方法会抛出EOFError
    :return:
    """
    pipe = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=proc_pipe_send,args=(pipe[0],
                                 ['url_' + str(i) for i in range(10)]))
    p2 = multiprocessing.Process(target=proc_pipe_recv,args=(pipe[1],))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == '__main__' :
    # fork_test()
    # process_demo()
    # process_pool_demo()
    # proc_queue_demo()
    proc_pipe_demo()