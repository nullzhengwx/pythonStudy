import random
import threading

import time

from gevent.pool import Pool


class myThread(threading.Thread):
    def __init__(self, name, urls):
        threading.Thread.__init__(self, name=name)
        self.urls = urls

    def run(self):
        print("Current %s is running ..." % threading.current_thread().name)
        for url in self.urls:
            print('%s ---->>> %s' % (threading.current_thread().name, url))
            time.sleep(random.random())
        print('%s ended.' % threading.current_thread().name)

def thread_run(urls):
    print("Current %s is running..." % threading.current_thread().name)
    for url in urls:
        print('%s ---->>> %s' % (threading.current_thread().name, url))
        time.sleep(random.random())
    print('%s ended.' % threading.current_thread().name)

def thread_demo() :
    print('%s is running...' % threading.current_thread().name)
    t1 = threading.Thread(target=thread_run, name='Thread_1',
                          args=(['url_1', 'url_2', 'url_3'],))
    t2 = threading.Thread(target=thread_run, name='Thread_2',
                          args=(['url_4', 'url_5', 'url_6'], ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print('%s ended.' % threading.current_thread().name)

mylock = threading.RLock()
num = 0
class myThread_lock(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self, name=name)

    def run(self):
        global num
        while True:
            mylock.acquire()
            print('%s locked, Number: %d' % (threading.current_thread().name, num))
            if num >= 4:
                mylock.release()
                print('%s released, Number: %d' % (threading.current_thread().name, num))
                break
            num += 1
            print('%s released, Number: %d' % (threading.current_thread().name, num))
            mylock.release()

def thread_lock_demo() :
    thread1 = myThread_lock('Thread_1')
    thread2 = myThread_lock('Thread_2')
    thread1.start()
    thread2.start()

from gevent import monkey; monkey.patch_all()
import gevent
import urllib.request

def gevent_run_task(url) :
    print('Visit --> %s' % url)
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        print('%d bytes received from %s' % (len(data), url))
    except Exception as e:
        print(e)

def gevent_demo() :
    urls = ['https://github.com/', 'https://www.python.org/', 'http://www.cnblogs.com/']
    greenlets = [gevent.spawn(gevent_run_task, url) for url in urls]
    gevent.joinall(greenlets)

def gevent_pool_run_task(url):
    print('Visit --> %s' % url)
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        print('%d bytes received from %s.' % (len(data), url))
    except Exception as e:
        print(e)
    return 'url:%s ---> finish' % url

def gevent_pool_demo() :
    pool = Pool(2)
    urls = ['https://gitbub.com/', 'https://www.python.org/', 'http://www.cnblogs.com/']
    results = pool.map(gevent_pool_run_task, urls)
    print(results)

if __name__ == '__main__' :
    # thread_demo()
    # thread_lock_demo()
    # gevent_demo()
    gevent_pool_demo()
