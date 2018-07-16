# coding:utf-8
import socket

def client_demo() :
    # 初始化Socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接目标的IP和端口
    s.connect(('', 9999))

    # 接受消息
    print('-->>' + s.recv(1024).decode('utf-8'))

    # 发送消息
    s.send(b'Hello, I am a client')
    print('-->>' + s.recv(1024).decode('utf-8'))
    s.send(b'exit')

    # 关闭Socket
    s.close()

if __name__== '__main__' :
    client_demo()