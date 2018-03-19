from asyncore import dispatcher
from asynchat import async_chat
import socket, asyncore

PORT = 5005
NAME = 'TestChat'

class CommandHandler:
    """
    类似于标准库中cmd.Cmd的简单命令处理程序.
    """
    def unknow(self, session, cmd):
        # 相应未知命令
        session.push(b'Unknow command:' + str.encode(cmd))

    def handle(self, session, line):
        line_str = bytes.decode(line)
        # 处理从给定的会话中接受到的行.
        if not line_str.strip():
            return
        # 拆分命令
        parts = line_str.split(' ', 1)
        cmd = parts[0]
        try:
            line_str = parts[1].strip()
        except IndexError:
            line_str = ''
        # 试着查找处理程序
        meth = getattr(self, 'do_' + cmd, None)
        try:
            # 假定它是可调用的
            meth(session, line_str)
        except TypeError:
            # 如果不可以被调用,此段代码响应未知的命令
            self.unknow(session, cmd)

class EndSession(Exception):
    pass

class Room(CommandHandler):
    """
    可以包括一个或多个用户(会话)的泛型环境.它负责基本的命令处理和广播
    """

    def __init__(self, server):
        self.server = server
        self.sessions = []

    def add(self, session):
        # 一个会话(用户)已进入房间
        self.sessions.append(session)

    def remove(self, session):
        # 一个会话(用户)已离开房间
        self.sessions.remove(session)

    def broadcast(self, line):
        # 向房间中的所有会话发送一行
        for session in self.sessions:
            session.push(line)

    def do_logout(self, session, line):
        # 响应logout命令
        raise EndSession

class LoginRoom(Room):
    """
    为刚刚连接上的用户准备的房间.
    """

    def add(self, session):
        Room.add(self, session)
        # 当用户进入时, 问候他或她
        self.broadcast(b'Welcome to ' + str.encode(self.server.name))

    def unknow(self, session, cmd):
        # 所有未知命令(除了login或者logout外的一切)
        # 会导致一个警告:
        session.push(b'Please log in\nUse "login <nick>"\r\n')

    def do_login(self, session, line):
        name = line.strip()
        # 确保用户输入了名字:
        if not name:
            session.push(b'Please enter a name\r\n')
        # 确保用户名并没有被使用:
        elif name in self.server.users:
            session.push(b'The name ' + str.encode(name) + b' is taken.\r\n')
            session.push(b'Please try again.\r\n')
        else:
            # 名字无问题,所以存储在会话中,并且将用户移动到主聊天室
            session.name = name
            session.enter(self.server.main_room)

class ChatRoom(Room):
    """
    为多用户相互聊天准备的房间.
    """

    def add(self, session):
        # 告诉所有人有新用户进入:
        self.broadcast(str.encode(session.name) + b' has entered the room.\r\n')
        self.server.users[session.name] = session
        Room.add(self, session)

    def remove(self, session):
        Room.remove(self, session)
        # 告诉所有人有用户离开:
        self.broadcast(str.encode(session.name) + b' has left the room.\r\n')

    def do_say(self, session, line):
        self.broadcast(str.encode(session.name) + b': ' + str.encode(line) + b'\r\n')

    def do_look(self, session, line):
        # 处理look命令, 该命令用于查看谁在房间内
        session.push(b'The following are in this room:\r\n')
        for other in self.sessions:
            session.push(str.encode(other.name) + b'\r\n')

    def do_who(self, session, line):
        # 处理who命令, 该命令用于查看谁登录了
        session.push(b'The following are logged in:\r\n')
        for name in self.server.users:
            session.push(str.encode(name) + b'\r\n')

class LogoutRoom(Room):
    """
    为单用户准备的简单房间.只用于将用户名从服务器移除
    """

    def add(self, session):
        # 当会话(用户)进入要移除的LogoutRoom时
        try:
            del self.server.users[session.name]
        except KeyError:
            pass

class ChatSession(async_chat):
    """
    单会话,负责和单用户通信
    """
    def __init__(self, server, sock):
        async_chat.__init__(self, sock)
        self.server = server
        # linux用'\n',windows用'\r\n',mac用'\r'表示换行
        self.set_terminator(b"\r\n")
        self.data = []
        self.name = None
        # 所有的会话都开始于单独的LoginRoom中:
        self.enter(LoginRoom(server))

    def enter(self, room):
        # 从当前房间移除自身(self), 并且将自身添加到下一个房间......
        try:
            cur = self.room
        except AttributeError:
            pass
        else:
            cur.remove(self)
        self.room = room
        room.add(self)

    def collect_incoming_data(self, data):
        self.data.append(data)

    def found_terminator(self):
        line = b"".join(self.data)
        self.data = []
        try:
            self.room.handle(self, line)
        except EndSession:
            self.handle_close()

    def handle_close(self):
        async_chat.handle_close(self)
        self.enter(LogoutRoom(self.server))

class ChatServer(dispatcher):
    """
    只有一个房间的聊天服务器
    """

    def __init__(self, port, name):
        dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        # try to re-use a server port if possible
        self.set_reuse_addr()
        self.bind(('', port))
        self.listen(5)
        self.name = name
        self.users = {}
        self.main_room = ChatRoom(self)

    def handle_accept(self):
        conn, addr = self.accept()
        ChatSession(self, conn)

if __name__=='__main__':
    s = ChatServer(PORT, NAME)
    try:
        asyncore.loop()
    except KeyboardInterrupt:
        print