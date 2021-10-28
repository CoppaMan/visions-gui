class A:
    def __init__(self):
        pass


class B:
    def __init__(self):
        self.a = A()


b = B()


def funcDec(func):
    localVariable = "I'm a local string"

    def wrapped(*args):
        print("Calling localVariable from funcDec " + localVariable)
        func(*args)
        print("done with calling f1")

    wrapped.attrib = localVariable
    return wrapped

@funcDec
def f1(x, y):
    print(x + y)
    print('f1.attrib: {!r}'.format(f1.attrib))

f1(2, 3)