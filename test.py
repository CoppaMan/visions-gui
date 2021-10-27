from time import sleep

from worker.thread import ModelWorker

class Test(ModelWorker):
    def __init__(self):
        ModelWorker.__init__(self)

    def function(self):
        for i in range(5):
            sleep(.5)
            print(i+1)
        self.notify_image()
        for i in range(5):
            sleep(.5)
            print(i+6)

test = Test()
test.start()

@test.register_done
def finish():
    print('Sequence is done!')

@test.register_image
def update():
    print('halfways!')