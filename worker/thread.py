from threading import Thread
import logging

class ThreadedWorker(Thread):
    def __init__(self):
        Thread.__init__(self)            
        self.stop_signal = False

        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.callbacks = []

    def register_done(self, callback):
        '''
        Register to the thread finish event
        '''
        self.callbacks.append(callback)
        return callback

    def notify_done(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def function(self):
        '''
        User function to run as a thread, overwrite this method
        '''
        self.logger.warning(
            'Running default function. No user function defined'
        )

    def stop(self):
        '''
        Stops the thread and returns control to the caller
        '''
        self.logger.debug('Sending stop signal')
        self.stop_signal = True
        try:
            self.logger.debug('Joining the thread')
            self.join()
            self.logger.debug('Thread joined')
        except Exception as e:
            self.logger.error(e)

    def run(self):
        '''
        Runs your function and notifies subscribers
        '''
        self.logger.debug('Execute function')
        self.function()
        self.logger.debug('Inform subscribers')
        self.notify_done()


class ModelWorker(ThreadedWorker):
    '''
    Special case of models running in a seperate thread
    '''
    def __init__(self):
        ThreadedWorker.__init__(self)

        self.image_callback = []
        self.progress_callback = []

    def register_image(self, callback):
        '''
        Register for image updates
        '''
        self.image_callback.append(callback)
        return callback

    def notify_image(self, *args, **kwargs):
        for callback in self.image_callback:
            callback(*args, **kwargs)

    def register_progress(self, callback):
        '''
        Register for progress update
        '''
        self.progress_callback.append(callback)
        return callback

    def notify_progress(self, *args, **kwargs):
        for callback in self.progress_callback:
            callback(*args, **kwargs)
