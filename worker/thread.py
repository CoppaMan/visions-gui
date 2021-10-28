from threading import Thread
import logging

import functools

class ThreadedWorker(Thread):
    def __init__(self):
        Thread.__init__(self)            
        self.stop_signal = False

        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.callbacks = []

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
        