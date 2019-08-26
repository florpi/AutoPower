"""
Utility methods for multiprocessing.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import multiprocessing
import multiprocessing.queues


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class SharedCounter(object):
    """
    A synchronized shared counter.

    Original source:
    https://github.com/vterron/lemon/blob/master/methods.py#L507
    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        return self.count.value


class Queue(multiprocessing.queues.Queue):
    """
    An overloaded implementation of a multiprocessing.Queue which also
    works on macOS (where, for example, .qsize() otherwise usually
    raises a NotImplementedError).

    Original source:
    https://github.com/vterron/lemon/blob/master/methods.py#L536
    """

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs,
                                    ctx=multiprocessing.get_context())
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        return self.size.value

    def empty(self):
        return not self.qsize()
