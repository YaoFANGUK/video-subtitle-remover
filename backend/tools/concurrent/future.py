from typing import List, Optional, Callable, Iterable, Sized, Tuple, Union

from PySide6.QtCore import QObject, Signal, QMutex, QSemaphore


class FutureError(BaseException):
    pass


class FutureFailed(FutureError):
    def __init__(self, _exception: Optional[BaseException]):
        super().__init__()
        self.exception = _exception

    def __repr__(self):
        return f"FutureFailed({self.exception})"

    def __str__(self):
        return f"FutureFailed({self.exception})"


class GatheredFutureFailed(FutureError):
    def __init__(self, failures: List[Tuple['Future', BaseException]]):
        super().__init__()
        self.failures = failures

    def __repr__(self):
        return f"GatheredFutureFailed({self.failures})"

    def __str__(self):
        return f"GatheredFutureFailed({self.failures})"

    def __iter__(self):
        return iter(self.failures)

    def __len__(self):
        return len(self.failures)


class FutureCancelled(FutureError):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"FutureCanceled()"

    def __str__(self):
        return f"FutureCanceled()"


class Future(QObject):
    result = Signal(object)  # self
    done = Signal(object)  # self
    failed = Signal(object)  # self
    partialDone = Signal(object)  # child future
    childrenDone = Signal(object)  # self

    def __init__(self, semaphore=0):
        super().__init__()
        self._taskID = None
        self._failedCallback = lambda e: None
        self._done = False
        self._failed = False
        self._result = None
        self._exception = None
        self._children = []
        self._counter = 0
        self._parent = None
        self._callback = lambda _: None
        self._mutex = QMutex()
        self._extra = {}
        self._semaphore = QSemaphore(semaphore)

    def __onChildDone(self, childFuture: 'Future') -> None:
        self._mutex.lock()
        if childFuture.isFailed():
            self._failed = True
        self._counter += 1
        self.partialDone.emit(childFuture)
        try:
            idx = getattr(childFuture, "_idx")
            self._result[idx] = childFuture._result
            self._mutex.unlock()
        except AttributeError:
            self._mutex.unlock()
            raise RuntimeError(
                "Invalid child future: please ensure that the child future is created by method 'Future.setChildren'")

        if self._counter == len(self._children):
            if self._failed:  # set failed
                l = []

                for i, child in enumerate(self._children):
                    if isinstance(e := child.getException(), FutureError):
                        l.append((self._children[i], e))

                self.setFailed(GatheredFutureFailed(l))
            else:
                self.setResult(self._result)

    def __setChildren(self, children: List['Future']) -> None:
        self._children = children
        self._result = [None] * len(children)

        for i, fut in enumerate(self._children):
            setattr(fut, f"_idx", i)
            fut.childrenDone.connect(self.__onChildDone)
            fut._parent = self

        for i, fut in enumerate(self._children):  # check if child is done
            if fut.isDone():
                self.__onChildDone(fut)

    def setResult(self, result) -> None:
        """
        :param result: The result to set
        :return: None

        do not set result in thread pool,or it may not set correctly
        please use in main thread,or use signal-slot to set result !!!
        """
        if self._done:
            raise RuntimeError("Future already done")

        self._result = result
        self._done = True

        if self._parent:
            self.childrenDone.emit(self)

        if self._callback:
            self._callback(result)

        self.result.emit(result)
        self.done.emit(self)

    def setFailed(self, exception) -> None:
        """
        :param exception: The exception to set
        :return: None
        """
        if self._done:
            raise RuntimeError("Future already done")

        self._exception = FutureFailed(exception)
        self._done = True
        self._failed = True

        if self._parent:
            self.childrenDone.emit(self)

        if self._failedCallback:
            self._failedCallback(self)

        self.failed.emit(self._exception)
        self.done.emit(self)

    def setCallback(self, callback: Callable[[object, ], None]) -> None:
        self._callback = callback

    def setFailedCallback(self, callback: Callable[['Future', ], None]) -> None:
        self._failedCallback = lambda e: callback(self)

    def hasException(self) -> bool:
        if self._children:
            return any([fut.hasException() for fut in self._children])
        else:
            return self._exception is not None

    def hasChildren(self) -> bool:
        return bool(self._children)

    def getException(self) -> Optional[BaseException]:
        return self._exception

    def setTaskID(self, _id: int) -> None:
        self._taskID = _id

    def getTaskID(self) -> int:
        return self._taskID

    def getChildren(self) -> List['Future']:
        return self._children

    @staticmethod
    def gather(futures: {Iterable, Sized}) -> 'Future':
        """
        :param futures: An iterable of Future objects
        :return: A Future object that will be done when all futures are done
        """
        future = Future()
        future.__setChildren(futures)
        return future

    @property
    def semaphore(self):
        return self._semaphore

    def wait(self):
        if self.hasChildren():
            for child in self.getChildren():
                child.wait()
        else:
            self.semaphore.acquire(1)

    def synchronize(self):
        self.wait()

    def isDone(self) -> bool:
        return self._done

    def isFailed(self) -> bool:
        return self._failed

    def getResult(self) -> Union[object, List[object]]:
        return self._result

    def setExtra(self, key, value):
        self._extra[key] = value

    def getExtra(self, key):
        return self._extra.get(key, None)

    def hasExtra(self, key):
        return key in self._extra

    def then(self, onSuccess: Callable, onFailed: Callable = None, onFinished : Callable = None):
        self.result.connect(onSuccess)

        if onFailed:
            self.failed.connect(onFailed)

        if onFinished:
            self.done.connect(onFinished)

        return self

    def __getattr__(self, item):
        return self.getExtra(item)

    def __repr__(self):
        return f"Future:({self._result})"

    def __str__(self):
        return f"Future({self._result})"

    def __eq__(self, other):
        return self._result == other._result