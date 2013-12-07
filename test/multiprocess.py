from multiprocessing import Process, Queue, cpu_count
import weakref

class MultiProcessWorker(object):
    worker_func = None
    worker_proc = None
    nr_worker = None

    task = None     # queue of tasks: (task id, task arg)
    result = None   # queue of task results: (task id, result)

    def __init__(self, worker, nr_worker = cpu_count()):
        """:param worker: worker callable, taking an argument and return the
        result"""
        self.worker_func = worker
        self.nr_worker = nr_worker

    def __del__(self):
        if not self.worker_proc:
            return
        for _ in self.worker_proc:
            self.task.put(None)
        for i in self.worker_proc:
            i.join()

    def run(self, tasks):
        self.start()
        assert self.task.empty()
        for i in range(len(tasks)):
            self.task.put((i, tasks[i]))

        rst = dict()
        for _ in tasks:
            i, r = self.result.get()
            rst[i] = r

        return [rst[i] for i in range(len(tasks))]

    def _worker(self):
        while True:
            task = self.task.get()
            if task is None:
                break
            tid, targ = task
            self.result.put((tid, self.worker_func(targ)))

    def start(self):
        if self.worker_proc:
            return
        self.task = Queue()
        self.result = Queue()
        self.worker_proc = [
                Process(target = type(self)._worker,
                    args = (weakref.proxy(self), ))
                for _ in range(self.nr_worker)]
        for i in self.worker_proc:
            i.start()


#worker = MultiProcessWorker(func)
#worker.run([args list])
#del worker
