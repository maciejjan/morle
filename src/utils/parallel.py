import multiprocessing
import time
import tqdm
from typing import Any, Callable, Iterable, List, Tuple


def parallel_execute(function :Callable[..., None] = None,
                     data :List[Any] = None,
                     num_processes :int = 1,
                     additional_args :Tuple = (),
                     show_progressbar :bool = False,
                     progressbar_total :int = None) -> Iterable:

    mandatory_args = (function, data, num_processes, additional_args)
    assert not any(arg is None for arg in mandatory_args)

    # partition the data into chunks for each process
    step = len(data) // num_processes
    data_chunks = []
    i = 0
    while i < num_processes-1:
        data_chunks.append(data[i*step:(i+1)*step])
        i += 1
    # account for rounding error while processing the last chunk
    data_chunks.append(data[i*step:])

    queue = multiprocessing.Queue(10000)
    queue_lock = multiprocessing.Lock()

    def _output_fun(x):
        successful = False
        queue_lock.acquire()
        try:
            queue.put_nowait(x)
            successful = True
        except Exception:
            successful = False
        finally:
            queue_lock.release()
        if not successful:
            # wait for the queue to be emptied and try again
            time.sleep(1)
            _output_fun(x)

    processes, joined = [], []
    for i in range(num_processes):
        p = multiprocessing.Process(target=function,
                                    args=(data_chunks[i], _output_fun) +\
                                         additional_args)
        p.start()
        processes.append(p)
        joined.append(False)

    progressbar, state = None, None
    if show_progressbar:
        if progressbar_total is None:
            progressbar_total = len(data)
        progressbar = tqdm.tqdm(total=progressbar_total)
    while not all(joined):
        count = 0
        queue_lock.acquire()
        try:
            while not queue.empty():
                yield queue.get()
                count += 1
        finally:
            queue_lock.release()
        if show_progressbar:
            progressbar.update(count)
        for i, p in enumerate(processes):
            if not p.is_alive():
                p.join()
                joined[i] = True
    if show_progressbar:
        progressbar.close()

