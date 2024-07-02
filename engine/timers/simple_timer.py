import time
from functools import wraps


class SimpleTimer(object):

    meta = {}
    _stack = []

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.__class__._stack.append(self.name)
        # self.start_time = time.time()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # end_time = time.time()
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        self.__class__._record_time(elapsed_time)
        self.__class__._stack.pop()

    @classmethod
    def _record_time(cls, elapsed_time):
        current_key = '.'.join(cls._stack)
        cls.meta.setdefault(current_key, {'total_time': 0, 'call_count': 0})
        cls.meta[current_key]['total_time'] += elapsed_time
        cls.meta[current_key]['call_count'] += 1

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with SimpleTimer(self.name):
                return func(*args, **kwargs)
        return wrapper

    @classmethod
    def get_time_info(cls):
        if not cls.meta:
            return "No timing information available."

        max_key_length = max(len(key) for key in cls.meta.keys())
        max_count_length = max(
            len(str(value['call_count'])) for value in cls.meta.values())
        max_total_time_length = max(
            len(f"{value['total_time']:.4f}") for value in cls.meta.values())

        infos = []
        for key in sorted(cls.meta.keys()):
            value = cls.meta[key]
            avg_time = value['total_time'] / value['call_count']
            infos.append(
                f"Key: {key:<{max_key_length}s} - "
                f"Call count: {value['call_count']:{max_count_length}d} - "
                f"Total time: {value['total_time']:{max_total_time_length}.4f}s - "
                f"Average time: {avg_time:.4f}s"
            )
        return '\n'.join(infos)


if __name__ == "__main__":
    @SimpleTimer('a')
    def example_function():
        time.sleep(0.5)
        with SimpleTimer('b'):
            time.sleep(0.3)
            with SimpleTimer('c'):
                time.sleep(0.2)
    for _ in range(5):
        example_function()

    print(SimpleTimer.get_time_info())
