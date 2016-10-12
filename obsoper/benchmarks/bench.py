"""Simple benchmarking tool"""
import time
import os
import json
import locate


class Suite(object):
    """Benchmark suite base class"""
    def setUp(self):
        """Ran before bench mark"""
        pass

    def tearDown(self):
        """Ran after bench mark"""
        pass


class Timer(object):
    """Simple timer context manager"""
    def __init__(self):
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def run(save_results=False, particular_method=None):
    """Simple benchmark suite runner"""
    locator = locate.Locator.fromchoice(particular_method)

    previous_duration = load()

    for suite_class in Suite.__subclasses__():
        banner(suite_class)
        suite = suite_class()
        suite.setUp()

        for method in locator.select(sorted(dir(suite))):
            current_duration = time_method(getattr(suite, method))

            report(method,
                   current_duration,
                   previous_duration.get(method))

            if save_results:
                save(method, current_duration)

        suite.tearDown()


def time_method(method):
    """Time a function"""
    times = []
    for _ in range(3):
        with Timer() as timer:
            method()
        times.append(timer.interval)
    return min(times)


def banner(suite_class):
    """inform user of benchmark suite"""
    message = "Suite: {}".format(suite_name(suite_class))
    print
    print message
    print "-" * len(message)


def suite_name(suite_class):
    """Name a benchmark suite"""
    return "{}.{}".format(suite_class.__module__,
                          suite_class.__name__)


def report(method, current, previous=None):
    """inform user of result"""
    message = "{}() best of 3 took {:.03f} sec.".format(method, current)
    if previous is not None:
        template = " {:+.1f}% ({:.03f} sec) {}"
        message += template.format(percent(current, previous),
                                   delta(current, previous),
                                   speed(current, previous))
    print message


def speed(current, previous):
    """Speed up direction calculation"""
    if (previous - current) > 0:
        return "faster"
    else:
        return "slower"


def delta(current, previous):
    """Speed up time delta calculation"""
    return abs(previous - current)


def percent(current, previous):
    """Speed up percentage calculation"""
    return (100. * (previous - current)) / previous


def load(path="bench.json"):
    """Load previous results"""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def save(name, duration, path="bench.json"):
    """Save results"""
    # Collect results
    results = load(path)
    results[name] = duration

    # I/O
    if os.path.exists(path):
        mode = "r+"
    else:
        mode = "w"
    with open(path, mode) as handle:
        json.dump(results, handle)
