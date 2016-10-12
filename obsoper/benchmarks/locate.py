"""
Locate benchmark methods
"""


class Locator(object):
    """Benchmark loader"""
    @classmethod
    def fromchoice(cls, choice=None):
        """Factory method"""
        if choice is None:
            return cls()
        else:
            return Particular(choice)

    def select(self, methods):
        """Generate benchmark methods"""
        for method in methods:
            if self.is_benchmark(method):
                yield method

    @staticmethod
    def is_benchmark(method):
        """Detect benchmark methods"""
        return method.lower().startswith("bench_")


class Particular(Locator):
    """Load particular benchmark method if available"""
    def __init__(self, particular_method):
        self.particular_method = particular_method

    def select(self, methods):
        """Generate benchmark methods"""
        for method in methods:
            if method.lower() == self.particular_method:
                yield method
