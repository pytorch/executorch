import sys


class LoggingPrinter:
    def __init__(self, filename=None):
        self.passthrough = True
        self.old_stdout = sys.stdout
        self.filename = filename
        if filename is not None:
            self.passthrough = False
            self.out_file = open(filename, "w")

    def write(self, text):
        self.old_stdout.write(text)
        if not self.passthrough:
            self.out_file.write(text)

    def close(self):
        if not self.passthrough:
            sys.stdout = self.old_stdout
            self.out_file.close()

    def flush(self):
        pass

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, type, value, traceback):
        if not self.passthrough:
            sys.stdout = self.old_stdout
