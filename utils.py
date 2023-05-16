import time

def preprocess_string(s):
    s = s.strip()  # remove leading and trailing whitespaces
    if s.startswith("#"):  # check if the string starts with "#"
        s = s[1:]  # remove the first character if it is "#"
    return s

class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Time elapsed {time.time()-self.t:.02f}s')