import string

import time
import numpy as np


class Profiler:
    timings = {}

    currentTimes = {}

    def startProfile(self, profName: string) -> None:
        self.currentTimes[profName] = time.time()

    def endProfile(self, profName: string) -> None:
        self.timings[profName] = time.time() - self.currentTimes[profName]

    def printTimings(self) -> None:
        print(self.timings)
