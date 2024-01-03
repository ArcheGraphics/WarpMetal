import cProfile
import timeit


class ScopedTimer:
    indent = -1

    enabled = True

    def __init__(
            self,
            name,
            active=True,
            print=True,
            detailed=False,
            dict=None,
            use_nvtx=False,
            color="rapids",
            synchronize=False,
    ):
        """Context manager object for a timer

        Parameters:
            name (str): Name of timer
            active (bool): Enables this timer
            print (bool): At context manager exit, print elapsed time to sys.stdout
            detailed (bool): Collects additional profiling data using cProfile and calls ``print_stats()`` at context exit
            dict (dict): A dictionary of lists to which the elapsed time will be appended using ``name`` as a key
            use_nvtx (bool): If true, timing functionality is replaced by an NVTX range
            color (int or str): ARGB value (e.g. 0x00FFFF) or color name (e.g. 'cyan') associated with the NVTX range
            synchronize (bool): Synchronize the CPU thread with any outstanding CUDA work to return accurate GPU timings

        Attributes:
            elapsed (float): The duration of the ``with`` block used with this object
        """
        self.name = name
        self.active = active and self.enabled
        self.print = print
        self.detailed = detailed
        self.dict = dict
        self.use_nvtx = use_nvtx
        self.color = color
        self.synchronize = synchronize
        self.elapsed = 0.0

        if self.dict is not None:
            if name not in self.dict:
                self.dict[name] = []

    def __enter__(self):
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.use_nvtx:
                import nvtx

                self.nvtx_range_id = nvtx.start_range(self.name, color=self.color)
                return

            self.start = timeit.default_timer()
            ScopedTimer.indent += 1

            if self.detailed:
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.use_nvtx:
                import nvtx

                nvtx.end_range(self.nvtx_range_id)
                return

            if self.detailed:
                self.cp.disable()
                self.cp.print_stats(sort="tottime")

            self.elapsed = (timeit.default_timer() - self.start) * 1000.0

            if self.dict is not None:
                self.dict[self.name].append(self.elapsed)

            indent = ""
            for i in range(ScopedTimer.indent):
                indent += "\t"

            if self.print:
                print("{}{} took {:.2f} ms".format(indent, self.name, self.elapsed))

            ScopedTimer.indent -= 1
