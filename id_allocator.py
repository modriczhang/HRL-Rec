class IdAllocator(object):
    def __init__(self):
        self._id = 1
        self._tbl = {}

    def allocate(self, x):
        if type(x) is not str:
            raise Exception('only str is supported in IdAllocator.')
        if x not in self._tbl:
            self._id += 1
            self._tbl[x] = self._id
        return self._tbl[x]

    def unique_id_num(self):
        return self._id