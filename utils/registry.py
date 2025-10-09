


class Registry(object):
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        format_str = self.__class__.__name__
        return format_str

    @property
    def name(self):
        return self._name
