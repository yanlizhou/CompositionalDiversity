import builtins


# class GrammarMixin:
#     primitive_funcs: list
#
#     def _register_primitives(self):
#         self._builtins = dict()
#         for func in self.primitive_funcs:
#             key = func.__name__
#             if key in builtins.__dict__:
#                 self._builtins[key] = builtins.__dict__[key]
#             builtins.__dict__[key] = func
#
#     def __del__(self):
#         for func in self.primitive_funcs:
#             del builtins.__dict__[func.__name__]
#
#         # TODO: restore only default python builtin functions
#         # At the moment, if we call
#         #      >>> grammar = GrammarA()
#         #      >>> grammar = GrammarB()
#         # then upon deletion of `grammar` we will restore GrammarA primitives
#         # which we don't want.
#
#         # builtins.__dict__.update(self._builtins)


class GrammarMixin:
    primitive_funcs: list

    def _register_primitives(self):
        for func in self.primitive_funcs:
            name = func.__name__
            builtins.__dict__[name] = func

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._register_primitives()
