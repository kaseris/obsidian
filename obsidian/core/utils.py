import sys
import inspect
import importlib


def get_members(module, disregard: list = None):
    """
    Returns a generator yielding the classes from a specified module.

    Args:
        module (str): The name of the module from which to extract classes.
        disregard (list, optional): A list of members to disregard. Defaults to None.

    Yields:
        tuple: A tuple containing the name and class object of each class in the module.

    Example:
        >>> for name, cls in get_members('my_module'):
        ...     print(name, cls)
        ...
        MyClass <class 'my_module.MyClass'>
        AnotherClass <class 'my_module.AnotherClass'>
    """
    if disregard is None:
        disregard = []

    _ = importlib.import_module(module)

    for _member, _module in inspect.getmembers(sys.modules[module]):
        if _member not in disregard:
            if inspect.isclass(_module):
                yield _member, _module
