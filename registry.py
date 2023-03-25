from typing import Callable


class Registry:
    """Factory class for creating modules."""
    registry = dict()

    @classmethod
    def register(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                print(f'Class {name} already exists. It will be replaced')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    def __getitem__(cls, item):
        return cls.registry[item]
    