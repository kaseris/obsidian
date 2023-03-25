from typing import Callable

class Registry:
    """
    Factory class for creating modules.

    Allows for fast instantiation of modules using string names provided in configuration files.
    """

    registry = dict()

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator method for registering a class in the registry.

        Args:
            name (str): The name of the class to register.

        Returns:
            Callable: A decorator that wraps the registered class.
        """

        def inner_wrapper(wrapped_class) -> Callable:
            """
            Wrapper function that registers a class in the registry.

            Args:
                wrapped_class (Callable): The class to register.

            Returns:
                Callable: The registered class.
            """
            if name in cls.registry:
                print(f'Class {name} already exists. It will be replaced')
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def __getitem__(cls, item: str) -> Callable:
        """
        Get the registered class with the specified name.

        Args:
            item (str): The name of the registered class to retrieve.

        Returns:
            Callable: The registered class.
        """
        return cls.registry[item]
