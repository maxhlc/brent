class Factory:

    # Registry of classes
    registry = {}

    def __init_subclass__(cls, **kwargs):
        # Call parent method
        super().__init_subclass__(**kwargs)

        # Ensure subclasses have their own seperate registries
        cls.registry = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(class_type):
            # Raise error if handler name already registered
            if name in cls.registry:
                raise RuntimeError(f"Class already registered: {name}")

            # Store handler name and type
            cls.registry[name] = class_type

            # Return handler
            return class_type

        # Return inner function
        return inner_wrapper

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        # Raise error if not in registry
        if name not in cls.registry:
            raise RuntimeError(f"Unknown class: {name}")

        # Extract class
        class_type = cls.registry[name]

        # Return initialised class
        return class_type(*args, **kwargs)
