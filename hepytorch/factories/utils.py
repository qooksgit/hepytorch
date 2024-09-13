from inspect import getmembers, isclass, isabstract
import typing


def load_classes(parent, modules) -> dict:
    map = {}  # key: name, value: class
    classes = getmembers(modules, lambda m: isclass(m) and not isabstract(m))
    for name, _type in classes:
        if isclass(_type) and issubclass(_type, parent):
            map.update([[name, _type]])
    return map


def get_instance(parent, modules, cfg) -> typing.Any:
    name = cfg.get("name")
    classes = load_classes(parent, modules)
    if name in classes:
        kwargs = cfg.get("kwargs")
        return classes[name](**kwargs)
    else:
        raise ValueError("Model not found: ", name)
