import inspect

try:
    from anytree import NodeMixin
except ImportError:  # pragma: no cover - lightweight fallback for bare environments
    class NodeMixin:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.parent = None


class NodeIOTypes:
    BINARY = "binary"
    FLOAT = "float"


class Node(NodeMixin):
    def __init__(
        self,
        input_types: list[NodeIOTypes],
        output_type: NodeIOTypes,
        max_childs: int,
    ):
        self.max_childs = max_childs
        self.input_types = input_types
        self.output_type = output_type
        self.childs = []

    def __repr__(self):
        return self.name

    def __call__(self, *inputs):
        return self.activate(*inputs)

    @property
    def name(self):
        return type(self).__name__ + "()"

    @property
    def description(self):
        doc = self.__class__.__doc__
        if doc:
            return inspect.cleandoc(doc)
        return ""

    @property
    def full(self):
        return len(self.childs) >= self.max_childs

    @property
    def params(self) -> dict:
        raise NotImplementedError

    def add_child(self, child: NodeMixin):
        child.set_parent(self)
        self.childs.append(child)

    def set_parent(self, parent):
        self.parent = parent

    def activate(self, *inputs):
        raise NotImplementedError

    def propagate(self):
        if not self.full:
            raise ValueError("can't propagate through this node if not node.full")
        return self.activate(*[c.propagate() for c in self.childs])

    def to_dict(self) -> dict:
        return {
            "class": type(self).__name__,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict):
        from hypoevolve.hypo.nodes import nodes

        node_class = getattr(nodes, data["class"])
        return node_class(**data["params"])
