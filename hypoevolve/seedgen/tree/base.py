try:
    from anytree import RenderTree
except ImportError:  # pragma: no cover - lightweight fallback for bare environments
    def RenderTree(root):
        def walk(node, prefix="", is_last=True, is_root=True):
            if node is None:
                return
            branch = ""
            if not is_root:
                branch = "└── " if is_last else "├── "
            yield prefix + branch, "", node
            children = list(getattr(node, "childs", []))
            next_prefix = prefix + ("    " if is_last else "│   ") if not is_root else ""
            for idx, child in enumerate(children):
                yield from walk(
                    child,
                    next_prefix,
                    idx == len(children) - 1,
                    False,
                )

        yield from walk(root)

from hypoevolve.seedgen.nodes.base import Node


class HypoTree:
    """
    Hypo Tree
    """

    def __init__(self, name=""):
        self._root = None
        self._currnode = None
        self.name = name
        self.nodes = []
        self.depth = 0

    @property
    def root(self):
        return self._root

    @property
    def current(self):
        return self._currnode

    def insert(self, node):
        # 최초 삽입인 경우 => 루트 노드로 넣고 종료]
        if not self._root:
            self._root = node
            self._currnode = self._root
            self.nodes.append(self._root)
            self.depth = 0  # 루트 노드 깊이는 0
            return

        # 현재 노드의 자식이 가득 차지 않은 경우 => 자식 노드로 넣고 종료
        if not self._currnode.full:
            self._currnode.add_child(node)
            self.nodes.append(node)

            depth = self.get_depth(node)
            self.depth = max(self.depth, depth)

            # 삽입한 노드가 리프 노드가 아닌 경우 => 방금 삽입한 노드로 이동
            if node.max_childs > 0:
                self._currnode = node

            # 삽입한 노드가 리프 노드인 경우 => full이 아닌 가까운 부모 노드로 이동
            else:
                while self._currnode.full:
                    if self._currnode.parent is None:
                        break

                    self._currnode = self._currnode.parent

        # 현재 노드의 자식이 가득 찬 경우 => 부모 노드로 이동 후 너비 확장
        else:
            self._currnode = self._currnode.parent
            self.insert(node)

    @property
    def iscompleted(self):
        """
        Clause Tree 삽입 완료 여부

        : 트리 상단으로 올라가면서 자식 노드 다 채웠는지 확인
        """
        temp = self._currnode

        while True:
            # 현재 노드의 자식이 있고 가득 차지 않은 경우 => 완료 아님
            if (temp.max_childs > 0) & (not temp.full):
                return False

            # 부모 노드가 없는 최상단 노드인 경우 => 완료
            if temp.parent is None:
                return True

            temp = temp.parent

    def get_depth(self, node):
        temp = node
        depth = 0

        while temp.parent is not None:
            depth += 1
            temp = temp.parent

        return depth

    def evaluate(self):
        return self.root.propagate()

    def get_node_descriptions(self) -> str:
        descriptions = []

        for node in self.nodes:
            descriptions.append(f"- {node.name}: {node.description}")

        return "\n".join(descriptions)

    def render(self, return_str: bool = False):
        string = "\n".join(
            [f"{pre}{node.name}" for pre, _, node in RenderTree(self.root)]
        )

        if return_str:
            return string

        print(string)

    def __call__(self, *args, **kwargs):
        return self.evaluate()

    def to_dict(self) -> dict:
        """
        ClauseTree를 딕셔너리로 변환
        """

        node_to_index = {id(node): idx for idx, node in enumerate(self.nodes)}

        serialized_nodes = [
            {
                **node.to_dict(),
                "parent_idx": node_to_index.get(id(node.parent))
                if node.parent
                else None,
            }
            for node in self.nodes
        ]

        return {
            "name": self.name,
            "depth": self.depth,
            "nodes": serialized_nodes,
            "root_idx": 0 if self.root else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HypoTree":
        """
        딕셔너리에서 HypoTree 인스턴스 생성
        """

        tree = cls(name=data["name"])

        if not data["nodes"]:
            return tree

        # 노드 생성 (각 노드의 from_dict 사용)
        nodes = [Node.from_dict(node_data) for node_data in data["nodes"]]

        # parent-child 관계 설정
        for idx, node_data in enumerate(data["nodes"]):
            if node_data["parent_idx"] is not None:
                nodes[node_data["parent_idx"]].add_child(nodes[idx])

        tree.nodes = nodes
        tree._root = nodes[data["root_idx"]] if data["root_idx"] is not None else None
        tree._currnode = tree._root
        tree.depth = data["depth"]
        return tree
