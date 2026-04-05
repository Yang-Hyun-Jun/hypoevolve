import copy

import numpy as np

from hypoevolve.hypo.nodes.base import Node, NodeIOTypes
from hypoevolve.hypo.tree.base import HypoTree


class HypoTreeGenerator:
    """
    Random Hypo Tree Generator
    """

    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def generate(self, max_depth: int, tree_max_iter: int = 20):
        """
        Generate a random Hypo Tree
        """
        state = self.reset(max_depth)
        mask = state["mask"]
        _iter = 0

        while True:
            _iter += 1
            prob = np.array(mask) / np.array(mask).sum()
            node_index = np.random.choice(np.arange(len(self.nodes)), p=prob)
            next_state = self.step(node_index)
            mask = next_state["mask"]

            if next_state["done"] or _iter > tree_max_iter:
                return self.tree

    def reset(self, max_depth: int, tree_name: str | None = "") -> dict:
        """
        Reset Hypo Tree Generator
        """

        # 생성할 Tree의 max_depth
        self.max_depth = max_depth
        # 생성할 Tree 객체 선언
        self.tree = HypoTree(tree_name)
        # node mask
        mask = np.array(self.get_initial_mask())
        return {"mask": mask, "done": False}

    def step(self, node_index: int) -> dict:
        """
        Insert a specific node and proceed one timestep
        """

        # 새로운 노드 인스턴스 생성 (깊은 복사로 완전히 독립적인 객체 생성)
        node_template = self.nodes[node_index]
        new_node = copy.deepcopy(node_template)

        self.tree.insert(new_node)
        is_done = self.tree.iscompleted
        is_last = self.tree.get_depth(self.tree.current) >= (self.max_depth - 1)

        if not is_last:
            # IO가 일치하고 리프 노드가 아닌 노드들만 고려
            mask = [
                1
                if (n.output_type in self.tree.current.input_types) & n.max_childs
                else 0
                for n in self.nodes
            ]

        else:
            # IO가 일치하고 리프 노드인 노드들만 고려
            mask = [
                1
                if (n.output_type in self.tree.current.input_types) & ~n.max_childs
                else 0
                for n in self.nodes
            ]

        # 배치할 수 있는 노드가 없으면 랜덤 생성 실패
        if (sum(mask) == 0) & ~is_done:
            self.max_depth += 1

        return {"mask": mask, "done": is_done}

    def get_initial_mask(self):
        """
        Hypo Tree Root Node Masking
        """
        initial_mask = [
            1 if (n.output_type == NodeIOTypes.BINARY) & (n.max_childs >= 1) else 0
            for n in self.nodes
        ]

        return initial_mask
