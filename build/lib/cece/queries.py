import numpy as np

class Query:
    def __init__(self, concepts=np.empty(0), roles=dict()):
        assert(isinstance(concepts, np.ndarray))
        self.node_count = len(concepts)
        #self.concepts = [frozenset(c) for c in concepts]
        self.concepts = concepts

        assert(isinstance(roles, dict))
        for role, adjacency_matrix in roles.items():
            assert(isinstance(adjacency_matrix, np.ndarray))
            n, m = adjacency_matrix.shape
            assert(n == m == self.node_count)

        self.roles = roles
        neighbors = {}
        for role, adjacency_matrix in roles.items():
            neighbors[role] = np.empty(self.node_count, list)
            for i in range(self.node_count):
                neighbors[role][i] = list(np.nonzero(adjacency_matrix[i,:])[0])
        self.neighbors = neighbors

    def __str__(self):
        return '\n'.join((
            str(self.concepts),
            str(self.roles),
        ))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def union(self, q):
        concepts = np.empty((self.node_count, q.node_count), dtype=frozenset)
        for i in range(self.node_count):
            concepts[i,:] = self.concepts[i] & q.concepts
        concepts = concepts.flatten()

        roles = dict.fromkeys(self.roles.keys() | q.roles.keys())
        for role in roles:
            try:
                adjacency_matrix1 = self.roles[role]
                adjacency_matrix2 = q.roles[role]
            except KeyError:
                continue
            roles[role] = np.kron(adjacency_matrix1, adjacency_matrix2)

        return Query(concepts, roles)

    def delete_node(self, i):
        self.node_count -= 1
        self.concepts = np.delete(self.concepts, i, 0)
        for role in self.roles:
            self.roles[role] = np.delete(np.delete(self.roles[role], i, 0), i, 1)

        return self

    def remove_non_connected(self):
        stack = [0]
        visited = np.zeros(self.node_count, dtype=bool)

        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            for role in self.roles.values():
                neighbors, = np.nonzero(role[node,:])
                stack.extend(neighbors)

        for node, vis in reversed(list(enumerate(visited))):
            if not vis:
                self.delete_node(node)

        return self

    def minimize(self):
        node_deleted = True
        while node_deleted:
            node_deleted = False
            i = self.node_count - 1
            while i >= 0:
                j = self.node_count - 1
                while i >= 0 and j >= 0:
                    if i != j:
                        if self.concepts[i] >= self.concepts[j]:
                            # check if j maps to i
                            for adjacency_matrix in self.roles.values():
                                if not (
                                        adjacency_matrix[i,i] >= (adjacency_matrix[j,j]
                                                                  or adjacency_matrix[i,j]
                                                                  or adjacency_matrix[j,i])
                                        # compare rows i and j but skip (i,j), (j,j) comparison
                                        and np.all(adjacency_matrix[i,:j] >= adjacency_matrix[j,:j])
                                        and np.all(adjacency_matrix[i,j+1:] >= adjacency_matrix[j,j+1:])
                                        # compare columns i and j but skip (j,i), (j,j) comparison
                                        and np.all(adjacency_matrix[:j,i] >= adjacency_matrix[:j,j])
                                        and np.all(adjacency_matrix[j+1:,i] >= adjacency_matrix[j+1:,j])
                                ):
                                    break
                            else:  # executed if j maps to i
                                node_deleted = True
                                self.delete_node(j)
                                # i may now correspond to a different node
                                if i > j:
                                    i -= 1
                    j -= 1
                i -= 1
        return self
