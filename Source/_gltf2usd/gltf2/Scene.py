class Scene:
    def __init__(self, scene_entry, scene_index, nodes):
        self._nodes = []
        if 'nodes' in scene_entry:
            for node_index in scene_entry['nodes']:
                self._nodes.append(nodes[node_index])

    def get_nodes(self):
        """Returns the node from the scene
        
        Returns:
            Scene[] -- nodes in the scene
        """

        return self._nodes