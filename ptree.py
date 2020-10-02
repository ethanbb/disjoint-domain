from __future__ import annotations
from typing import Mapping, Set, Optional
import xml.etree.ElementTree as ET
import functools
import operator


class PropositionalTree:
    def __init__(self, name: str, props: Optional[Mapping[str, Set[str]]] = None,
                 parent: Optional[PropositionalTree] = None):
        """
        Create a propositional tree (node).
        :param name: The name of this node (e.g. 'living thing')
        :param props: A dictionary of properties, not including ISA (e.g. {'can': 'grow', 'is': 'living'})
        :param parent: Another PropositionalTree that represents the most specific other category that any member of
                       this category belongs to (ancestor by the "ISA" relation). If None, this is the root node.
        """
        self.name = name
        self.props = props if (props is not None) else {}
        self.parent = parent

    def __str__(self):
        abs_name = self.name
        curr_node = self
        while curr_node.parent is not None:
            curr_node = curr_node.parent
            abs_name = curr_node.name + '/' + abs_name

        return abs_name

    def __repr__(self):
        return f'PropositionalTree(name={repr(self.name)}, props={repr(self.props)}, parent={repr(self.parent)})'

    def add_property(self, relation: str, attribute: str):
        if relation.lower() == 'isa':
            raise ValueError('ISA properties cannot be added directly')

        if relation not in self.props.keys():
            self.props[relation] = {attribute}
        else:
            self.props[relation].add(attribute)

    def get_local_related_attributes(self, relation: str):
        """Helper for get_related_attributes that operates only on the current entity - not its ancestors"""

        if relation.lower() == 'isa':
            return {self.name}
        elif relation in self.props.keys():
            return self.props[relation]
        return set()

    def get_related_attributes(self, relation: str):
        """
        Given relation r, returns a set of attributes a such that (r, a) is in the set of properties for
        this entity or one of its ancestors.

        If relation is 'ISA', instead returns the set of names of this entity and all of its ancestors.
        """

        attrs = self.get_local_related_attributes(relation)
        if self.parent is not None:
            attrs |= self.parent.get_related_attributes(relation)

        return attrs

    def get_all_local_attributes(self):
        """Make a set of all attributes by any relation, plus the entity name for 'ISA'"""
        return functools.reduce(operator.or_, self.props.values(), {self.name})

    def get_all_attributes(self):
        attrs = self.get_all_local_attributes()
        if self.parent is not None:
            attrs |= self.parent.get_all_attributes()

        return attrs


def from_xml(xml_path: str):
    """Create a full PTree from an XML file"""

    # accumulate these sets as we parse the XML
    items = set() # leaf items
    relations = {'ISA'}
    attributes = set() # all attributes, including entity names

    def add_node(node_def: ET.Element, parent: Optional[PropositionalTree]):
        """Returns a dict of all nodes created (recursively)"""

        name = node_def.attrib['name']
        new_node = PropositionalTree(name, parent=parent)
        attributes.add(name)

        for key, val in node_def.attrib.items():
            if key == 'name':
                continue

            relations.add(key)
            attrs = val.split(' ')
            for attr in attrs:
                attributes.add(attr)
                new_node.add_property(key, attr)

        # parse children
        if len(node_def) == 0:
            items.add(name)

        all_nodes = {name: new_node}
        for subnode_def in node_def:
            all_nodes.update(add_node(subnode_def, parent=new_node))

        return all_nodes

    tree = ET.parse(xml_path).getroot()
    nodes = {}
    for root in tree: # probably just one, but I guess we could have a forest...
        nodes.update(add_node(root, parent=None))

    return {'nodes': nodes, 'items': items, 'relations': relations, 'attributes': attributes}
