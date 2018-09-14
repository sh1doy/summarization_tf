"""Utilities"""
from chainer.dataset import to_device
from chainer import functions as F
import numpy as np
import pydot
from collections import defaultdict
import pickle


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


class TreeLSTMNode:
    def __init__(self, h, c=None, parent=None, children=[], num=0):
        self.h = h
        self.c = c
        self.parent = parent  # TreeLSTMNode
        self.children = children  # list of TreeLSTMNode
        self.num = num


def remove_identifier(root, mark="\"identifier=", replacement="$ID"):
    """remove identifier of all nodes"""
    if mark in root.label:
        root.label = replacement
    for child in root.children:
        remove_identifier(child)
    return(root)


def print_traverse(root, indent=0):
    """print tree structure"""
    print(" " * indent + root.label)
    for child in root.children:
        print_traverse(child, indent + 2)


def traverse(root):
    """traverse all nodes"""
    res = [root]
    for child in root.children:
        res = res + traverse(child)
    return(res)


def traverse_leaf(root):
    """traverse all leafs"""
    res = []
    for node in traverse(root):
        if node.children == []:
            res.append(node)
    return(res)


def traverse_label(root):
    """return list of tokens"""
    li = [root.label]
    for child in root.children:
        li += traverse_label(child)
    return(li)


def traverse_leaf_label(root):
    """traverse all leafs"""
    res = []
    for node in traverse(root):
        if node.children == []:
            res.append(node.label)
    return(res)


def partial_traverse(root, kernel_depth, depth=0,
                     children=[], depthes=[], left=[]):
    """indice start from 0 and counts do from 1"""
    children.append(root.num)
    depthes.append(depth)
    if root.parent is None:
        left.append(1.)
    else:
        num_sibs = len(root.parent.children)
        if num_sibs == 1:
            left.append(1.)
        else:
            left.append(
                1 - (root.parent.children.index(root) / (num_sibs - 1)))

    if depth < kernel_depth - 1:
        for child in root.children:
            res = partial_traverse(child, kernel_depth,
                                   depth + 1, children, depthes, left)
            children, depthes, left = res

    return(children, depthes, left)


def read_pickle(path):
    return pickle.load(open(path, "rb"))


def consult_tree(root, dic):
    nodes = traverse(root)
    for n in nodes:
        n.label = dic[n.label]
    return nodes[0]


def to_device0(x):
    return(to_device(0, x))


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


def get_sequence_mask(xs, xp):
    x_len = np.array([len(x) for x in xs], "int32")
    mask = xp.tile(xp.arange(x_len.max()).reshape(1, -1), (x_len.shape[0], 1))
    mask = mask < xp.array(x_len).reshape(-1, 1)
    return mask


def get_tree_from_dot_file(path):
    graph = pydot.graph_from_dot_file(path)[0]
    nodes = graph.get_nodes()
    edges = graph.get_edges()
    dic = {n.get_name(): Node(label=n.get_label(), parent=None, children=[]) for n in nodes}
    for e in edges:
        dic[e.get_source()].children.append(dic[e.get_destination()])
        dic[e.get_destination()].parent = dic[e.get_source()]
    return dic["n0"]


def temp_escaper(path):
    text = open(path, "r").read()
    escape_seq = "value=\'\"\'"
    after = "value=\\\'\\\"\\\'"
    if escape_seq in text:
        text = text.replace(escape_seq, after)
    open(path, "w").write(text)


def depth_split(root, depth=0):
    '''
    root: Node or LSTMNode
    return: dict
    '''
    res = defaultdict(list)
    res[depth].append(root)
    for child in root.children:
        for k, v in depth_split(child, depth + 1).items():
            res[k] += v
    return res


def depth_split_batch(roots):
    '''
    roots: list of Node
    return: dict
    '''
    res = defaultdict(list)
    for root in roots:
        for k, v in depth_split(root).items():
            res[k] += v
    return res


def sequence_apply(func, xs):
    '''
    xs: list of [any, dim]
    return: list of func([any, dim])
    '''
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = func(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs
