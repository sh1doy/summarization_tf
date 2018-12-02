import sys
from tqdm import tqdm
from glob import glob
from utils import Node, traverse_label, traverse
import pickle
import os
from joblib import Parallel, delayed
from collections import Counter
import re
from os.path import abspath
import nltk


def parse(path):
    with open(path, "r") as f:
        num_objects = f.readline()
        nodes = [Node(num=i, children=[]) for i in range(int(num_objects))]
        for i in range(int(num_objects)):
            label = " ".join(f.readline().split(" ")[1:])[:-1]
            nodes[i].label = label
        while 1:
            line = f.readline()
            if line == "\n":
                break
            p, c = map(int, line.split(" "))
            nodes[p].children.append(nodes[c])
            nodes[c].parent = nodes[p]
        nl = f.readline()[:-1]
    return nodes[0], nl


def is_invalid_com(s):
    return s[:2] == "/*" and len(s) > 1


def is_invalid_seq(s):
    return len(s) < 4


def get_method_name(root):
    for c in root.children:
        if c.label == "name (SimpleName)":
            return c.children[0].label[12:-1]


def is_invalid_tree(root):
    labels = traverse_label(root)
    if root.label == 'root (ConstructorDeclaration)':
        return True
    if len(labels) >= 100:
        return True
    method_name = get_method_name(root)
    for word in ["test", "Test", "set", "Set", "get", "Get"]:
        if method_name[:len(word)] == word:
            return True
    return False


def clean_nl(s):
    if s[-1] == ".":
        s = s[:-1]
    s = s.split(". ")[0]
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    return s


def tokenize(s):
    return ["<s>"] + nltk.word_tokenize(s) + ["</s>"]


def parse_dir(path_to_dir):
    files = sorted(glob(path_to_dir + "/*"))
    set_name = path_to_dir.split("/")[-1]

    nls = {}
    skip = 0

    for file in tqdm(files, "parsing {}".format(path_to_dir)):
        tree, nl = parse(file)
        nl = clean_nl(nl)
        if is_invalid_com(nl):
            skip += 1
            continue
        if is_invalid_tree(tree):
            skip += 1
            continue
        number = int(file.split("/")[-1])
        seq = tokenize(nl)
        if is_invalid_seq(seq):
            skip += 1
            continue
        nls[abspath("./dataset/tree/" + set_name + "/" + str(number))] = seq
        with open("./dataset/tree_raw/" + set_name + "/" + str(number), "wb", 1) as f:
            pickle.dump(tree, f)

    print("{} files skipped".format(skip))

    if set_name == "train":
        vocab = Counter([x for l in nls.values() for x in l])
        nl_i2w = {i: w for i, w in enumerate(
            ["<PAD>", "<UNK>"] + sorted([x[0] for x in vocab.most_common(30000)]))}
        nl_w2i = {w: i for i, w in enumerate(
            ["<PAD>", "<UNK>"] + sorted([x[0] for x in vocab.most_common(30000)]))}
        pickle.dump(nl_i2w, open("./dataset/nl_i2w.pkl", "wb"))
        pickle.dump(nl_w2i, open("./dataset/nl_w2i.pkl", "wb"))

    return nls


def pickling():
    args = sys.argv

    if len(args) <= 1:
        raise Exception("(usage) $ python dataset.py [dir]")

    data_dir = args[1]

    dirs = [
        "dataset",
        "dataset/tree_raw",
        "dataset/tree_raw/train",
        "dataset/tree_raw/valid",
        "dataset/tree_raw/test",
        "dataset/nl"
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    for path in [data_dir + "/" + s for s in ["train", "valid", "test"]]:
        set_name = path.split("/")[-1]
        nl = parse_dir(path)
        with open("./dataset/nl/" + set_name + ".pkl", "wb", 1) as f:
            pickle.dump(nl, f)


def isnum(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def get_labels(path):
    tree = pickle.load(open(path, "rb"))
    return traverse_label(tree)


def get_bracket(s):
    if "value=" == s[:6] or "identifier=" in s[:11]:
        return None
    p = "\(.+?\)"
    res = re.findall(p, s)
    if len(res) == 1:
        return res[0]
    return s


def get_identifier(s):
    if "identifier=" == s[:11]:
        return "SimpleName_" + s[11:]
    else:
        return None


def is_SimpleName(s):
    return "SimpleName_" == s[:11]


def get_values(s):
    if "value=" == s[:6]:
        return "Value_" + s[6:]
    else:
        return None


def is_value(s):
    return "Value_" == s[:6]


def make_dict():
    labels = Parallel(n_jobs=-1)(delayed(get_labels)(p) for p in tqdm(
        glob("./dataset/tree_raw/train/*"), "reading all labels"))
    labels = [l for s in labels for l in s]

    non_terminals = set(
        [get_bracket(x) for x in tqdm(
            list(set(labels)), "collect non-tarminals")]) - set([None, "(SimpleName)"])
    non_terminals = sorted(list(non_terminals))

    ids = Counter(
        [y for y in [get_identifier(x) for x in tqdm(
            labels, "collect identifiers")] if y is not None])
    ids_list = [x[0] for x in ids.most_common(30000)]

    values = Counter(
        [y for y in [get_values(x) for x in tqdm(
            labels, "collect values")] if y is not None])
    values_list = [x[0] for x in values.most_common(1000)]

    vocab = ["<UNK>", "SimpleName_<UNK>", "Value_<NUM>", "Value_<STR>"]
    vocab += non_terminals + ids_list + values_list + ["(", ")"]

    code_i2w = {i: w for i, w in enumerate(vocab)}
    code_w2i = {w: i for i, w in enumerate(vocab)}

    pickle.dump(code_i2w, open("./dataset/code_i2w.pkl", "wb"))
    pickle.dump(code_w2i, open("./dataset/code_w2i.pkl", "wb"))


def remove_SimpleName(root):
    for node in traverse(root):
        if "=" not in node.label and "(SimpleName)" in node.label:
            if node.children[0].label[:11] != "identifier=":
                raise Exception("ERROR!")
            node.label = "SimpleName_" + node.children[0].label[11:]
            node.children = []
        elif node.label[:11] == "identifier=":
            node.label = "SimpleName_" + node.label[11:]
        elif node.label[:6] == "value=":
            node.label = "Value_" + node.label[6:]

    return root


def modifier(root, dic):
    for node in traverse(root):
        if is_SimpleName(node.label):
            if node.label not in dic:
                node.label = "SimpleName_<UNK>"
        elif is_value(node.label):
            if node.label not in dic:
                if isnum(node.label):
                    node.label = "Value_<NUM>"
                else:
                    node.label = "Value_<STR>"
        else:
            node.label = get_bracket(node.label)
        if node.label not in dic:
            raise Exception("Unknown word", node.label)

    return root


def rebuild_tree(path, dst, dic):
    root = pickle.load(open(path, "rb"))
    root = remove_SimpleName(root)
    root = modifier(root, dic)
    pickle.dump(root, open(dst, "wb"), 1)


def preprocess_trees():

    dirs = [
        "./dataset",
        "./dataset/tree",
        "./dataset/tree/train",
        "./dataset/tree/valid",
        "./dataset/tree/test",
        "./dataset/nl"
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    sets_name = [
        "./dataset/tree_raw/train/*",
        "./dataset/tree_raw/valid/*",
        "./dataset/tree_raw/test/*"
    ]

    dic = set(pickle.load(open("./dataset/code_i2w.pkl", "rb")).values())

    for sets in sets_name:
        files = sorted(list(glob(sets)))
        dst = [x.replace("tree_raw", "tree") for x in files]
        Parallel(n_jobs=-1)(
            delayed(rebuild_tree)(p, d, dic) for p, d in tqdm(
                list(zip(files, dst)), "preprocessing {}".format(sets)))


if __name__ == "__main__":
    nltk.download('punkt')
    sys.setrecursionlimit(10000)
    pickling()
    make_dict()
    preprocess_trees()
