import pandas as pd
from dataclasses import dataclass


@dataclass
class Node:
    '''
    Class representing single node in decision tree model
    :left_node: node that is created by splitting this node with
        'yes' in condition
    :right_node: node that is created by splitting this node with
        'no' in condition
    :data: data of the node
    :depth: how far from the root is this node
    :value: only for leaves, mean of the classification values in this node
    :split_condition: feature used to split the node
    :split_condition_value: if the feature is continuous or discreet,
        it represents the value that was used for <= condition
    '''
    left_child: 'Node' = None
    right_child: 'Node' = None
    data: tuple[pd.DataFrame, pd.Series] = None
    depth: int = None
    value: any = None
    split_condition: str = None
    split_condition_value: float = None
