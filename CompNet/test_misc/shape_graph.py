"""Utility for test """
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import networkx as nx
from loguru import logger
import numpy as np
import random
from tqdm import tqdm

from CompNet.utils.io_utils import resize_img, read_img
from CompNet.utils.vis_utils import get_n_colors
from CompNet.test_misc.common import match_gt_size


def init_shape_graph(data_dir,
                     shape_list,
                     shape_graph_dir, ):
    logger.info(f'Initialize and save {len(shape_list)} shape graph to {shape_graph_dir}')
    for i in tqdm(range(len(shape_list))):
        obj = shape_list[i]
        obj_fn = Path(data_dir) / obj / f'{obj}.npy'
        out_fn = shape_graph_dir / f'{obj}.gpickle'
        t = ShapeGraph(obj_fn=obj_fn,
                       mask_prefix='partmask')
        nx.write_gpickle(t, out_fn)
    logger.info(f'Finish initalization shape graph.\n')


def load_img(img_fn, flags=1, IMG_HEIGHT=256):
    logger.debug(f'Load img {img_fn} with flag {flags}')
    img = read_img(str(img_fn), flags)
    if img.shape[0] != IMG_HEIGHT:
        img = resize_img(img, IMG_HEIGHT, IMG_HEIGHT)
    return img


class ShapeGraph(nx.Graph):
    """ShapeGraph maintain shape part info"""
    def __init__(self, obj_fn, mask_prefix='partmask'):
        super(ShapeGraph, self).__init__()
        self.mask_prefix = mask_prefix
        self.build_shape(obj_fn)

    def log_graph(self):
        for n in self.nodes:
            logger.debug(f"Node {n}, pred box: {self.nodes[n]['pred_box']}")

    def build_shape(self, obj_fn):
        logger.debug(f'Build shape graph from {obj_fn}')
        obj_data = np.load(obj_fn, allow_pickle=True).item()
        render_dir = Path(obj_fn).parent

        # Graph attributes
        curr_obj = Path(obj_fn).stem
        self.name = curr_obj  # shape name

        img_fn = render_dir/'img.png'
        self.img = load_img(img_fn)  # shape image

        # vis_nodes
        vis_nodes = obj_data['vis_parts']
        obj_nodes = obj_data['nodes']
        obj_edges = obj_data['edges']
        adj_edges = [x for x in obj_edges if 'ADJ' in x]

        # add edge between adjacent nodes
        for (edge_type, id1, id2) in adj_edges:
            if id1 in vis_nodes and id2 in vis_nodes:
                self.add_edge(id1, id2)

        # add each part information
        color_list = get_n_colors(self.number_of_nodes())
        for i, node in enumerate(self.nodes):
            part_mask_fn = render_dir/f'{self.mask_prefix}_{node}.png'
            part_mask = load_img(part_mask_fn, 0)
            part_box = np.array(obj_nodes[node]['box']).copy()

            # update node attribute
            self.add_node(node,
                          visited=False,
                          gt_box=part_box,
                          mask_img=part_mask,
                          pred_box=np.zeros(12,),
                          tp=[],
                          color=np.array(color_list[i]))

    def init_start_node(self, s=None, choice='max', gt_option=None):
        """
        Args:
            s: start node, if given s, use s as start node
            choice: start node choice, either random or choosing part with largest mask
            gt_option: use gt size/rotation/center or not

        Returns:

        """
        # initialization nodes visited label and next steps
        for node in self.nodes:
            self.nodes[node]['visited'] = False
        self.next_steps = []

        # find start node
        if s:
            start_node = s
        else:
            if choice == 'max':
                logger.debug(f'Choose start node using {choice}')
                start_node = list(self.nodes)[0]
                max_mask = np.sum(self.nodes[start_node]['mask_img'])
                for n in self.nodes:
                    if np.sum(self.nodes[n]['mask_img']) > max_mask:
                        start_node = n
                        max_mask = np.sum(self.nodes[n]['mask_img'])  # update max_mask
            elif choice == 'random':
                logger.debug(f'Choose start node using {choice}')
                start_node = random.choice(list(self.nodes))

        # upstaae start node information
        logger.debug(f'Choose start node {start_node}')
        self.nodes[start_node]['visited'] = True  # visit label

        # update current node information
        if gt_option:
            logger.info(f'Start node use gt {gt_option} as prediction {gt_option}!')
            if gt_option == 'rot':
                self.nodes[start_node]['pred_box'][6:] = self.nodes[start_node]['gt_box'][6:]
            elif gt_option == 'size':
                # matching gt_size
                self.nodes[start_node]['pred_box'][3:6] = match_gt_size(self.nodes[start_node]['pred_box'],
                                                                        self.nodes[start_node]['gt_box'])
            elif gt_option == 'center':
                self.nodes[start_node]['pred_box'][:3] = self.nodes[start_node]['gt_box'][:3]
                logger.debug(f"Start node {start_node} use gt center pred box: {self.nodes[start_node]['pred_box']}")
                logger.debug(f"Start node {start_node} gt box: {self.nodes[start_node]['gt_box']}")
            elif gt_option == 'all':
                self.nodes[start_node]['pred_box'] = self.nodes[start_node]['gt_box'].copy()
            elif gt_option == 'none':
                pass
            else:
                raise ValueError(f'Not support initialization mode {gt_option}')

        self.sf_update_graph(start_node)  # update next choice list

        return start_node

    def get_step_choices(self, option='all'):
        """Sequential generation, get next step choice from all next_steps"""
        if option == 'all':
            return self.next_steps
        elif option == 'random':
            return [random.choice(self.next_steps)]
        elif option == 'first':
            return [self.next_steps[0]]
        elif option == 'max':
            # sort by mask image
            tp_mask_area_list = []
            for i, (u, v) in enumerate(self.next_steps):
                tp_mask_area_list.append((np.sum(self.nodes[u]['mask_img']),
                                          np.sum(self.nodes[v]['mask_img']),
                                          i))
            tp_mask_area_list.sort(reverse=True)
            i = tp_mask_area_list[0][2]
            return [self.next_steps[i]]
        elif option == 'maxvolume':
            tp_mask_area_list = []
            for i, (u, v) in enumerate(self.next_steps):
                size_u = self.nodes[u]['pred_box'][3:6]
                size_v = self.nodes[v]['pred_box'][3:6]
                volume_u = size_u[0] * size_u[1] * size_u[2]
                volume_v = size_v[0] * size_v[1] * size_v[2]
                tp_mask_area_list.append((volume_u,
                                          volume_v,
                                          i))
            tp_mask_area_list.sort(reverse=True)
            i = tp_mask_area_list[0][2]
            for item in tp_mask_area_list:
                logger.debug(f'sorted {item}')
            logger.debug(f'choose id {i}, {self.next_steps[i]}')
            return [self.next_steps[i]]
        else:
            raise ValueError(f'Not implement step choice {option}.')

    def get_all_pairs(self, option='all'):
        part_list = list(self.nodes)
        num_part = len(part_list)
        all_tp_list = []
        for i in range(num_part):
            for j in range(i+1, num_part):
                all_tp_list.append([part_list[i], part_list[j]])
        return all_tp_list

    def sf_update_graph(self, u):
        """visit node u, remove edges to u and add edges from u to unviist nodes"""
        self.nodes[u]['visited'] = True
        self.next_steps = [x for x in self.next_steps if x[1] != u]
        tmp_steps = [(u, v) for v in self.neighbors(u)
                     if not self.nodes[v]['visited']]
        self.next_steps += tmp_steps
        logger.debug(f'Update next step choices: {self.next_steps}')

    def visit_all_nodes(self):
        """Judge to stop sequential generation or not."""
        if len(self.next_steps) == 0:
            logger.debug(f'Stop due to no more available steps')
            return True

        for u in self.nodes:
            if not self.nodes[u]['visited']:
                return False
        logger.debug(f'Stop due to all nodes are visited')
        return True

    def pred_use_gt(self, choice='rot'):
        if choice == 'rot':
            for pid in self.nodes:
                self.nodes[pid]['pred_box'][6:] = self.nodes[pid]['gt_box'][6:]
        elif choice == 'size':
            for pid in self.nodes:
                self.nodes[pid]['pred_box'][3:6] = match_gt_size(self.nodes[pid]['pred_box'],
                                                                 self.nodes[pid]['gt_box'])
                # self.nodes[pid]['pred_box'][3:6] = self.nodes[pid]['gt_box'][3:6]
        elif choice == 'center':
            for pid in self.nodes:
                self.nodes[pid]['pred_box'][:3] = self.nodes[pid]['gt_box'][:3]
        else:
            raise ValueError(f'Error, prediction use gt {choice}')

    def output_npy(self,
                   out_pred_fn):
        box_dict = {}
        box_list = []
        for pid in self.nodes:
            box_dict[pid] = self.nodes[pid]['pred_box']
            box_list.append(self.nodes[pid]['pred_box'])
        np.save(out_pred_fn, box_dict)


class EqualSizeGraph(nx.Graph):

    def get_number_of_components(self):
        visited = {}
        for n in self.nodes:
            visited[n] = False

        components = []

        count = 0
        for n in self.nodes:
            if not visited[n]:
                count += 1
                component_nodes = list(nx.dfs_preorder_nodes(self, source=n))
                components.append(component_nodes)
                for x in component_nodes:
                    visited[x] = True

        return count, components

    def log_graph(self):
        logger.info(f'Graph information')
        num_comp, comp_nodes = self.get_number_of_components()
        logger.info(f'\n{nx.info(self)} '
                    f'\nNumber of components: {num_comp}')
        for i, item in enumerate(comp_nodes):
            logger.info(f'Component {i}, nodes {item}')
