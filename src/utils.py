'''
utils.py in deepgeom

author  : cfeng; Yuqiong Li
created : 1/27/18 1:15 AM
'''

import os
import sys
import argparse
import errno
from open3d import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, proj3d
import math


def check_exist_or_mkdirs(path):
    '''thread-safe mkdirs if not exist'''
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def check_exist_or_remove(path):
    '''thread-safe remove if exist'''
    if os.path.exists(path):
        os.remove(path)


def vis_pts(pts, clr, cmap):
    """
    visualize points
    :param pts: points
    :param clr: color
    :param cmap:
    :return:
    """
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)

    ax.set_alpha(0)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if clr is None:
        M = ax.get_proj()
        _,_,clr = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min())

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ax.scatter(
        pts[:,0],pts[:,1],pts[:,2],
        c=clr,
        zdir='x',
        s=20,
        cmap=cmap,
        edgecolors='k'
    )
    return fig


def count_parameter_num(params):
    cnt = 0
    for p in params:
        cnt += np.prod(p.size())
    return cnt


def make_box():
    """
    function to make grids on a 3D unit box
    @param lower: lower bound
    @param upper: upper bound
    @param num: number of points on an axis. Default 18
    rvalue: 2D numpy array of dim0 = num**2*6, num1 = 3. Meaning a point cloud
    """
    lower = -0.5
    upper = 0.5
    num = 18
    a = np.linspace(lower, upper, num)
    b = np.linspace(lower, upper, num)
    grid = np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))])

    c1 = np.repeat(0.5, len(grid))
    c1 = np.reshape(c1, (len(c1), -1))
    c2 = np.repeat(-0.5, len(grid))
    c2 = np.reshape(c2, (len(c2), -1))

    up = np.hstack((grid, c1))  # upper face, z == 0.5
    low = np.hstack((grid, c2))  # lower face, z == -0.5
    front = up[:, [0, 2, 1]]  # front face, y == 0.5
    back = low[:, [0, 2, 1]]  # back face, y == -0.5
    right = up[:, [2, 0, 1]]  # right face, x == 0.5
    left = low[:, [2, 0, 1]]  # left face, x == -0.5

    six_faces = np.vstack((front, back, right, left, up, low))
    return six_faces


def make_cylinder():
    """
    function to make a grid from a cyliner centered at (0, 0, 0). The cyliner's radius is 1, height is 0.5
    Method:
    1) the surrounding surface is 4 times the area of the upper and lower cicle. So we sample 4 times more points from it
    2) to match with the box, total number of points is 1944
    3) for the upper and lower surface, points are sampled with fixed degree and fixed distance along the radius
    4) for the middle surface, points are sampled along fixed lines along the height
    """
    # make the upper and lower face, which is not inclusive of the boundary points
    theta = 10  # dimension
    n = 9  # number of points for every radius
    r = 0.5
    radius_all = np.linspace(0, 0.5, n + 2)[1:10]  # radius of sub-circles
    res = []
    for i, theta in enumerate(range(0, 360, 10)):
        x = math.sin(theta)
        y = math.cos(theta)
        for r in radius_all:
            res.append([r * x, r * y])
    # add z axis
    z = np.reshape(np.repeat(0.5, len(res)), (len(res), -1))
    upper = np.hstack((np.array(res), z))  # upper face
    z = np.reshape(np.repeat(-0.5, len(res)), (len(res), -1))
    lower = np.hstack((np.array(res), z))  # lower face

    # design of middle layer: theta = 5 degree, with every divide is 18 points including boundaries
    height = np.linspace(-0.5, 0.5, 18)
    res = []
    for theta in range(0, 360, 5):
        x = 0.5 * math.sin(theta)
        y = 0.5 * math.cos(theta)
        for z in height:
            res.append([x, y, z])
    middle = np.array(res)

    cylinder = np.vstack((upper, lower, middle))
    return cylinder


def make_sphere():
    """
    function to sample a grid from a sphere
    """
    theta = np.linspace(0, 360, 36)  # determining x and y
    phi = np.linspace(0, 360, 54)  # determining z

    res = []
    for p in phi:
        z = math.sin(p) * 0.5
        r0 = math.cos(p) * 0.5
        for t in theta:
            x = math.sin(t) * r0
            y = math.cos(t) * r0
            res.append([x, y, z])

    sphere = np.array(res)
    return sphere


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rand_ortho_rotation_matrix():
    k = np.zeros((3,), dtype=int)
    k[np.random.randint(0, 3)] = 1 if np.random.rand() > 0.5 else -1
    K = skew(k)
    all_theta = [0, 90, 180, 270]
    theta = np.deg2rad(all_theta[np.random.randint(0, 4)])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R.astype(int)


def vis_points(points, fname):
    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(points)
    vis.get_render_option().load_from_json('renderopt.json')
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-40)
    vis.run()
    vis.capture_screen_image(fname)
    vis.destroy_window()



def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


class TrainTestMonitor(object):

    def __init__(self, log_dir, plot_loss_max=4., plot_extra=False):
        assert(os.path.exists(log_dir))

        stats_test = np.load(os.path.join(log_dir, 'stats_test.npz'))
        stats_train_running = np.load(os.path.join(log_dir, 'stats_train_running.npz'))

        self.title = os.path.basename(log_dir)
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        plt.title(self.title)

        # Training loss
        iter_loss = stats_train_running['iter_loss']
        self.ax1.plot(iter_loss[:,0], iter_loss[:,1],'-',label='train loss',color='r',linewidth=2)
        self.ax1.set_ylim([0, plot_loss_max])
        self.ax1.set_xlabel('iteration')
        self.ax1.set_ylabel('loss')

        # Test accuracy
        iter_acc = stats_test['iter_acc']
        max_accu_pos = np.argmax(iter_acc[:,1])
        test_label = 'max test accuracy {:.3f} @ {}'.format(iter_acc[max_accu_pos,1],max_accu_pos+1)
        self.ax2.plot(iter_acc[:,0], iter_acc[:,1],'o--',label=test_label,color='b',linewidth=2)
        self.ax2.set_ylabel('accuracy')

        if plot_extra:
            # Training accuracy
            iter_acc = stats_train_running['iter_acc']
            self.ax2.plot(iter_acc[:,0], iter_acc[:,1],'--',label='train accuracy',color='b',linewidth=.8)
            # Test loss
            iter_loss = stats_test['iter_loss']
            self.ax1.plot(iter_loss[:,0], iter_loss[:,1],'--',label='test loss',color='r',linewidth=.8)

        self.ax1.legend(loc='upper left', framealpha=0.8)
        self.ax2.legend(loc='lower right', framealpha=0.8)
        self.fig.show()


"""def show_results(dataset, label_show):
    points_1 = PointCloud()
    points_2 = PointCloud()
        # vis_pts(pointcloud[0], 'b', 'tab20')
        # vis_pts(points[0].cpu().detach(), 'b', 'tab20')
        # plt.show()
        pointcloud = pointcloud[0].cpu().numpy()
        points = points[0].cpu().detach().numpy()
        points_1.points = Vector3dVector(pointcloud)
        points_2.points = Vector3dVector(points)
        # vis_points(points_1, 'ground_truth_%d.png' % idx)
        # vis_points(points_2, 'prediction_%d.png' % idx)
        write_point_cloud('ground_truth_%d.pcd' % idx, points_1)
        write_point_cloud('prediction_%d.pcd' % idx, points_2)
        if idx >= 100:
            break
        # plt.imsave('internal_rgb_%d.png' % idx, rgb[0][:, :, ::-1])
    return 0
"""