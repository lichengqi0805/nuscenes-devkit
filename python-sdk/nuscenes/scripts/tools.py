import numpy as np
import vispy
import vispy.io
import vispy.scene as scene
from vispy import app
import threading
from queue import Queue
import zmq
import sys
sys.path.append('..')
sys.path.append('.')
import threading
from loguru import logger
from queue import Queue
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numba
import os
import torch

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

def get_corners(bbox, ret_center=False):
    '''
    bbox format: [x,y,z,l,w,h,yaw]
    coordinate frame:
    +z
    |
    |
    ----- +y
    \
     \
      +x

    Corner numbering::
         5------4
         |\     |\
         | \    | \
         6--\---7  \
          \  \   \  \
     l     \  1------0    h
      e     \ |    \ |    e
       n     \|     \|    i
        g     2------3    g
         t      width     h
          h               t

    First four corners are the ones facing front.
    The last four are the ones facing back.
    '''
    l, w, h = bbox[3:6]
    #                      front           back
    xs = l/2 * np.array([1, 1, 1, 1] + [-1,-1,-1,-1])
    ys = w/2 * np.array([1,-1,-1, 1] * 2)
    zs = h/2 * np.array([1, 1,-1,-1] * 2)
    pts = np.vstack([xs, ys, zs])       # (3, 8)

    center = bbox[:3]
    yaw = bbox[6]
    R = Rotation.from_euler('z', yaw).as_matrix()   # (3, 3)
    pts = (R @ pts).T + center
    if ret_center == True:
        return pts, center
    return pts

def get_visual_lines(bbox, color='r', width=1):
    from vispy.scene import visuals

    pts = get_corners(bbox)
    connect = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # front
        [4, 5], [5, 6], [6, 7], [7, 4],  # back
        [0, 4], [1, 5], [2, 6], [3, 7],  # side
        [0, 2], [1, 3], # front cross
    ])
    lines = visuals.Line(pos=pts, connect=connect, color=color, width=width,
                         antialias=True, method='gl')
    return lines

def get_visual_arrows(bbox, color='g', width=1):
    from vispy.scene import visuals
    center = bbox[:3]
    pts = get_corners(bbox)
    front_center = (pts[0] + pts[1]) / 2
    connect = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # front
        [4, 5], [5, 6], [6, 7], [7, 4],  # back
        [0, 4], [1, 5], [2, 6], [3, 7],  # side
        [0, 2], [1, 3], # front cross
    ])
    return visuals.Line.arrow.ArrowVisual(pos=pts, connect=connect, color=color, width=width, 
                        method='gl', antialias=True)
def pc_in_box(box, pc, box_scaling=1, mask=False):
    center_x, center_y, center_z, length, width, height = box[0:6]
    yaw = box[6]
    if mask == False:
        return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)
    else:
        return pc_in_box_inner_mask(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


@numba.njit
def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    ndims = pc.shape[1]
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos +
                    (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin +
                    (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], ndims), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


@numba.njit
def pc_in_box_inner_mask(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos +
                    (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin +
                    (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    return indices

@numba.njit
def downsample(points, voxel_size=0.05):
    sample_dict = dict()
    for i in range(points.shape[0]):
        point_coord = np.floor(points[i] / voxel_size)
        sample_dict[(int(point_coord[0]), int(point_coord[1]), int(point_coord[2]))] = True
    res = np.zeros((len(sample_dict), 3), dtype=np.float32)
    idx = 0
    for k, v in sample_dict.items():
        res[idx, 0] = k[0] * voxel_size + voxel_size / 2
        res[idx, 1] = k[1] * voxel_size + voxel_size / 2
        res[idx, 2] = k[2] * voxel_size + voxel_size / 2
        idx += 1
    return res

def pca(points):
    '''
    Args
    -----
        points: np.ndarray, shape (N, 3)
    Return
    ------
        mu, covariance, eigen_value, eigen_vector
    '''
    pts_num = points.shape[0]
    mu = np.mean(points, axis=0)
    normalized_points = points - mu
    covariance = (1/pts_num - 1) * normalized_points.T @ normalized_points
    eigen_vals, eigen_vec = np.linalg.eig(covariance)
    return mu, covariance, eigen_vals, eigen_vec

COLORS = plt.get_cmap('Paired').colors

def get_color(i):
    return COLORS[i % len(COLORS)] + (1,)


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon=True
        # logger.debug(f'{fn} start a thread')
        thread.start()
    return wrapper

class VisMessenger:
    def __init__(self, port='19999'):
        self.socket = zmq.Context().socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")

    def array_to_msg(self, array):
        _shape = np.array(array.shape, dtype=np.int32)
        return [array.dtype.name.encode(),
                _shape.tostring(),
                array.tostring()]

    def send_numpy(self, array, nptype='pc', name=''):
        '''
        type: "pc" or "bbox"
        '''
        assert isinstance(array, np.ndarray)
        # _msg = self.array_to_msg(array)
        head_msg = str(nptype) + '/' + str(name)
        self.socket.send_string(head_msg, zmq.SNDMORE)
        self.socket.send_pyobj(array)


class VisCanvas(scene.SceneCanvas):

    PTS_OPT = dict(alpha=0.8, spherical=True)



    def __init__(self, pc_size=5.0, pc_font_size=10.0, label_font_size=5, server=False):
        super().__init__(keys=None, size=(1000, 800), title='PointCloud Canvas', show=True)
        self.unfreeze()

        self.view = self.central_widget.add_view()
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)

        self.view.camera = 'turntable'
        self.view.camera.center = [0, 0, 0]
        self.view.camera.distance = 10
        self.view.camera.elevation = 30
        self.view.camera.azimuth = -90
        # Press Shift to translate camera view
        self.view.camera.translate_speed = 25
        # GUI
        self.pc_points_size = pc_size
        self.pc_font_size = pc_font_size
        self.label_text_size = label_font_size * 100
        # Visual Elements
        self._pc = dict()
        self._bbox = dict()
        self._bbox_color = list()
        self._texts = list()
        self.numpy_queue = Queue()
        if server == True:
            self._init_server_mode()


    def _init_server_mode(self):
        self.data_thread = threading.Thread(target=self.recv_data, daemon=True)
        self.data_thread.start()
        pass

    def recv_data(self):
        print('thread start')
        socket = zmq.Context().socket(zmq.SUB)  
        socket.setsockopt_string(zmq.SUBSCRIBE, '')  
        port = '19999'
        socket.connect("tcp://localhost:%s" % port)
        while True:
            topic = socket.recv_string()
            msg = socket.recv_pyobj()
            print(topic)
            self.numpy_queue.put([topic, msg])
            event = app.KeyEvent(type='key_press', text='*')
            self.events.key_press(event)

    def _init_bbox(self):
        self.bbox_all_points = np.empty((0,3))
        self.bbox_all_connect = np.empty((0,2))
        self.bbox_all_colors = np.empty((0,4))
        self.connect_template = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # side
            [0, 2], [1, 3], # front cross
        ])
        self.all_labels_text = []
        self.all_labels_pos = []
    
    def add_pc(self, pc, color=None, name='pointcloud', size=None):
        '''
        Add Point Cloud to canvas
        Args:
        ------
            pc : numpy.ndarray
            color: numpy.ndarray
        '''
        if color is None:
            color = 'w'
        if size is not None:
            self.pc_points_size = size
        if name == '':
            name = 'pointcloud'
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        if hasattr(self, name):
            self.__dict__[name].set_data(pos=pc, face_color=color, edge_color=None, size=self.pc_points_size)
        else:
            setattr(self, name , scene.visuals.Markers(pos=pc, face_color=color, edge_color=None, size=self.pc_points_size, parent=self.view.scene, **self.PTS_OPT ))

        # Add visual element to dict for further modification
        self._pc[name] = self.__dict__[name]
        return self.__dict__[name]

    @threaded
    def add_mesh(self, vertices, faces, name='mesh'):
        from vispy.visuals.filters import ShadingFilter, WireframeFilter
        mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, 
                                                 parent=self.view.scene)
        wireframe_filter = WireframeFilter(width=0.5)
        # Note: For convenience, this `ShadingFilter` would be created automatically by
        # the `MeshVisual with, e.g. `mesh = MeshVisual(..., shading='smooth')`. It is
        # created manually here for demonstration purposes.
        shading_filter = ShadingFilter(shininess=100)
        # The wireframe filter is attached before the shading filter otherwise the
        # wireframe is not shaded.
        mesh.attach(wireframe_filter)
        mesh.attach(shading_filter)
        self.__dict__[name] = mesh
        
    
    def add_bbox(self, bbox, color=(1,1,1,1), name='bbox', width=None, label_pos='center'):
        '''
        Add Bbox to canvas
        -------
        bbox: (n, 9) numpy ndarray
        color: (n, 4) numpy ndarray
        name: bbox name in canvas
        width: bounding box line width
        label_pos: 'center' or other
        '''
        self._init_bbox()
        self._bbox_color.append(color)
        if width is None:
            width = 4
        if name == '':
            name = 'bbox'
        try:
            assert len(color) == 4
        except Exception as e:
            print(e)
            return
        pts, center = get_corners(bbox, ret_center=True)
        text_pos = center if label_pos == 'center' else pts[0]
        self._bbox[name] = [pts, center, bbox, text_pos.tolist()]

        for curr_idx, bbox_name in enumerate(self._bbox):
            self.bbox_all_points = np.append(self.bbox_all_points, self._bbox[bbox_name][0], axis=0)
            self.bbox_all_connect = np.append(self.bbox_all_connect, self.connect_template + curr_idx * 8, axis=0)
            color = np.asanyarray(self._bbox_color[curr_idx]).reshape(1, 4)
            curr_color = np.repeat(color, 8, axis=0)
            self.bbox_all_colors = np.append(self.bbox_all_colors, curr_color, axis=0)
            self.all_labels_text.append(bbox_name)
            self.all_labels_pos.append(self._bbox[bbox_name][3])

        if hasattr(self, 'bbox'):
            self.bbox.set_data(pos=self.bbox_all_points, color=self.bbox_all_colors, connect=self.bbox_all_connect, width=width)
        else:
            self.bbox = scene.visuals.Line(pos=self.bbox_all_points, connect=self.bbox_all_connect,
                                           color=self.bbox_all_colors, width=width, parent=self.view.scene)


    def add_text(self, text, pos, color='white', font_size=20):
        text_info = dict(
            text=text,
            pos=pos,
            color=color,
            font_size=font_size
        )
        self._text.append(text_info)
                
    def show_label(self):
        if hasattr(self, "sot_bbox_text"):
            self.sot_bbox_text.pos = self.all_labels_pos
            self.sot_bbox_text.text = self.all_labels_text
            self.sot_bbox_text.font_size = self.label_text_size
            self.sot_bbox_text.color = (1, 0, 1, 1)
            self.sot_bbox_text.parent = self.view.scene
        else:
            self.sot_bbox_text = scene.visuals.Text(pos=self.all_labels_pos, color=(1, 0, 1, 1), text=self.all_labels_text,
                                                        font_size=self.label_text_size, bold=False, parent=self.view.scene)


    def clear_bbox(self):
        self._bbox = dict()
        self._bbox_color = []
        

    def color_pc_in_bbox(self):
        for bbox_name in self._bbox:
            bbox = self._bbox[bbox_name][2]
            pc_in_bbox_indices = pc_in_box(bbox, self.points, mask=True)
            self.points_color[pc_in_bbox_indices] = np.array([1, 1, 0, 1])
        self.world_points.set_data(
            pos=self.points, edge_color=self.points_color, face_color=self.points_color, 
            size=self.pc_points_size)
        self.update()

    def run(self):
        vispy.app.run()

    def render_img(self, dir='/home/nio/workspace/sot_new/bboxvis/example.png',
                   cam_center=(0,0,0), cam_distance=10):
        self.view.camera.center = cam_center
        self.view.camera.distance = cam_distance
        img = self.render()
        vispy.io.write_png(dir, img)
    
    def on_key_press(self, event):
        self.view_center = list(self.view.camera.center)
        if (event.text == '*'):
            this_data = self.numpy_queue.get()
            head_msg = this_data[0]
            nptype, name = head_msg.split('/')
            if nptype == 'pc':
                self.add_pc(this_data[1], name=name)
                logger.info('Received a Point Cloud')
            elif nptype == 'bbox':
                self.add_bbox(this_data[1])
                logger.info('Received a Bbox')
        if(event.text == 'w' or event.text == 'W'):
            dx, dy = self.get_cam_delta()
            self.view_center[0] += dx
            self.view_center[1] += dy
            self.view.camera.center = self.view_center

        if(event.text == 's' or event.text == 'S'):
            dx, dy = self.get_cam_delta()
            self.view_center[0] -= dx
            self.view_center[1] -= dy
            self.view.camera.center = self.view_center

        if(event.text == 'a' or event.text == 'A'):
            dx, dy = self.get_cam_delta()
            self.view_center[0] -= dy
            self.view_center[1] += dx
            self.view.camera.center = self.view_center

        if(event.text == 'd' or event.text == 'D'):
            dx, dy = self.get_cam_delta()
            self.view_center[0] += dy
            self.view_center[1] -= dx
            self.view.camera.center = self.view_center

        if(event.key == 'up'):
            self.view_center[2] += 1
            self.view.camera.center = self.view_center

        if(event.key == 'down'):
            self.view_center[2] -= 1
            self.view.camera.center = self.view_center

        if(event.text == 'c' or event.text == 'C'):
            # Centered
            self.view_center[0] = 0
            self.view_center[1] = 0
            self.view_center[2] = 0
            self.view.camera.center = self.view_center
    
    def get_cam_delta(self):
        theta = self.view.camera.azimuth
        dx = -np.sin(theta * np.pi / 180)
        dy = np.cos(theta * np.pi / 180)
        return dx, dy