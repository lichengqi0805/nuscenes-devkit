from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, TrackingMetricData
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.color_map import get_colormap
import numpy as np
from tools import VisCanvas
from glob import glob
import vispy
from loguru import logger
from pathlib import Path
import os
from pyquaternion import Quaternion
import json
from loguru import logger
import pickle as pkl
try:
    from wis3d import Wis3D
except ImportError:
    print('Wis3D not found, will not be able to visualize in browser.')

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


class annotator(VisCanvas):
    def __init__(self, nuscene_path, result_path, scene_idx=0):
        self._index = 0
        self.scene_idx = scene_idx
        
        self.version = 'v1.0-trainval'
        
        self.nusc = NuScenes(version=self.version, dataroot=nuscene_path, verbose=True)
        self.result_path = result_path
        print(f'Load prediction from {self.result_path}')
        with open(self.result_path, 'r') as f:
            self.tracking_result = json.load(f)
            
        self.eval_set = 'train'
        logger.info('loading predictions')
        pred_boxes, self.meta = load_prediction(self.result_path, 500, TrackingBox,
                                                verbose=True) # max bboxes per sample: 500
        
        # lidarseg colormap
        self.colormap = get_colormap()

        if os.path.isfile(f'tracks_gt_{self.eval_set}.pkl'):
            with open(f'tracks_gt_{self.eval_set}.pkl', 'rb') as f:
                self.tracks_gt = pkl.load(f)
        else:
            logger.info('load gt bboxes')
            gt_boxes = load_gt(self.nusc, self.eval_set, TrackingBox, verbose=True)
            print(gt_boxes)
            gt_boxes = add_center_dist(self.nusc, gt_boxes)
            self.tracks_gt = create_tracks(gt_boxes, self.nusc, self.eval_set, gt=True)
            
            with open(f'tracks_gt_{self.eval_set}.pkl', 'wb') as f:
                pkl.dump(dict(self.tracks_gt), f)
        
        try:
            pred_boxes = add_center_dist(self.nusc, pred_boxes)
            self.tracks_pred = create_tracks(pred_boxes, self.nusc, self.eval_set, gt=False)
        except Exception as e:
            logger.error(e)
        
        self.scene_tokens = sorted(list(self.tracks_gt.keys()))
        logger.info(f'number of scene tokens: {len(self.scene_tokens)}')
        
        # self.scene_token = self.scene_tokens[self.scene_idx]
        print(f'Non empty scenes: {self.non_empty_scenes}')
        self.scene_token = self.non_empty_scenes[self.scene_idx]
        print(f'Current scene token: {self.scene_token}')
        self.sample_ts = sorted(self.tracks_gt[self.scene_token].keys())
        self.track_ts = sorted(self.tracks_pred[self.scene_token].keys())
        self.wis_3d = Wis3D(self.scene_token)
                
        # tracking id coloring
        self.id2color = {}  # The color of each track.
        
        super().__init__()
    
    
    @property
    def non_empty_scenes(self):
        scenes = []
        for scene_token in self.scene_tokens:
            for ts in self.tracks_pred[scene_token].keys():
                if self.tracks_pred[scene_token][ts] != []:
                    scenes.append(scene_token)
                    break
        return scenes
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, value):
        if 0 <= value < len(self.sample_ts):
            self._index = value
        else:
            self._index = value % len(self.sample_ts)
        
    @property
    def pc(self):
        ts = self.sample_ts[self.index]
        bboxes = self.tracks_gt[self.scene_token][ts]
        sample_token = bboxes[0].sample_token
        sample = self.nusc.get('sample', sample_token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_data_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(lidar_data_path)

        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        rotation = calibrated_sensor['rotation']
        translation = calibrated_sensor['translation']
        pc.rotate(Quaternion(rotation).rotation_matrix)
        pc.translate(np.array(translation))
        
        return pc.points.T[:, :3]
       
    @property
    def lidar_seg(self):
        ts = self.sample_ts[self.index]
        bboxes = self.tracks_gt[self.scene_token][ts]
        sample_token = bboxes[0].sample_token
        sample = self.nusc.get('sample', sample_token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_seg_path = os.path.join(self.nusc.dataroot, 'lidarseg', self.version,
                                      lidar_data['token'] + "_lidarseg.bin")
        if self.version != "v1.0-test":
            points_labels = np.fromfile(lidar_seg_path, dtype=np.uint8)
    
        return points_labels
    
    def add_bbox_lidar(self):
        # Load gt tracking bbox
        ts = self.sample_ts[self.index]
        bboxes = self.tracks_gt[self.scene_token][ts]
        sample_token = bboxes[0].sample_token
        sample = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

        for idx, bbox in enumerate(bboxes):
            
            nuscenes_format_box = Box(bbox.translation, bbox.size, Quaternion(bbox.rotation))
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            nuscenes_format_box.translate(-np.array(pose_record['translation']))
            nuscenes_format_box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            
            our_format_bbox = np.array([nuscenes_format_box.center[0], nuscenes_format_box.center[1], 
                                        nuscenes_format_box.center[2], nuscenes_format_box.wlh[1], nuscenes_format_box.wlh[0], 
                                        nuscenes_format_box.wlh[2], quaternion_yaw(Quaternion(nuscenes_format_box.orientation)), 1])
            
            if bbox.tracking_id not in self.id2color.keys():
                self.id2color[bbox.tracking_id] = (float(hash(bbox.tracking_id + 'r') % 256) / 255,
                                                    float(hash(bbox.tracking_id + 'g') % 256) / 255,
                                                    float(hash(bbox.tracking_id + 'b') % 256) / 255,
                                                    1)
            color = self.id2color[bbox.tracking_id]
            self.add_bbox(our_format_bbox, name=f'gt_{bbox.tracking_id[-3:]}', color=(1,1,1,1), width=2)

            self.wis_3d.add_3d_bbox(our_format_bbox, name=f'gt_{bbox.tracking_id[-3:]}')
        
        try:
            # load prediction bbox
            pred_bboxes = self.tracks_pred[self.scene_token][ts]
            trk_ids = []
            for idx, bbox in enumerate(pred_bboxes):
                # Determine color for this tracking id.
                if bbox.tracking_id not in self.id2color.keys():
                    self.id2color[bbox.tracking_id] = (float(hash(bbox.tracking_id + 'r') % 256) / 255,
                                                    float(hash(bbox.tracking_id + 'g') % 256) / 255,
                                                    float(hash(bbox.tracking_id + 'b') % 256) / 255,
                                                    1)
                color = self.id2color[bbox.tracking_id]
                trk_ids.append(bbox.tracking_id)
                nuscenes_format_box = Box(bbox.translation, bbox.size, Quaternion(bbox.rotation))
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                nuscenes_format_box.translate(-np.array(pose_record['translation']))
                nuscenes_format_box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
                
                our_format_bbox = np.array([nuscenes_format_box.center[0], nuscenes_format_box.center[1], 
                                            nuscenes_format_box.center[2], nuscenes_format_box.wlh[1], nuscenes_format_box.wlh[0], 
                                            nuscenes_format_box.wlh[2], quaternion_yaw(Quaternion(nuscenes_format_box.orientation)), 1])
                
                self.add_bbox(our_format_bbox, name=f'pred_{bbox.tracking_id}', color=color, label_pos='front')
            
            logger.info(f'pred trk_ids: {sorted(trk_ids)}')

        except Exception as e:
            logger.info(f'No prediction for {ts}')
            
    def render_frame(self):
        lidar_seg = self.lidar_seg
        lidar_seg_cls = [self.nusc.lidarseg_idx2name_mapping[x] for x in lidar_seg.tolist()]
        lidar_seg_color = np.array([self.colormap[c] for c in lidar_seg_cls]) / 255
        lidar_seg_color = np.hstack((lidar_seg_color, np.ones((lidar_seg_color.shape[0], 1))))
        if len(self.pc) != len(lidar_seg_color):
            lidar_seg_color=None

        self.wis_3d.set_scene_id(self.index)
        self.add_pc(self.pc[:, :3], color=lidar_seg_color)
        self.clear_bbox()
        self.add_bbox_lidar()
        self.show_label()
        
        
        try:
            assert len(self._bbox) == len(self._bbox_color)
        
        except Exception as e:
            logger.warning(f'bbox: {len(self._bbox)}, bbox_color: {len(self._bbox_color)}')
        
        self.wis_3d.add_point_cloud(self.pc[:, :3], colors=lidar_seg_color, name=str(self.index).zfill(5))
        
        
    def on_key_press(self, event):
            
        self.view_center = list(self.view.camera.center)
                
        if(event.key == 'left'):
            self.index -= 1
            logger.info(f'index: {self.index}, total: {len(self.sample_ts)}')
            self.render_frame()
            
        if(event.key == 'right'):
            self.index += 1
            logger.info(f'index: {self.index}, total: {len(self.sample_ts)}')
            self.render_frame()
            
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
        
        if (event.text == '='):
            self.pc_points_size += 1
            logger.info(f'point size: {self.pc_points_size}')
            self.render_frame()
            
        if (event.text == '-'):
            self.pc_points_size -= 1
            logger.info(f'point size: {self.pc_points_size}')
            self.render_frame()
        
        if (event.text == 'n'):
            self.scene_idx = (self.scene_idx + 1) % len(self.non_empty_scenes)
            print(f'Current scene idx: {self.scene_idx}')
            self.scene_token = self.non_empty_scenes[self.scene_idx]
            print(f'Current scene token: {self.scene_token}')
            self.sample_ts = sorted(self.tracks_gt[self.scene_token].keys())
            self.wis_3d = Wis3D(self.scene_token)
            self.render_frame()
            
        if (event.text == 'p'):
            self.scene_idx = (self.scene_idx - 1) % len(self.non_empty_scenes)
            print(f'Current scene idx: {self.scene_idx}')
            self.scene_token = self.non_empty_scenes[self.scene_idx]
            print(f'Current scene token: {self.scene_token}')
            self.sample_ts = sorted(self.tracks_gt[self.scene_token].keys())
            self.wis_3d = Wis3D(self.scene_token)
            self.render_frame()
        

    def run(self):
        self.render_frame()

        vispy.app.run()

if __name__ == '__main__':
    nuscene_path = '/media/nio/Storage/nuscenes'
    results_path = ''
    tool = annotator(nuscene_path, result_path)
    tool.run()
