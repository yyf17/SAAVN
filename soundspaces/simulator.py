
from typing import Any, List
from collections import defaultdict, namedtuple
import logging
import time
import pickle
import os

import librosa
import scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve
import numpy as np
import networkx as nx

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import Config, AgentState, ShortestPathPoint
from soundspaces.utils import load_metadata, _to_tensor

#from habitat.core.registry import registry
# from ss_baselines.common.registry import registry

class DummySimulator:
    """
    Dummy simulator for avoiding loading the scene meshes when using cached observations.
    """
    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


# @registry.register_simulator(name="SoundSpaces")
class SoundSpaces(HabitatSim):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator first loads the graph of current environment and moves the agent among nodes.
    Any sounds can be specified in the episode and loaded in this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        print("2:sim enter--->SoundSpaces")
        super().__init__(config)
        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sound = None
        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._node2index = None
        self._frame_cache = dict()
        self._audiogoal_cache = dict()
        self._spectrogram_cache = dict()
        self._scene_observations = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False

        self.points, self.graph = load_metadata(self.metadata_dir)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node
        self._load_sound_sources()
        logging.info('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.close()
            del self._sim
            self._sim = DummySimulator()
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent_state = super().get_agent_state(agent_id)
        else:
            agent_state = self._sim.get_agent_state()

        return agent_state

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        if not self.config.USE_RENDERED_OBSERVATIONS:
            super().set_agent_state(position, rotation, agent_id=agent_id, reset_sensors=reset_sensors)
        else:
            pass

    @property
    def binaural_rir_dir(self):
        return os.path.join(self.config.AUDIO.BINAURAL_RIR_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def metadata_dir(self):
        return os.path.join(self.config.AUDIO.METADATA_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]

    def reconfigure(self, config: Config) -> None:
        self.config = config
        is_same_sound = config.AGENT_0.SOUND == self._current_sound
        if not is_same_sound:
            self._current_sound = self.config.AGENT_0.SOUND

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self._update_agents_state()
                self._frame_cache = dict()
            else:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        if not is_same_scene or not is_same_sound:
            self._audiogoal_cache = dict()
            self._spectrogram_cache = dict()

        self._episode_step_count = 0

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.GOAL_POSITION)
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))

        logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index, self.get_orientation()))

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            assert not self.config.USE_RENDERED_OBSERVATIONS
            sim_obs = self._sim.get_sensor_observations()
            for sensor in sim_obs:
                sim_obs[sensor] = sim_obs[sensor]
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        logging.debug('Reset simulation')
        if not self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._previous_step_collided = False
        # STOP: 0, FORWARD: 1, LEFT: 2, RIGHT: 2
        if action == HabitatSimActions.STOP:
            self._is_episode_active = False
        else:
            prev_position_index = self._receiver_position_index
            prev_rotation_angle = self._rotation_angle
            if action == HabitatSimActions.MOVE_FORWARD:
                # the agent initially faces -Z by default
                self._previous_step_collided = True
                for neighbor in self.graph[self._receiver_position_index]:
                    p1 = self.graph.nodes[self._receiver_position_index]['point']
                    p2 = self.graph.nodes[neighbor]['point']
                    direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                    if direction == self.get_orientation():
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
            elif action == HabitatSimActions.TURN_LEFT:
                # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
                self._rotation_angle = (self._rotation_angle + 90) % 360
            elif action == HabitatSimActions.TURN_RIGHT:
                self._rotation_angle = (self._rotation_angle - 90) % 360

            if self.config.CONTINUOUS_VIEW_CHANGE:
                intermediate_observations = list()
                fps = self.config.VIEW_CHANGE_FPS
                if action == HabitatSimActions.MOVE_FORWARD:
                    prev_position = np.array(self.graph.nodes[prev_position_index]['point'])
                    current_position = np.array(self.graph.nodes[self._receiver_position_index]['point'])
                    for i in range(1, fps):
                        intermediate_position = prev_position + i / fps * (current_position - prev_position)
                        self.set_agent_state(intermediate_position.tolist(), quat_from_angle_axis(np.deg2rad(
                                            self._rotation_angle), np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)
                else:
                    for i in range(1, fps):
                        if action == HabitatSimActions.TURN_LEFT:
                            intermediate_rotation = prev_rotation_angle + i / fps * 90
                        elif action == HabitatSimActions.TURN_RIGHT:
                            intermediate_rotation = prev_rotation_angle - i / fps * 90
                        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                             quat_from_angle_axis(np.deg2rad(intermediate_rotation),
                                                                  np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                     quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
            else:
                self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                          quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        self._episode_step_count += 1

        # log debugging info
        logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_index, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        if self.config.CONTINUOUS_VIEW_CHANGE:
            observations['intermediate'] = intermediate_observations

        return observations

    def get_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    @property
    def azimuth_angle(self):
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    @property
    def reaching_goal(self):
        return self._source_position_index == self._receiver_position_index

    def _update_observations_with_audio(self, observations):
        audio = self.get_current_audio_observation()
        observations.update({"audio": audio})

    def _load_sound_sources(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sound = sound_file.split('.')[0]
            sr, audio_data = wavfile.read(os.path.join(self.source_sound_dir, sound_file))
            assert sr == 44100
            if sr != self.config.AUDIO.RIR_SAMPLING_RATE:
                audio_data = scipy.signal.resample(audio_data, self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound] = audio_data
    
    def _compute_euclidean_distance_between_sr_locations(self):
        p1 = self.graph.nodes[self._receiver_position_index]['point']
        p2 = self.graph.nodes[self._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        return d

    def _compute_audiogoal(self):
        binaural_rir_file = os.path.join(self.binaural_rir_dir, str(self.azimuth_angle), '{}_{}.wav'.format(
            self._receiver_position_index, self._source_position_index))
        try:
            sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32

        except ValueError:
            logging.warning("{} file is not readable".format(binaural_rir_file))
            binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)
        if len(binaural_rir) == 0:
            logging.debug("Empty RIR file at {}".format(binaural_rir_file))
            binaural_rir = np.zeros((self.config.AUDIO.RIR_SAMPLING_RATE, 2)).astype(np.float32)

        # by default, convolve in full mode, which preserves the direct sound
        binaural_convolved = [fftconvolve(self.current_source_sound, binaural_rir[:, channel]
                                          ) for channel in range(binaural_rir.shape[-1])]
        audiogoal = np.array(binaural_convolved)[:, :self.current_source_sound.shape[0]]

        free_range = audiogoal.shape[1] - self.config.AUDIO.RIR_SAMPLING_RATE
        if free_range > 0:
            slide_k = np.random.randint(0,high=free_range)
            audiogoal = audiogoal[:,slide_k:slide_k+self.config.AUDIO.RIR_SAMPLING_RATE]

        return audiogoal

    def get_current_audiogoal_observation(self):
        join_index = (self._source_position_index, self._receiver_position_index, self.azimuth_angle)
        if join_index not in self._audiogoal_cache:
            self._audiogoal_cache[join_index] = self._compute_audiogoal()

        return self._audiogoal_cache[join_index]

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        sr_index = (self._source_position_index, self._receiver_position_index)
        join_index = sr_index + (self.azimuth_angle,)
        if join_index not in self._spectrogram_cache:
            audiogoal = self._compute_audiogoal()
            spectrogram = audiogoal2spectrogram(audiogoal)
            self._spectrogram_cache[join_index] = spectrogram

        return self._spectrogram_cache[join_index]

    def geodesic_distance(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None
        steps = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE

        return steps

    def get_straight_shortest_path_points(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None

        shortest_path = nx.shortest_path(self.graph, source=index_a, target=index_b)
        points = list()
        for node in shortest_path:
            points.append(self.graph.nodes()[node]['point'])
        return points

    @property
    def previous_step_collided(self):
        return self._previous_step_collided
