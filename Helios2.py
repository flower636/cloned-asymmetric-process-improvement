import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import time
from arena_api.__future__.save import Writer
from arena_api.system import system
from arena_api.buffer import BufferFactory, _Buffer
from arena_api.enums import PixelFormat

from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import ctypes

from scipy.signal import find_peaks

import cv2
import json

from cv2.typing import MatLike
from numpy.typing import NDArray

import logging
import inspect
import traceback

import ctypes
from typing import Union, Literal, Any

def load_config(config_path: str) -> dict:
    """Loads a JSON configuration file.
    Args:
        config_path (str): Path to the JSON configuration file.
    Returns:
        dict: The loaded configuration as a dictionary.
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    config = {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from {config_path}: {e.msg}", e.doc, e.pos) from e
    return config

def yn_prompt(prompt: str) -> bool:
            response = ""
            while response not in ["y", "n"]:
                response = input(prompt).lower()
                if response not in ["y", "n"]:
                    print("Invalid input. Please enter Y or N.")
            return response == "y"
        
def str_prompt(prompt: str)-> str:
    response = ""
    while response.strip() == "":
        response = input(prompt)
        if response.strip() == "":
            print("String cannot be empty.")
    return response

def float_prompt(prompt: str, default: Any = None) -> float:
            while True:
                response = input(prompt)
                if response == "" and default is not None:
                    return default
                try:
                    value = float(response)
                    return value
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
        

class Logger:
    def __init__(self, logfile: str, name, verbose:bool=False):
        from logging.handlers import RotatingFileHandler
        log_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(filename)s:%(funcName)s:%(lineno)d]'
        )
        rotating_handler = RotatingFileHandler(logfile, maxBytes=5*1024*1024, backupCount=3)
        rotating_handler.setFormatter(log_formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(rotating_handler)
            if verbose:
                self.logger.addHandler(stream_handler)

    def log(self, message:str, e:None|Exception=None, caller=None):
        """Logs with caller location and optional exception details and traceback for errors."""
        identifier = id(caller) if caller else None
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        if len(outer_frames) > 1:
            caller_frame = outer_frames[1]
            location = f"[{caller_frame.filename}:{caller_frame.function}:{caller_frame.lineno}]"
        else:
            location = "[unknown location]"
        if e:
            self.logger.error(f"{identifier} {message} {location}")
            self.logger.error(f"{identifier} Exception details: {e} {location}")
            tb = traceback.format_exc()
            if tb and tb != 'NoneType: None\n':
                self.logger.error(f"{identifier} Traceback:\n{tb}")
            else:
                self.logger.error(f"{identifier} No traceback available.")
        else:
            self.logger.info(f"{identifier} {message} {location}")

class PointsProcessingError(Exception):
    """Exception raised for errors in point cloud or region processing."""
    pass

class Helios2Error(Exception):
    """Exception raised for errors specific to Helios2 camera operations."""
    pass

from dataclasses import dataclass
@dataclass
class Gripper:
    id: str
    diameter: float 
    length: float
    attached_template: str | None 
    gripperrack_template: str | None

@dataclass
class Sleeve:
    id: str
    od: float
    bcd: float
    length:float

class GripperTemplateSet:
    def __init__(self, templates_dir:str):
        self.logger = Logger("Helios2.log", __name__)
        self.templates_dir = templates_dir
        self.templates_json_path = os.path.join(templates_dir, "grippers.json")
        if not os.path.exists(self.templates_json_path):
            raise FileNotFoundError(f"Templates file not found: {self.templates_json_path}")
        else:
            with open(self.templates_json_path, 'r') as f:
                self.templates = json.load(f)
        self.logger.log(f"GripperTemplateSet object {id(self)} created with templates from {self.templates_json_path}", caller=self)

    def __del__(self):
        self.logger.log(f"GripperTemplateSet instance {id(self)} deleted.", caller=self)

    def list_grippers(self)->None:
        """Prints all the grippers in `self.templates`.
        Returns:
            None
        """
        for gripper_id in self.templates:
            gripper = self.templates[gripper_id]
            if 'attached_template' in gripper:
                attached_template = gripper['attached_template']
            else:
                attached_template = None
            if 'gripperrack_template' in gripper:
                gripperrack_template = gripper['gripperrack_template']
            else:
                gripperrack_template = None
            print(f"Gripper ID: {gripper_id}, Attached Template: {attached_template}, Gripper Rack Template: {gripperrack_template}")

    def get_id_list(self, attached:bool=False)-> list[str]:
        """Returns a list of all gripper IDs in `self.templates`.
        Args:
            attached (bool): If True, returns only IDs of attached grippers. If False only returns IDs of grippers with gripper rack templates. Default is False.
        Returns:
            list: A list of all gripper IDs.
        """
        if attached:
            return [gripper_id for gripper_id in self.templates.keys() if self.templates[gripper_id].get('attached_template', False)]
        else:
            return [gripper_id for gripper_id in self.templates.keys() if self.templates[gripper_id].get('gripperrack_template', False)]

    def get_template(self, gripper_id:str|int, attached:bool=False)-> MatLike | None:
        """Returns the template image for the given ID and template type (attached or gripperrack).
        Args:
            gripper_id (str): The identifier for the gripper template.
            attached (bool): If True, returns the attached gripper template. If False, returns the gripperrack template.
        Returns:
            np.ndarray: The template image as a grayscale numpy array.
        """
        try:
            if gripper_id not in self.templates:
                raise ValueError(f"Template with ID '{gripper_id}' not found.")
            template_key = "attached_template" if attached else "gripperrack_template"
            if template_key not in self.templates[gripper_id]:
                raise ValueError(f"Template type '{template_key}' not found for ID '{gripper_id}'.")
            template_filename = self.templates[gripper_id][template_key]
            template_path = os.path.join(self.templates_dir, template_filename)
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template image not found: {template_path}")
            return cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            self.logger.log(f"Error getting template with ID '{gripper_id}' (attached={attached}):", e, caller=self)
            raise e

import threading

class Helios2:
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs): 
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super(Helios2, cls).__new__(cls)
        return cls._instance
    """Class to initialize and utilize a Helios2 camera.
    Attributes:
        position (np.ndarray): The position of the camera [x, y, z]. (unused)
        orientation (np.ndarray): The orientation of the camera [roll, pitch, yaw]. (unused)
        width (int): The width of the camera image in pixels. (unused)
        height (int): The height of the camera image in pixels. (unused)
        fov (int): The field of view of the camera in degrees. (unused)
        focal_length (float or None): The focal length of the camera. (unused)
        buffer: The last captured buffer from the camera (unused).
        min_range (int): The minimum valid range for depth values.
        max_range (int): The maximum valid range for depth values.
        device: The connected Helios2 device object.
        nodemap: The nodemap of the connected device for camera settings.
    """
    def __init__(self, verbose:bool=False)->None:
        if Helios2._initialized:
            return
        self.logger = Logger("Helios2.log", __name__, verbose=verbose)

        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno
        self.logger.log(f"Helios2 object {id(self)} created in {caller_file} at line {caller_line}", caller=self)

        self.position = np.array([0, 0, 0])
        self.orientation = np.array([0, 0, 0])
        self.width = 640
        self.height = 480
        self.fov = 69
        self.focal_length = None
        self.buffer = None

        self.min_range:float = 0
        self.max_range:float = 8300

        tries = 0
        tries_max = 3
        sleep_time_secs = 10
        devices = None
        while tries < tries_max:  # Wait for device for 60 seconds
            devices = system.create_device()
            if not devices:
                print(
                    f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                        f'secs for a device to be connected!')
                for sec_count in range(sleep_time_secs):
                    time.sleep(1)
                    print(f'{sec_count + 1 } seconds passed ',
                          '.' * sec_count, end='\r')
                tries += 1
            else:
                break
        else:
            raise Exception(f'No Helios2 camera found!')
        for device in devices:
            print(device)
        
        #device = system.select_device(devices)
        self.device = devices[0]
        print(f'Device used:\n\t{self.device}')

        # #### Check if Helios2 camera is being used for the example
        isHelios2 = False
        # validate if Scan3dCoordinateSelector node exists.
        # If not, it is (probably) not a Helios Camera running the example
        try:
            self.device.nodemap['Scan3dOperatingMode'].value
        except KeyError as e:
            self.logger.log('Scan3dOperatingMode node is not found. Please make sure that Helios device is used for the example.\n', e, caller=self)
            raise Exception('Scan3dOperatingMode node is not found. Please make sure that Helios device is used for the example.\n')

        # validate if Scan3dCoordinateOffset node exists.
        # If not, it is (probably) that Helios Camera has an old firmware
        try:
            self.device.nodemap['Scan3dCoordinateOffset'].value
        except KeyError as e:
            self.logger.log('Scan3dCoordinateOffset node is not found. Please update Helios firmware.\n', e, caller=self)
            raise Exception('Scan3dCoordinateOffset node is not found. Please update Helios firmware.\n') from e

        # check if Helios2 camera used for the example
        device_model_name_node = self.device.nodemap['DeviceModelName'].value
        print(device_model_name_node)
        if 'HLT' in device_model_name_node or 'HTP003S-001' in device_model_name_node:
            isHelios2 = True

        print('\nSettings nodes')
        self.nodemap = self.device.nodemap
        self.tl_stream_nodemap = self.device.tl_stream_nodemap
        # set pixel format
        print('\tSetting pixelformat to Coord3D_ABCY16')
        self.nodemap.get_node('PixelFormat').value = 'Coord3D_ABCY16'

        # set operating mode distance
        print('\tSetting 3D operating mode')
        if isHelios2 is True:
            self.nodemap['Scan3dOperatingMode'].value = 'Distance5000mmMultiFreq' # 'Distance8300mmMultiFreq'
            # self.nodemap['Scan3dOperatingMode'].value = 'Distance8300mmMultiFreq'
        else:
            self.nodemap['Scan3dOperatingMode'].value = 'Distance1500mm'

        # set exposure time
        self.nodemap['Scan3dHDRMode'].value = 'Off'
        self.nodemap['ExposureTimeSelector'].value = "Exp1000Us" #'Exp62_5Us' # Exp250Us #'Exp1000Us'
        self.nodemap['ConversionGain'].value = 'High'
        #self.nodemap['Scan3dImageAccumulation'].value = 32
        self.nodemap['Scan3dImageAccumulation'].value = 1
        self.nodemap['Scan3dSpatialFilterEnable'].value = False
        self.nodemap['Scan3dConfidenceThresholdEnable'].value = True
        if self.nodemap['Scan3dConfidenceThresholdEnable'].value:
            self.nodemap['Scan3dConfidenceThresholdMin'].value = 3000
        # self.nodemap['Scan3dHDRMode'].value = 'StandardHDR' # 'LowNoiseHDRX32' # "StandardHDR"
        self.nodemap['Scan3dFlyingPixelsRemovalEnable'].value = True
        self.nodemap['Scan3dFlyingPixelsDistanceThreshold'].value = 20
        self.logger.log("Helios2 camera initialized successfully.", caller=self)

        Helios2._initialized = True

    def __del__(self):
        self.logger.log(f"Helios2 instance {id(self)} deleted.", caller=self)

    def default_nodevalues(self, config_file:str="./config/config.json"):
        """Sets nodevalues to predetermined defaults.
        """
        try:
            config = load_config(config_file)
            default_nodemap_values = config["default_nodemap_values"]
            default_tl_stream_nodemap_values = config["default_tl_stream_nodemap_values"]
        except FileNotFoundError as e:
            self.logger.log(f"Config file not found:", e, caller=self)
            raise e
        except Exception as e:
            self.logger.log(f"Error loading default node values from config file:", e, caller=self)
            raise e
        for node, value in default_nodemap_values.items():
            try:
                self.nodemap[node].value = value
            except KeyError as e:
                error_msg = f"Node '{node}' not found in nodemap. Please check config file."
                self.logger.log(error_msg, e, caller=self)
                raise KeyError(error_msg) from e
            except Exception as e:
                error_msg = f"An error occurred while setting node '{node}' to value '{value}'."
                self.logger.log(error_msg, e, caller=self)
                raise Helios2Error(error_msg) from e 
        for node, value in default_tl_stream_nodemap_values.items():
            try:
                self.tl_stream_nodemap[node].value = value
            except KeyError as e:
                error_msg = f"Node '{node}' not found in tl_stream_nodemap. Please check config file."
                self.logger.log(error_msg, e, caller=self)
                raise KeyError(error_msg) from e
            except Exception as e:
                error_msg = f"An error occurred while setting node '{node}' to value '{value}'."
                self.logger.log(error_msg, e, caller=self)
                raise Helios2Error(error_msg) from e

    def nodevalues_preset(self, preset:str):
        options = ["default", "gripper_detection", "sleeve_presence", "sleeve_diameter"] # TODO: add nodevalues for sleeve presence and sleeve diameter
        if preset not in options:
            e = ValueError(f"Invalid preset. Options are: {options}.")
            self.logger.log(f"Invalid preset. Options are: {options}.", e, caller=self)
            raise e
        match preset:
            case "default":
                self.default_nodevalues()
            case "gripper_detection":
                self.nodemap['Scan3dConfidenceThresholdEnable'].value = False
                self.nodemap['Scan3dFlyingPixelsRemovalEnable'].value = False
#                self.nodemap["Scan3dFlyingPixelsDistanceThreshold"].value = 15 # mm
                self.nodemap['Scan3dHDRMode'].value = "Off"
                pass
            case "sleeve_presence": # TODO:
                raise NotImplementedError("A nodevalue preset for sleeve presence checking has not been implemented yet.")
            case "sleeve_diameter": # TODO:
                raise NotImplementedError("A nodevalue preset for sleeve diameter checking has not been implemented yet.")
            case _:
                print(f"Preset {preset} is not a valid choice. Choose from {options}.")
                self.logger.log(f"Preset {preset} is not a valid choice. Choose from {options}.", caller=self)

    def capture_buffer(self)->_Buffer:
        """Captures a buffer and return a copy.
        Returns:
            buffer (arena_api.buffer._Buffer): Copy of the captured buffer.
        Raises:
            Helios2Error: If an error occurs during buffer capture.
        """
        with self.device.start_stream():
            try:
                self.logger.log("Starting stream with 1 buffer...", caller=self)
                print(f'\nStream started with 1 buffer')

                # This would timeout or return 1 buffers
                buffer = self.device.get_buffer()
                print(buffer)
                print('\tBuffer received')

                # Requeue the chunk data buffers
                self.buffer = buffer
                buffer_copy = BufferFactory.copy(buffer)
                self.device.requeue_buffer(buffer)
                print(f'\tImage buffer requeued')
                self.logger.log("Buffer captured successfully.", caller=self)
                return buffer_copy
            except Exception as e:
                self.logger.log(f'Error during buffer capture: {e}', e, caller=self)
                raise Helios2Error(f'Error during buffer capture: {e}') from e

    def capture(self, capture_type:Literal['ply', 'depthmap', 'jpg', 'raw', 'bmp'], buffer:_Buffer|None=None)-> np.ndarray | None:
        """
        Captures or saves a buffer depending on the specified `type`.
        Args:
            capture_type (str, optional): The type of capture to perform. Options are:
                - None: Prompts the user to choose a capture type.
                - "buffer": Captures and returns a buffer from the camera.
                - "ply": Captures a buffer and saves it as a PLY point cloud file ('./buffer.ply').
                - "depthmap": Captures a buffer, generates a grayscale depth map, saves it as 'depthmap.jpg', and returns the grayscale array.
        Returns:
            buffer (arena_api.buffer._Buffer or array or None):
                - If capture_type is "buffer": Returns a copy of the captured buffer.
                - If capture_type is "ply": Returns the buffer after saving as a PLY file.
                - If capture_type is "depthmap": Returns a grayscale array representing the depth map.
                - If capture_type is None or invalid: Returns None.
        """
        cases = ['ply', 'depthmap', 'jpg', 'raw', 'bmp']
        if capture_type not in cases:
            e = ValueError(f"Invalid capture type. Options are: {cases}.")
            self.logger.log(f"Invalid capture type. Options are: {cases}.", e, caller=self)
            raise e
        # if capture_type == None:
        #     e = ValueError(f"Type must be specified. Options are: {cases}.")
        #     self.logger.log(f"Type must be specified. Options are: {cases}.", e, caller=self)
        #     raise e
        if buffer is None:
            try:
                buffer = self.capture_buffer()
            except Helios2Error as e:
                self.logger.log(f"Error capturing buffer: {e}", e, caller=self)
                raise e
        # create an image writer
        # writer = Writer.from_buffer(buffer)
        writer = Writer(
            width=buffer.width,
            height=buffer.height,
            bits_per_pixel=buffer.bits_per_pixel
        )
        self.logger.log(f"Created Writer with width:{buffer.width}, height:{buffer.height} and {buffer.bits_per_pixel} bits per pixel.", caller=self)
        match capture_type:
            case "ply":
                try:
                    # NOTE: ply kwargs:
                    # 'filter_points'
                    # 'is_signed'
                    # 'scale'
                    # 'offset_a'
                    # 'offset_b'
                    # 'offset_c'
                    kwargs = {
                        "offset_a": 0.0,
                        "offset_b": 0.0,
                        "offset_c": 0.0,
                    }
                    writer.save(buffer, './buffer.ply', **kwargs)
                    print(f'\tImage saved {writer.saved_images[-1]}')
                    self.logger.log("Point cloud captured and saved successfully as ./buffer.ply.", caller=self)
                except Exception as e:
                    self.logger.log(f'Error during PLY capture: {e}', e, caller=self)
                    raise Helios2Error(f'Error during PLY capture: {e}') from e
            case "jpg":
                try:
                    # FIXME: fix jpg capture
                    # NOTE: jpg kwargs:
                    # 'quality' 
                    # 'progressive'
                    # 'subsampling'
                    # 'optimize'
                    # NOTE: subsampling options
                    # SC_JPEG_SUBSAMPLING_411 = 1
                    # SC_JPEG_SUBSAMPLING_420 = 2
                    # SC_JPEG_SUBSAMPLING_422 = 3
                    # SC_NO_JPEG_SUBSAMPLING = 0  # default
                    kwargs = {
                        "quality": 95,
                        "progressive": False,
                        "subsampling": 0,
                        "optimize": False,
                    }
                    number_of_pixels = buffer.width * buffer.height
                    channels_per_pixel = 1
                    array_grayscale_size_in_bytes = channels_per_pixel * number_of_pixels
                    buffer_3d_step_size = 4
                    CustomArrayType = (ctypes.c_byte * array_grayscale_size_in_bytes)
                    array_grayscale_for_jpg = CustomArrayType()
                    buffer_intensity_pixel_index = 0
                    pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_int16))

                    for _ in range(number_of_pixels):
                        intensity = pdata_16bit[buffer_intensity_pixel_index + 3 ]
                        intensity = int(intensity * 255 / 65535)  # Scale to 0-255
                        array_grayscale_for_jpg[buffer_intensity_pixel_index] = intensity
                        buffer_intensity_pixel_index += 1
                        print(buffer_intensity_pixel_index)
                    uint8_ptr = ctypes.POINTER(ctypes.c_ubyte)
                    ptr_array_grayscale_for_jpg = uint8_ptr(array_grayscale_for_jpg)
                    bits_per_pixel = 8
                    array_grayscale_for_jpg_size_in_bytes = int(
                        buffer.width * buffer.height * bits_per_pixel / 8
                    )
                    intensity_buffer = BufferFactory.create(
                        ptr_array_grayscale_for_jpg,
                        array_grayscale_size_in_bytes,
                        buffer.width,
                        buffer.height,
                        PixelFormat.Coord3D_ABCY16)

                    writer_jpg = Writer.from_buffer(intensity_buffer)
                    writer_jpg.save(intensity_buffer, 'intentisy.jpg')
                    print(f'\tImage saved {writer.saved_images[-1]}')
                    self.logger.log("JPG captured and saved successfully as ./buffer.jpg.", caller=self)
                except Exception as e:
                    self.logger.log(f'Error during JPG capture:', e, caller=self)
                    raise Helios2Error(f'Error during JPG capture: {e}') from e
            case "bmp":
                # FIXME: fix bmp capture
                try:
                    writer.save(buffer, './buffer.bmp')
                    print(f'\tImage saved {writer.saved_images[-1]}')
                    self.logger.log("Point cloud captured and saved successfully as ./buffer.bmp.", caller=self)
                except Exception as e:
                    self.logger.log(f'Error during BMP capture: {e}', e, caller=self)
                    raise Helios2Error(f'Error during BMP capture: {e}') from e
            case "raw":
                # FIXME: fix raw capture
                try:
                    print(f'\tImage saved {writer.saved_images[-1]}')
                    self.logger.log("Point cloud captured and saved successfully as ./buffer.raw.", caller=self)
                except Exception as e:
                    self.logger.log(f'Error during RAW capture: {e}', e, caller=self)
                    raise Helios2Error(f'Error during RAW capture: {e}') from e
            case "depthmap":
                with self.device.start_stream():
                    try:
                        scale_z = self.nodemap['Scan3dCoordinateScale'].value
                        buffer_3d = self.device.get_buffer()
                        print('\tbuffer received')
                        number_of_pixels = buffer_3d.width * buffer_3d.height
                        Grayscale_channels_per_pixel = 1  # Single channel for grayscale
                        Grayscale_pixel_size_bytes = Grayscale_channels_per_pixel
                        array_grayscale_size_in_bytes = Grayscale_pixel_size_bytes * number_of_pixels
                        buffer_3d_step_size = 4
                        CustomArrayType = (ctypes.c_byte * array_grayscale_size_in_bytes)
                        array_grayscale_for_jpg = CustomArrayType()
                        buffer_3d_pixel_index = 0

                        array_grayscale_pixel_index = 0
                        pdata_16bit = ctypes.cast(buffer_3d.pdata, ctypes.POINTER(ctypes.c_int16))

                        for _ in range(number_of_pixels):
                            z = pdata_16bit[buffer_3d_pixel_index + 2]
                            z = int(z * scale_z)
                            if self.min_range<z<=self.max_range:
                                grayscale =  int(
                                    (self.max_range-z)*(255/(self.max_range - self.min_range))
                                )
                            else:
                                grayscale = 0
                            array_grayscale_for_jpg[array_grayscale_pixel_index] = grayscale
                            buffer_3d_pixel_index += buffer_3d_step_size
                            array_grayscale_pixel_index += Grayscale_channels_per_pixel

                        uint8_ptr = ctypes.POINTER(ctypes.c_ubyte)
                        ptr_array_grayscale_for_jpg = uint8_ptr(array_grayscale_for_jpg)
                        bits_per_pixel = 8  # Grayscale is 8 bits per pixel
                        array_grayscale_for_jpg_size_in_bytes = int(
                            buffer_3d.width * buffer_3d.height * bits_per_pixel / 8
                        )
                        heat_buffer = BufferFactory.create(
                            ptr_array_grayscale_for_jpg,
                            array_grayscale_for_jpg_size_in_bytes,
                            buffer_3d.width,
                            buffer_3d.height,
                            PixelFormat.Mono8)
                        write = True
                        if write == True:
                            writer_jpg = Writer.from_buffer(heat_buffer)
                            writer_jpg.save(heat_buffer, 'depthmap.jpg')
                        BufferFactory.destroy(heat_buffer)
                        self.logger.log("Depth map captured and saved successfully as depthmap.jpg.",
                                        caller=self)
                        return array_grayscale_for_jpg
                    except Exception as e:
                        self.logger.log(f'Error during depth map capture: {e}', e, caller=self)
                        raise Helios2Error(f'Error during depth map capture: {e}') from e
            case _:
                e = ValueError(f"Invalid type specified. Options are: {cases}.")
                self.logger.log(f"Invalid type specified. Options are: {cases}.", e, caller=self)
                raise e

    def slice(
        self,
        z:float,
        margin:float,
        buffer:None|_Buffer=None,
        points=None,
        plot:bool=False
    )-> np.ndarray:
        """Keeps only points from a buffer within z+/-margin mm.
        Args:
            z (float): The Z value to slice the points around in mm.
            margin (float): Width of the slice in mm.
            buffer (arena_api.buffer._Buffer or None): Buffer to slice points from. If None, captures a new buffer.
            points (np.ndarray or None): Array with shape (n, 3) containing [x,y,z] points. If None, will be extracted from the buffer.
            plot (bool): Enable/disable plotting of the sliced points in 3D.
        """
        if buffer is None and points is None:
            try:
                buffer = self.capture_buffer()
                points = self.buffer_to_numpy(buffer)
            except Helios2Error as e:
                self.logger.log("Error capturing buffer", e, caller=self)
                raise e
            except Exception as e:
                self.logger.log("Error extracting points from buffer", e, caller=self)
                raise e
        if points is None:
            assert buffer is not None, "This should not be possible."
            points = self.buffer_to_numpy(buffer)
        try:
            mask = np.abs(points[:, 2] - z) <= margin
        except Exception as e:
            self.logger.log("Error creating mask for slicing points", e, caller=self)
            mask = np.zeros(points.shape[0], dtype=bool)
        points = points[mask]

        if plot and points is not None:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color([0, 0.651, 0.929])

                # Add coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

                print("Showing 3D visualization of extracted region...")
                o3d.visualization.draw_geometries( # pyright: ignore[reportAttributeAccessIssue]
                    [pcd, coord_frame], #type: ignore
                    window_name="Extracted 3D Region",
                    width=800, height=600)

            except Exception as e:
                print(f"Error occurred while plotting points: {e}")
                self.logger.log(f"There was an error plotting the results.", e, caller=self)
        return points

    def set_position(
        self,
        x:float,
        y:float,
        z:float,
        roll:float,
        pitch:float,
        yaw:float
    )->None:
        """Set the position and orientation of the camera.
        """
        self.position = np.array([x,y,z])
        self.orientation = np.array([roll,pitch,yaw])

    def get_histogram_peaks(
        self,
        buffer=None,
        raw_depth_values=None,
        plot:bool=True,
        filter_zeros:bool=True,
        threshold:float|np.ndarray=100,
        bins:int=500,
        prominence:None|float|np.ndarray=None
    )->tuple[np.ndarray, np.ndarray]:
        """Creates a histogram of Z-values and finds peaks and their corresponding values.
        Args:
            buffer (arena_api.buffer._Buffer): Buffer to find peaks in. If None, captures a new buffer.
            raw_depth_values (list): List of depth values to find peaks in.
            plot (bool): Enable/disable plotting of intermediate results.
            filter_zeros (bool): Filter out zero values to prevent an uneven histogram.
            threshold (int): Minumum number of points for a bin to be considered a peak.
            bins (int): Number of bins to divide the histrogram in.
            prominence (): Required prominence of peaks. Either a number, ``None``, an array matching `x` or a 2-element sequence of the former. The first element is always interpreted as the  minimal and the second, if supplied, as the maximal required prominence.
        """
        if buffer is None:
            buffer = self.capture_buffer()
        if raw_depth_values is None:
            raw_depth_values, _ = self.get_raw_depth_values(buffer=buffer,plot_histogram=plot, filter_zeros=True)
        raw_depth_values = np.array(raw_depth_values)
        if filter_zeros:
            raw_depth_values = raw_depth_values[raw_depth_values != 0]
        # Calculate histogram
        hist, bin_edges = np.histogram(raw_depth_values, bins=bins)

        # Find peaks in the histogram
        peaks, _ = find_peaks(hist, height=0, threshold=threshold, prominence=prominence)
        peak_values = bin_edges[peaks]
        peak_heights = hist[peaks]
        if plot:
            plt.plot(bin_edges[:-1], hist)
            plt.plot(peak_values, peak_heights, "x")
            plt.title("Histogram of Raw Depth Values")
            plt.xlabel("Depth Value")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
        return peak_values, peak_heights

    def get_closest_surface(
        self,
        buffer:None|_Buffer=None,
        raw_depth_values=None,
        sensitivity:float=20,
        plot_histogram:bool=False,
        lower_threshold:float=600,
        filter_zeros:bool=False
    )-> float | None:
        """Finds the closest surface in a buffer.
        Args:
            buffer (arena_api.buffer._Buffer or None): Buffer to find the closest surface in. If None, captures a new buffer.
            raw_depth_values (list or None): List of Z-values to find the closest values in.
            sensitivity (float): Scalar value controlling the cutoff percentile for determining the average closest surface.
            plot_histogram (bool): Plot histogram of Z-values.
            filter_zeros (bool): Filter out zero_values.
        Returns:
            closest_surface (float): Mean of a portion of the lowest Z-values.
        Raises:
            PointsProcessingError: If no depth values are found or if the depth values array is invalid.
            Helios2Error: If an error occurs during buffer capture.
            ValueError: If both `xy` and `points` are provided or if neither is provided.
            Exception: If an error occurs during determination of the closest surface.
        """
        self.logger.log("Getting closest surface from the camera", caller=self)
        try:
            if raw_depth_values is None:
                raw_depth_values, depth_image = self.get_raw_depth_values(buffer=buffer,plot_histogram=plot_histogram, filter_zeros=True)
                if raw_depth_values is None and depth_image is None:
                    e = PointsProcessingError("No depth values found in the buffer.")
                    self.logger.log("No depth values found in the buffer.", e, caller=self)
                    raise e
            raw_depth_values = np.array(raw_depth_values)
            if filter_zeros:
                raw_depth_values = raw_depth_values[raw_depth_values != 0]
            if lower_threshold > 0:
                raw_depth_values = raw_depth_values[raw_depth_values > lower_threshold]
            n = int(len(raw_depth_values) * (1/sensitivity))
            if n == 0:
                self.logger.log("No depth values found.", caller=self)
                raise PointsProcessingError("No depth values found.")
            try:
                bottom = np.partition(raw_depth_values.flatten(), n)[:n]
            except Exception as e:
                self.logger.log(f"Error in partitioning depth values: {e}", e, caller=self)
                raise PointsProcessingError(f"Error in partitioning depth values: {e}") from e
            # valid_depths = [value for value in bottom if value > 0] # old
            valid_depths = bottom[bottom > 0] # new
            if plot_histogram:
                try:
                    plt.hist(valid_depths, bins=200, color='blue', alpha=0.7)
                    plt.title("Histogram of Raw Depth Values")
                    plt.xlabel("Depth Value")
                    plt.ylabel("Frequency")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.show()
                except Exception as e:
                    print(f"Error in plotting histogram: {e}")
                    return None
            #print(sorted(bottom))
            average_bottom = np.mean(bottom)
            self.logger.log(f"Closest surface found successfully ({average_bottom} mm).", caller=self)
            return float(average_bottom)
        except PointsProcessingError as e:
            self.logger.log(f"Error in get_closest_surface: {e}", e, caller=self)
            raise e
        except Exception as e:
            self.logger.log(f"Unexpected error in get_closest_surface: {e}", e, caller=self)
            raise e

    def get_closest_surface_from_region(
        self,
        xy:None|tuple[float, float]=None,
        points=None,
        buffer:None|_Buffer=None,
        sensitivity:float=20,
        lower_threshold:float=0,
        upper_threshold:float=8000,
        margin:float=250,
        plot:bool=False
    )-> tuple[float, np.ndarray] | tuple[None, None]:
        """Gets the closest surface from a set of points as returned from 'extract_3d_region()'
        Args:
            xy (tuple or None): The (x, y) coordinates to find the closest surface.
            points (np.array or None): Array with shape (n, 3) containing [x,y,z].
            margin (float): Margin to be used when extracting a region around the xy coordinates.
        Returns:
            closest_surface (float): Mean of a portion of the lowest Z-values.
        Raises:
            PointsProcessingError: If no points are found or if the points array is invalid.
            Helios2Error: If an error occurs during buffer capture.
            ValueError: If both `xy` and `points` are provided or if neither is provided.
            Exception: If an error occurs during determination of the closest surface.
        """
        self.logger.log(f"Getting closest surface at {xy} with margin {margin}", caller=self)
        if (buffer is None) and (points is None):
            try:
                buffer = self.capture_buffer()
            except Helios2Error as e:
                self.logger.log(f"Error capturing buffer: {e}", e, caller=self)
                raise Helios2Error(f"Error capturing buffer: {e}") from e
        assert xy is not None
        x,y = xy
        points = self.extract_3d_region(
            buffer=buffer,
            points=points,
            xmin=x-margin,
            xmax=x+margin,
            ymin=y-margin,
            ymax=y+margin,
            zmin=0,
            zmax=10000,
            plot=plot)
        if len(points) == 0:
            e = PointsProcessingError("No points found in the specified region.")
            self.logger.log("No points found in the specified region.", e, caller=self)
            return None, None
        if (len(points) == 0) or points is None:
            e = PointsProcessingError("Points array cannot be empty or None")
            self.logger.log("Points array cannot be empty or None", e, caller=self)
            raise e
        try:
            raw_depth_values = points[:, 2]
            if lower_threshold > 0:
                mask = (raw_depth_values > lower_threshold) & (raw_depth_values < upper_threshold)
            else:
                mask = raw_depth_values < upper_threshold
            raw_depth_values = raw_depth_values[mask]
            n = int(len(raw_depth_values) * (1/sensitivity))
            if n == 0:
                print("No depth values found")
                return None, None
            bottom = np.partition(raw_depth_values.flatten(), n)[:n]
            valid_depths = bottom[bottom > 0]  # Filter out non-positive depths
            average_bottom = float(np.mean(valid_depths))
            self.logger.log(f"Closest surface found successfully ({average_bottom} mm).", caller=self)
            return average_bottom, points
        except Exception as e:
            self.logger.log(f"Error in get_closest_surface_from_region: {e}", e, caller=self)
            raise Exception(f"Error in get_closest_surface_from_region: {e}") from e

    def get_closest_surfaces_from_regions(
        self,
        xy_list:list[tuple[float, float]],
        buffer:None|_Buffer=None,
        sensitivity:float=20,
        lower_threshold:float=600,
        upper_threshold:float=8000,
        margin:float=250,
    )-> list[float] | None:
        """Gets the closest surface to the camera in a specified 3d region.
        Args:
            xy_list (list of tuples): List of (x, y) coordinates to find the closest surfaces.
            buffer (arena_api.buffer._Buffer or None): Buffer to find the closest surfaces in. If None, captures a new buffer.
            sensitivity (float): Scalar value controlling the cutoff percentile for determining the average closest surface.
            lower_threshold (float): Minimum Z-value to consider when finding the closest surface.
            upper_threshold (float): Maximum Z-value to consider when finding the closest surface.
            margin (float): Margin to be used when extracting a region around the xy coordinates.
            plot (bool): Enable/disable plotting of the extracted regions in 3D.
        Returns:
            closest_surfaces (list of float): List of mean of a portion of the lowest Z-values for each (x, y) coordinate.
        Raises:
            PointsProcessingError: If no points are found or if the points array is invalid.
            Helios2Error: If an error occurs during buffer capture.
            ValueError: If `xy_list` is empty.
            Exception: If an error occurs during determination of the closest surfaces.
        """
        closest_surfaces = []
        try:
            for xy in xy_list:
                closest_surface=None
                x, y = xy
                xmin = x - margin
                xmax = x + margin
                ymin = y - margin
                ymax = y + margin
                zmin = lower_threshold
                zmax = upper_threshold

                region = self.extract_3d_region(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zmin,
                    zmax,
                    buffer,
                    plot=False
                )

                if region is not None:
                    raw_depth_values = region[:, 2]
                else:
                    print("No points found in the specified region.")
                    return None
                if lower_threshold > 0:
                    raw_depth_values = raw_depth_values[raw_depth_values > lower_threshold]
                n = int(len(raw_depth_values) * (1/sensitivity))
                if n == 0:
                    print("No depth values found")
                    return None
                bottom = np.partition(raw_depth_values.flatten(), n)[:n]
                valid_depths = bottom[bottom > 0]  # Filter out non-positive depths
                closest_surface = float(np.mean(valid_depths))
                self.logger.log(f"Closest surface found successfully ({closest_surface} mm).", caller=self)
                if closest_surface is not None:
                    closest_surfaces.append(closest_surface)
                else:
                    closest_surfaces.append((None))
        except Exception as e:
            self.logger.log(f"Error in get_closest_surfaces_from_regions: {e}", e, caller=self)
            raise Exception(f"Error in get_closest_surfaces_from_regions: {e}") from e
        return closest_surfaces

    def axis_values_histograms(self, buffer:None|_Buffer=None)->None:
        """Creates histograms of the X, Y and Z axis values in a buffer.
        Args:
            buffer (arena_api.buffer._Buffer or None): Buffer to create histograms from. If None, captures a new buffer.
        Returns:
            None
        """
        if buffer is None:
            buffer = self.capture_buffer()
        try:
            rawx, ximg = self.get_raw_axis_values(0, buffer=buffer, plot_histogram=False, filter_zeros=True, lower_threshold=None, upper_threshold=None)
            rawy, yimg = self.get_raw_axis_values(1, buffer=buffer, plot_histogram=False, filter_zeros=True, lower_threshold=None, upper_threshold=None)
            rawz, zimg = self.get_raw_axis_values(2, buffer=buffer, plot_histogram=False, filter_zeros=True, lower_threshold=None, upper_threshold=None)
        except Exception as e:
            self.logger.log(f"Error getting raw axis values: {e}", e, caller=self)
            raise Helios2Error(f"Error getting raw axis values: {e}") from e

        # checks to stop LSP from complaining
        if rawx is None or rawy is None or rawz is None:
            self.logger.log("No raw axis values found in the buffer.", caller=self)
            raise PointsProcessingError("No raw axis values found in the buffer.")
        if len(rawx) == 0 or len(rawy) == 0 or len(rawz) == 0:
            self.logger.log("Raw axis values arrays cannot be empty.", caller=self)
            raise PointsProcessingError("Raw axis values arrays cannot be empty.")
        if zimg is None or yimg is None or ximg is None:
            self.logger.log("No images found in the buffer.", caller=self)
            raise PointsProcessingError("No images found in the buffer.")
        # plotting the images
        plt.figure(figsize=(10, 8))
        plt.imshow(zimg, cmap='gray')
        plt.colorbar(label='Axis Value (mm)')
        plt.title(f'Axis Z Image')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(yimg, cmap='gray')
        plt.colorbar(label='Axis Value (mm)')
        plt.title(f'Axis Y Image')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(ximg, cmap='gray')
        plt.colorbar(label='Axis Value (mm)')
        plt.title(f'Axis X Image')
        plt.show()

        _, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axs[0].hist(rawx, bins=1000, color='red', alpha=0.7)
        axs[0].set_title("Histogram of X Axis Values")
        axs[0].set_ylabel("Frequency")
        axs[0].grid(True, linestyle='--', alpha=0.7)

        axs[1].hist(rawy, bins=1000, color='green', alpha=0.7)
        axs[1].set_title("Histogram of Y Axis Values")
        axs[1].set_ylabel("Frequency")
        axs[1].grid(True, linestyle='--', alpha=0.7)

        axs[2].hist(rawz, bins=1000, color='blue', alpha=0.7)
        axs[2].set_title("Histogram of Z Axis Values")
        axs[2].set_xlabel("Value")
        axs[2].set_ylabel("Frequency")
        axs[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def get_highest_point(
        self,
        buffer:None|_Buffer=None,
        axis:int=0,
        raw_axis_values=None,
        sensitivity:float=20,
        plot_histogram:bool=False,
        filter_zeros:bool=False,
        lower_threshold:float=300,
        upper_threshold:float=2000
    )-> float | None:
        """
        Gets the highest point in a specified axis from a buffer.
        Args:
            buffer (arena_api.buffer._Buffer or None): Buffer to get the highest point from.
            axis (int): Axis to get the highest point from. 0 for X, 1 for Y, 2 for Z.
            raw_axis_values (list or None): List of axis values to find the highest point in.
            sensitivity (float): Scalar value controlling the cutoff percentile for determining the average highest point.
            plot_histogram (bool): Plot histogram of axis values.
            filter_zeros (bool): Filter out zero values to prevent an uneven histogram.
            lower_threshold (int): Lower threshold for filtering axis values.
            upper_threshold (int): Upper threshold for filtering axis values.
        Returns:
            float: Mean of a portion of the highest points.
        """ 

        if axis == 0:
            print("Getting highest point in x-axis (z-axis perpendicular to the floor)")
        elif axis == 1:
            print("Getting highest point in y-axis (z-axis perpendicular to the floor)")
        elif axis == 2:
            print("Getting highest point in z-axis (z-axis perpendicular to the floor)")
        elif 0 < axis < 3:
            print("axis must be 0 or 1 (x or y).")
            return None

        if raw_axis_values is None:
            raw_axis_values, depth_image = self.get_raw_axis_values(axis, buffer=buffer, plot_histogram=plot_histogram, filter_zeros = filter_zeros, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
            if raw_axis_values is None and depth_image is None:
                print("No depth values found")
                return None
        raw_axis_values = np.array(raw_axis_values)
        if filter_zeros:
            raw_axis_values = raw_axis_values[raw_axis_values != 0]
        n = int(len(raw_axis_values) * (1/sensitivity))
        if n == 0:
            print("No depth values found")
            return None
        bottom = np.partition(raw_axis_values.flatten(), n)[:n]
        valid_depths = [value for value in bottom if value > 0]
        if plot_histogram:
            plt.hist(valid_depths, bins=200, color='blue', alpha=0.7)
            plt.title(f"Histogram of axis-{axis} Values")
            plt.xlabel("Depth Value")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

        #print(sorted(bottom))
        average_bottom = np.mean(bottom)
        return float(average_bottom)

    def get_intensity_image(
        self,
        buffer:None|_Buffer=None,
        logarithmic:bool=False,
        plot:bool=False,
        multiplier:None|float=None
    )-> np.ndarray:
        """Returns intensity image from Coord3D_ABCY16 buffer data.
        Coord3D_ABCY16 format:
        - 4-channel point cloud XYZ + Intensity, 16 bits for each channel, unsigned
        - Channel 0: A (X coordinate)
        - Channel 1: B (Y coordinate) 
        - Channel 2: C (Z coordinate)
        - Channel 3: Y (Intensity)
        """
        if buffer is None:
            try:
                buffer = self.capture_buffer()
            except Helios2Error as e:
                self.logger.log(f"Error capturing buffer:", e, caller=self)
                raise Helios2Error(f"Error capturing buffer: {e}") from e
        axis = 3  # Intensity channel (Y values in ABCY16 format)

        number_of_pixels = buffer.width * buffer.height
        pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))

        # Extract intensity values from channel 3
        intensity_values = [
            pdata_16bit[i * 4 + axis]  # Keep full 16-bit range initially
            for i in range(number_of_pixels)
        ]

        # Reshape to image format
        intensity_image = np.reshape(intensity_values, (buffer.height, buffer.width))

        # Convert to float32 for processing
        intensity_image = intensity_image.astype(np.float32)

        if logarithmic:
            # Apply logarithmic scaling (add small value to avoid log(0))
            intensity_image = np.log10(intensity_image + 1)

        # Normalize the intensity image to 0-1 range for display
        intensity_image = cv2.normalize(intensity_image, intensity_image, 0, 1, cv2.NORM_MINMAX, dtype=-1)

        if multiplier is not None:
            intensity_image = intensity_image * multiplier

        # Now scale to 0-255
        intensity_image = np.clip(intensity_image * 255, 0, 255)
        intensity_image = intensity_image.astype(np.uint8)

        if plot:
            plt.imshow(intensity_image, cmap='hot')
            plt.colorbar(label='intensity')
            plt.title('Intensity Image (Logarithmic Scale)')
            plt.show()

        return intensity_image

    def get_depth_image(
        self,
        buffer:None|_Buffer=None,
        plot:bool=False,
        zmin:float=0,
        zmax:float=3000,
        invert:bool=False
    )-> np.ndarray:
        """Returns depth image from Coord3D_ABCY16 buffer data, with values constrained between zmin and zmax.
        Coord3D_ABCY16 format:
        - 4-channel point cloud XYZ + Intensity, 16 bits for each channel, unsigned
        - Channel 0: A (X coordinate)
        - Channel 1: B (Y coordinate) 
        - Channel 2: C (Z coordinate)
        - Channel 3: Y (Intensity)
        """
        if buffer is None:
            try:
                buffer = self.capture_buffer()
            except Helios2Error as e:
                self.logger.log(f"Error capturing buffer:", e, caller=self)
                raise e
        if zmin >= zmax:
            e = ValueError("zmin must be less than zmax")
            self.logger.log("zmin must be less than zmax", e, caller=self)
            raise e
        axis = 2  # Depth channel (C values in ABCY16 format)
        self.nodemap['Scan3dCoordinateSelector'].value = "CoordinateC"
        scale_z = self.nodemap['Scan3dCoordinateScale'].value
        offset_z = self.nodemap['Scan3dCoordinateOffset'].value

        number_of_pixels = buffer.width * buffer.height
        pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))

        # Extract depth values from channel 2 and apply scale_z
        depth_values = [
            pdata_16bit[i * 4 + axis] * scale_z + offset_z
            for i in range(number_of_pixels)
        ]
        depth_image = np.reshape(depth_values, (buffer.height, buffer.width))
        depth_image = np.clip(depth_image, zmin, zmax)
        depth_image = depth_image.astype(np.float32)
        if invert:
            depth_image = zmax - depth_image + zmin
        if plot:
            plt.imshow(depth_image, cmap='hot')
            plt.colorbar(label='Depth (scaled)')
            plt.title('Depth Image (Scaled by scale_z, constrained)')
            plt.show()
        return depth_image


    def compare_exposures(
        self,
        logarithmic:bool=False,
        multiplier:None|float=None,
        plot:bool=False
    )-> list[np.ndarray]:
        """Creates a figure comparing different exposure settings.
        Args:
            logarithmic (bool): Take log10() of intensity values.
            multiplier (float): Multiplier to apply to the intensity values.
            plot (bool): Enable/disable plotting of the images.
        """
        og_exposure = self.nodemap['ExposureTimeSelector'].value
        og_HDRMode = self.nodemap['Scan3dHDRMode'].value
        self.nodemap['Scan3dHDRMode'].value = 'Off'  # Disable HDR for exposure comparison
        eposures = ['Exp62_5Us', 'Exp250Us', 'Exp1000Us']

        images = []
        for exposure in eposures:
            self.nodemap['ExposureTimeSelector'].value = exposure
            image = self.get_intensity_image(self.capture_buffer(), logarithmic=logarithmic, plot=plot)
            images.append(image)
        self.nodemap['Scan3dHDRMode'].value = 'StandardHDR'  # Restore HDR mode
        image = self.get_intensity_image(self.capture_buffer(), logarithmic=logarithmic, multiplier=multiplier, plot=plot)
        images.append(image)
        self.nodemap['ExposureTimeSelector'].value = og_exposure
        self.nodemap['Scan3dHDRMode'].value = og_HDRMode
        if plot:
            # show plot with the three images
            _, axs = plt.subplots(1, 4, figsize=(15, 5))
            for i, exposure in enumerate(eposures):
                axs[i].imshow(images[i], cmap='hot')
                axs[i].set_title(f'Exposure: {exposure}')
                axs[i].axis('off')
            axs[3].imshow(image, cmap='hot')
            axs[3].set_title(f'Exposure: StandardHDR (logarithmic={logarithmic})')
            plt.tight_layout()
            plt.show()
        return images

    def get_raw_depth_values(
        self,
        buffer:None|_Buffer=None,
        plot_histogram:bool=False,
        filter_zeros:bool = True,
        lower_threshold=None
    )->tuple[list, np.ndarray]:
        """Capture raw depth values from the camera and optionally display a histogram.
        Args:
            buffer (arena_api.buffer._Buffer or None): Buffer to extract depth values from. If None, a new buffer will be captured.
            plot_histogram (bool): If True, display a histogram of the raw depth values.
            filter_zeros (bool): If True, filter out zero values from the depth values.
            lower_threshold (int or None): Lower threshold for filtering depth values. If specified, only depth values greater than this threshold will be returned.
        Returns:
            raw_depth_values (list): List of the depth values in the buffer.
            depth_image (np.ndarray): 2D array containing the depth values in the buffer.
        """
        raw_depth_values, depth_image = None, None
        if buffer == None:
            with self.device.start_stream():
                scale_z = self.nodemap['Scan3dCoordinateScale'].value
                buffer_3d = self.device.get_buffer()
                print('\tbuffer received')
                buffer = BufferFactory.copy(buffer_3d)
                print(f'\tImage buffer requeued')
                self.device.requeue_buffer(buffer_3d)
                self.buffer = buffer
                number_of_pixels = buffer.width * buffer.height
                pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_int16))

                # Extract raw depth values
                raw_depth_values = [
                    int(pdata_16bit[i * 4 + 2] * scale_z)
                    for i in range(number_of_pixels)
                ]

                depth_image = np.reshape(raw_depth_values, (buffer.height, buffer.width))
                print(depth_image.shape)

                if lower_threshold is not None:
                    print(f"Applying lower threshold: {lower_threshold}")
                    raw_depth_values = [value for value in raw_depth_values if value > lower_threshold]
                    depth_image[depth_image < lower_threshold] = 0
                elif filter_zeros:
                    print("zero values filtered")
                    raw_depth_values = [value for value in raw_depth_values if value != 0]

                # Optionally display a histogram
                if plot_histogram:
                    valid_depths = [value for value in raw_depth_values if value > 0]
                    plt.hist(valid_depths, bins=200, color='blue', alpha=0.7)
                    plt.title("Histogram of Raw Depth Values")
                    plt.xlabel("Depth Value")
                    plt.ylabel("Frequency")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.show()
                return raw_depth_values, depth_image
        else:
            scale_z = self.nodemap['Scan3dCoordinateScale'].value

            number_of_pixels = buffer.width * buffer.height
            pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_int16))

            # Extract raw depth values
            raw_depth_values = [
                int(pdata_16bit[i * 4 + 2] * scale_z)
                for i in range(number_of_pixels)
            ]

            depth_image = np.reshape(raw_depth_values, (buffer.height, buffer.width))
            if filter_zeros:
                raw_depth_values = [value for value in raw_depth_values if value != 0]

            if raw_depth_values == []:
                print("No depth values found")
                return raw_depth_values, depth_image

            if plot_histogram:
                valid_depths = [value for value in raw_depth_values if value > 0]
                plt.hist(valid_depths, bins=200, color='blue', alpha=0.7)
                plt.title("Histogram of Raw Depth Values")
                plt.xlabel("Depth Value")
                plt.ylabel("Frequency")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()

            return raw_depth_values, depth_image

    def get_raw_axis_values(
        self,
        axis,
        buffer:None|_Buffer=None,
        plot_histogram:bool=False,
        filter_zeros:bool=True,
        lower_threshold:None|float=None,
        upper_threshold:None|float=None
    )-> tuple[list, np.ndarray] | tuple[None, None]:
        """Extracts raw values for the specified axis (0=X, 1=Y, 2=Z) from the camera buffer. (Z is perpendicular to the floor in this case)
        Args:
            axis (int): Axis to extract values for (0=X, 1=Y, 2=Z).
            buffer (arena_api.buffer._Buffer or None): Buffer to extract values from. If None, a new buffer will be captured.
            plot_histogram (bool): If True, display a histogram of the raw axis values.
            filter_zeros (bool): If True, filter out zero values from the raw axis values.
            lower_threshold (int or None): Lower threshold for filtering axis values.
            upper_threshold (int or None): Upper threshold for filtering axis values.
        """
        if axis not in [0, 1, 2]:
            e = ValueError("Axis must be 0 (X), 1 (Y), or 2 (Z)")
            self.logger.log("Axis must be 0 (X), 1 (Y), or 2 (Z)", e, caller=self)
            raise e

        # Map axis to coordinate selector
        coordinate_selectors = ['CoordinateA', 'CoordinateB', 'CoordinateC']

        # Get buffer if not provided
        if buffer is None:
            with self.device.start_stream():
                # Get scale and offset for the specific coordinate
                self.nodemap['Scan3dCoordinateSelector'].value = coordinate_selectors[axis]
                scale = self.nodemap['Scan3dCoordinateScale'].value
                offset = self.nodemap['Scan3dCoordinateOffset'].value
                print(f'Axis = {axis}, Offset = {offset}, Scale = {scale}')
                buffer_3d = self.device.get_buffer()
                print('\tbuffer received')
                buffer = BufferFactory.copy(buffer_3d)
                print(f'\tImage buffer requeued')
                self.device.requeue_buffer(buffer_3d)
                self.buffer = buffer
        else:
            # Get scale and offset for the specific coordinate
            self.nodemap['Scan3dCoordinateSelector'].value = coordinate_selectors[axis]
            scale = self.nodemap['Scan3dCoordinateScale'].value
            offset = self.nodemap['Scan3dCoordinateOffset'].value
            print(f'Axis = {axis}, Offset = {offset}, Scale = {scale}')

        number_of_pixels = buffer.width * buffer.height
        pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))

        # Extract raw axis values with proper scaling and offset
        raw_axis_values = [
            int(pdata_16bit[i * 4 + axis] * scale + offset)
            for i in range(number_of_pixels)
        ]

        axis_image = np.reshape(raw_axis_values, (buffer.height, buffer.width))
        print(axis_image.shape)

        if filter_zeros:
            print("zero values filtered")
            raw_axis_values = [value for value in raw_axis_values if value != 0]

        if lower_threshold is not None or upper_threshold is not None:
            if lower_threshold is not None:
                print(f"Applying lower threshold: {lower_threshold}")
                axis_image[axis_image < lower_threshold] = 0
            if upper_threshold is not None:
                print(f"Applying upper threshold: {upper_threshold}")
                axis_image[axis_image > upper_threshold] = 0

            raw_axis_values = [
                value for value in raw_axis_values
                if (lower_threshold is None or value > lower_threshold)
                and (upper_threshold is None or value < upper_threshold)
            ]

        if not raw_axis_values:
            print("No axis values found")
            return None, None

        if plot_histogram:
            valid_values = [value for value in raw_axis_values if value > 0]
            if valid_values:
                plt.hist(valid_values, bins=200, color='blue', alpha=0.7)
                plt.title(f"Histogram of Raw Axis {axis} Values")
                plt.xlabel("Axis Value")
                plt.ylabel("Frequency")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()
        return raw_axis_values, axis_image

    def buffer_to_numpy(self, buffer:_Buffer)-> np.ndarray:
        """
        Converts a buffer to a numpy array containing (x,y,z), applying coordinate correction for camera orientation.
        Args: 
            buffer (arena_api.buffer._Buffer or None): Buffer to convert to numpy array. If None, captures a new buffer. 
        Returns:
            np.ndarray: Numpy array with shape (n, 3) containing [x, y, z] coordinates (corrected for camera orientation).
        """
        coordinates = np.array([])
        if buffer.pixel_format != PixelFormat.Coord3D_ABCY16:
            e = ValueError("Buffer pixel format must be Coord3D_ABCY16")
            self.logger.log("Buffer pixel format must be Coord3D_ABCY16", e, caller=self)
            raise e
        try:
            number_of_pixels = buffer.width * buffer.height

            self.nodemap['Scan3dCoordinateSelector'].value = 'CoordinateA'
            scale_x = self.nodemap['Scan3dCoordinateScale'].value
            offset_x = self.nodemap['Scan3dCoordinateOffset'].value

            self.nodemap['Scan3dCoordinateSelector'].value = 'CoordinateB'
            scale_y = self.nodemap['Scan3dCoordinateScale'].value
            offset_y = self.nodemap['Scan3dCoordinateOffset'].value

            self.nodemap['Scan3dCoordinateSelector'].value = 'CoordinateC'
            scale_z = self.nodemap['Scan3dCoordinateScale'].value
            offset_z = self.nodemap['Scan3dCoordinateOffset'].value
        except Exception as e:
            self.logger.log(f"Error getting coordinate scales from Helios2 camera.", e, caller=self)
            raise Helios2Error(f"Error getting coordinate scales from Helios2 Camera {str(e)}") from e
        try:
            pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))
            arr = np.ctypeslib.as_array(pdata_16bit, shape=(number_of_pixels * 4,))
            arr = arr.reshape((number_of_pixels, 4))

            x = arr[:, 0].astype(np.float32) * scale_x + offset_x
            y = arr[:, 1].astype(np.float32) * scale_y + offset_y
            z = arr[:, 2].astype(np.float32) * scale_z + offset_z

            # Filter out invalid points (x, y, z all nonzero)
            valid = (x != 0) & (y != 0) & (z != 0)
            coordinates = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        except Exception as e:
            self.logger.log("Error converting buffer to numpy array", e, caller=self)
            raise e
        if coordinates.size > 0:
            return coordinates
        else:
            print("No valid coordinates found in buffer")
            return np.array([]).reshape(0, 3)

    def get_clusters(self, points, plot=False)-> tuple[np.ndarray, set]:
        """Divide projected points into clusters using DBSCAN
        Args:
            points (np.ndarray): Numpy array with shape (n, 3) containing [x, y, z] coordinates.
            plot (bool): If True, plot the clusters.
        Returns:
            labels (np.ndarray): Array of cluster labels for each point.
            unique_labels (set): Set of unique cluster labels.
        """
        points = np.array(points)
        clustering = DBSCAN(eps=10, min_samples=10, n_jobs=-1)#.fit(points)
        clustering.fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        if plot:
            plt.figure(figsize=(10, 7))
        for label in unique_labels:
            if label == -1:
                # Noise points
                color = 'k'
                marker = 'x'
            else:
                if plot:
                    color = plt.cm.jet(float(label) / max(unique_labels)) #type: ignore
                else:
                    color = 'b'
                marker = 'o'
            cluster_points = points[labels == label]
            if plot:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f"Cluster {label}", marker=marker)

        if plot:
            plt.title("Clusters of Projected Points")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
            plt.show()
        return labels, unique_labels

    def circle_fitting(
        self,
        points,
        labels,
        unique_labels,
        plot=False,
        min_points=100,
        od_percentile=99,
        id_percentile=1,
        return_image=False
    )-> tuple[list, list, np.ndarray | None]:
        """Fit circles to clusters of projected points and calculate outer and inner diameters.
        Args:
            points (np.ndarray): Numpy array with shape (n, 3) containing [x, y, z] coordinates.
            labels (np.ndarray): Array of cluster labels for each point.
            unique_labels (set): Set of unique cluster labels.
            plot (bool): If True, plot the fitted circles and diameters.
            min_points (int): Minimum number of points required to fit a circle.
            max_radius (float): Maximum radius for circle fitting. (Not accessed)
            od_percentile (int): Percentile for calculating outer diameter.
            id_percentile (int): Percentile for calculating inner diameter.
            return_image (bool): If True, return the image with fitted circles.
        Returns:
            outer_diameters (list): List of outer diameters for each cluster.
            inner_diameters (list): List of inner diameters for each cluster.
            image (np.ndarray or None): Image with fitted circles if return_image is True, otherwise None.
        """
        outer_diameters = []
        inner_diameters = []
        image = None

        def fit_circle(points)-> np.ndarray:
            def residuals(params, x, y)->np.ndarray:
                xc, yc, r = params
                return np.sqrt((x - xc)**2 + (y - yc)**2) - r

            x = points[:, 0]
            y = points[:, 1]
            x_m = np.mean(x)
            y_m = np.mean(y)
            r_initial = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))
            initial_guess = [x_m, y_m, r_initial]

            result = least_squares(residuals, initial_guess, args=(x, y))
            return result.x  # xc, yc, r

        def is_circular(
            cluster_points,
            center,
            threshold=0.2,
        )-> bool:
            """
            Check if a cluster is circular by evaluating the standard deviation of distances
            from the points to the center relative to the radius.
            """
            distances = np.sqrt((cluster_points[:, 0] - center[0])**2 + (cluster_points[:, 1] - center[1])**2)
            std_dev = np.std(distances)
            mean_distance = np.mean(distances)
            circularity_ratio = std_dev / mean_distance
            return circularity_ratio < threshold

        if plot or return_image:
            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 0], points[:, 1], c='b', label="Projected Points", s=10)

        circle_params = []  # (od, inner_d, xc, yc, r)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            if len(cluster_points) < min_points:
                print(f"Cluster {label} has insufficient points for circle fitting.")
                continue
            xc, yc, r = fit_circle(cluster_points)
            center = (xc, yc)

            if not is_circular(cluster_points, center, r):
                print(f"Cluster {label} is not circular. Skipping.")
                continue

            distances = cdist(cluster_points, [center]).flatten()
            od = 2 * np.percentile(distances, od_percentile)
            inner_d = 2 * np.percentile(distances, id_percentile)
            outer_diameters.append(od)
            inner_diameters.append(inner_d)
            circle_params.append((od, inner_d, xc, yc, r))
        if plot or return_image:
            if outer_diameters and inner_diameters and circle_params:
                min_outer_diameter = min(outer_diameters)
                min_outer_index = outer_diameters.index(min_outer_diameter)
                od, id_val, xc, yc, r = circle_params[min_outer_index]
                ax = plt.gca()

                circle = Circle((xc, yc), r, color='r', fill=False, linewidth=2, label='Fitted Circle')
                outer_circle = Circle((xc, yc), od/2, color='r', fill=False, linewidth=2, linestyle='--', label='Outer Diameter')
                inner_circle = Circle((xc, yc), id_val/2, color='r', fill=False, linewidth=2, linestyle=':', label='Inner Diameter')
                ax.add_artist(circle)
                ax.add_artist(outer_circle)
                ax.add_artist(inner_circle)
                plt.text(xc, yc + od / 2 + 5, f"OD: {od:.2f}", color='0', fontsize=15, ha='center')
                plt.text(xc, yc + id_val / 2 - 15, f"ID: {id_val:.2f}", color='0', fontsize=15, ha='center')
                # plt.legend()
        if return_image and (plot or return_image):
            plt.gcf().canvas.draw()
            buf = plt.gcf().canvas.print_to_buffer() # type: ignore
            image = np.frombuffer(buf[0], dtype=np.uint8)
            image = image.reshape(buf[1][::-1] + (4,))  # RGBA format
            # Convert RGBA to RGB
            image = image[:, :, :3]
        if plot:
            plt.show()
        plt.close('all')
        if return_image:
            return outer_diameters, inner_diameters, image
        else:
            return outer_diameters, inner_diameters, None

    def get_centers_of_interest_from_config(
        self,
        sleeve_rack_id:str|int,
        config_file:str,
        check:str="diameter"
    )-> list[list[float]]:
        """Gets centers of interest from a configuration file based on the sleeve ID.
        Args:
            sleeve_rack_id (str|int): Unique identifier for the sleeve rack.
            config_file (str): Path to the configuration file in JSON format.
            check (str): Type of check to perform, either "presence" or "diameter
        Returns:
            centers_of_interest (list): List of centers of interest as [x, y].
        Raises:
            ValueError: If the check type is not recognized or if the config file is not provided.
            Exception: If no centers of interest are found in the config file or if there is an error creating ROI's.
        """
        centers_of_interest = []
        match check:
            case "presence":
                if config_file:
                    try:
                        with open(config_file, 'r') as f:
                            data = json.load(f)
                            # Find the sleeve with matching ID
                            for sleeve in data.get("sleeves", []):
                                if sleeve.get("id") == sleeve_rack_id:
                                    centers_of_interest = sleeve.get("sleeve_coordinates", [])
                                    if centers_of_interest == []:
                                        e = Exception("No centers of interest found for sleeve presence in the config file.")
                                        self.logger.log("No centers of interest found for sleeve presence in the config file.", e, caller=self)
                                        raise e
                                    break
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        self.logger.log(f"Error reading config file: {e}", e, caller=self)
                        raise e
                else:
                    e = ValueError("Please provide a config file for sleeve presence check.")
                    self.logger.log("Please provide a config file for sleeve presence check.", e, caller=self)
                    raise e
                if centers_of_interest == []:
                    e = Exception("No centers of interest found in config file.")
                    self.logger.log("No centers of interest found in config file.", e, caller=self)
                    raise e
            case "diameter":
                if config_file:
                    try:
                        with open(config_file, 'r') as f:
                            data = json.load(f)
                            # Find the sleeve with matching ID
                            for sleeve in data.get("sleeves", []):
                                if sleeve.get("id") == sleeve_rack_id:
                                    centers_of_interest = sleeve.get("diameter_measurement", [])
                                    if centers_of_interest == []:
                                        e = Exception("No center of interest found for diameter measurement in the config file.")
                                        self.logger.log("No center of interest found for diameter measurement in the config file.", e, caller=self)
                                        raise e
                                    break
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        self.logger.log(f"Error reading config file: {e}", e, caller=self)
                        raise Exception(f"Error reading config file: {e}") from e
                else:
                    e = ValueError("Please provide a config file for diameter measurement.")
                    self.logger.log("Please provide a config file for diameter measurement.", e, caller=self)
                    raise e
            case _:
                e = ValueError('Choose between check="diameter" or "presence"')
                self.logger.log('Choose between check="diameter" or "presence"', e, caller=self)
                raise e
        return centers_of_interest

    def check_for_sleeve_diameter(
        self,
        sleeve_rack_id:str|int,
        config_file:str,
        xymargin:float,
        zmargin:float,
        buffer:None|_Buffer=None,
        points:None|np.ndarray=None,
        lower_threshold:float=0,
        upper_threshold:float=8000,
        max_sleeve_diameter=300,
        min_sleeve_diameter=100,
        return_image:bool=False,
        plot:bool=False
    )-> tuple[float | None, float | None, np.ndarray | None]:
        """Check for sleeve diameter in buffer at positions provided in `config_file`
        Args:
            sleeve_rack_id (str): ID of the sleeve rack to get the coordinates for in the config file.
            config_file (str): `.json` file containing the calibrated sleeve positions.
            xymargin (float): Margin added to the `max_sleeve_diameter` when defining XY region to consider.
            zmargin (float): Margin for slicing the buffer in Z plane.
            buffer (None|_Buffer): Buffer to check for sleeve diameter. If None, captures a new buffer.
            lower_threshold (float): Lower threshold passed to `get_closest_surface_from_region()`.
            upper_threshold (float): Unused parameter.
            max_sleeve_diameter (float): Maximum sleeve diameter to consider for detection.
            return_image (bool): If True, return the image of the projected points with fitted circles.
            plot (bool): If True, plot the results.
        """
        self.logger.log("Starting sleeve diameter check", caller=self)
        if buffer is None and points is None:
            try:
                buffer = self.capture_buffer()
            except Helios2Error as e:
                self.logger.log(f"Error capturing buffer:", e, caller=self)
                raise Helios2Error(f"Error capturing buffer: {e}") from e
        if not os.path.exists(config_file):
            e = FileNotFoundError(f"Config file {config_file} does not exist.")
            self.logger.log(f"Config file {config_file} does not exist.", e, caller=self)
            raise e
        try:
            centers_of_interest = self.get_centers_of_interest_from_config(sleeve_rack_id, config_file, check="diameter")
        except ValueError as e:
            self.logger.log(f"ValueError for {config_file} and ID {sleeve_rack_id}.", e, caller=self)
            raise e
        except Exception as e:
            self.logger.log(f"Exception occurred while getting regions of interest from config file {config_file} for ID {sleeve_rack_id}.", e, caller=self)
            raise e
        if len(centers_of_interest) > 1:
            e = ValueError("Multiple centers of interest provided. Function only works for one sleeve position. Please confirm config file.")
            self.logger.log("Multiple centers of interest provided. Function only works for one sleeve position. Please confirm config file.", e, caller=self)
            raise e
        if len(centers_of_interest) == 0:
            e = ValueError("No centers of interest provided. Please confirm config file.")
            self.logger.log("No centers of interest provided. Please confirm config file.", e, caller=self)
            raise e
        x, y = centers_of_interest[0] # TODO: FIXME: WARNING: value has been set to 0,0!
        # print("WARNING: center of interest value for diameter checking has been set to 0,0 instead of x, y = centers_of_interest[0]")
        # x, y = 0,0
        try:
            closest_surface, points = self.get_closest_surface_from_region(
                (x,y),
                buffer=buffer,
                points=points,
                sensitivity=50,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                margin=max_sleeve_diameter/2+xymargin,
                plot=plot
            )
        except Exception as e:
            self.logger.log("Exception occurred while getting closest surface from region.", e, caller=self)
            raise e
        if closest_surface is None or points is None:
            print("No closest surface found or no points available.")
            self.logger.log("No closest surface found or no points available.", caller=self)
            return None, None, None
        try:
            roi_points = self.slice(z=closest_surface, points=points, margin=zmargin, plot=plot)
            roi_points = roi_points[:, :2]  # Keep only x and y coordinates
        except Exception as e:
            self.logger.log("Exception occurred while slicing points.", e, caller=self)
            raise e
        labels, unique_labels = self.get_clusters(roi_points, plot=plot)
        outer_diameters, inner_diameters, image = self.circle_fitting(roi_points, labels, unique_labels, plot=plot, return_image=return_image)
        if outer_diameters and inner_diameters:
            min_outer_diameter = min(outer_diameters)
            min_outer_index = outer_diameters.index(min_outer_diameter)
            inner_diameter = inner_diameters[min_outer_index]
            if min_outer_diameter > max_sleeve_diameter*1.2 or min_outer_diameter < min_sleeve_diameter*0.8:
                self.logger.log("No sleeve detected.", caller=self)
                return None, None, None
            else:
                print(f"Sleeve detected with:\n\tOD: {min_outer_diameter}\n\tID: {inner_diameter}")
                self.logger.log(f"Sleeve detected with:\n\tOD: {min_outer_diameter}\n\tID: {inner_diameter}", caller=self)
                return min_outer_diameter, inner_diameter, image
        else:
            print("No sleeve detected.")
            self.logger.log("No sleeve detected.", caller=self)
            return None, None, None

    def get_regions_of_interest_from_config(
        self,
        sleeve_rack_id:str|int,
        config_file:str,
        check:str="diameter"
    )-> list[list[int]]:
        """Gets regions of interest from a configuration file based on the sleeve ID.
        Args:
            id (str): The ID of the sleeve.
            config_file (str): Path to the configuration file in JSON format.
            check (str): Type of check to perform, either "presence" or "diameter
        Returns:
            regions_of_interest (list): List of regions of interest as [xmin, xmax, ymin, ymax].
        Raises:
            ValueError: If the check type is not recognized or if the config file is not provided.
            Exception: If no centers of interest are found in the config file or if there is an error creating ROI's.
        """
        centers_of_interest = []
        regions_of_interest = []
        if check=="presence":
            if config_file:
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                        # Find the sleeve with matching ID
                        for sleeve in data.get("sleeves", []):
                            if sleeve.get("id") == sleeve_rack_id:
                                centers_of_interest = sleeve.get("sleeve_coordinates", [])
                                if centers_of_interest == []:
                                    e = Exception("No centers of interest found for sleeve presence in the config file.")
                                    self.logger.log("No centers of interest found for sleeve presence in the config file.", e, caller=self)
                                    raise e
                                break
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    self.logger.log(f"Error reading config file: {e}", e, caller=self)
                    raise e
            else:
                e = ValueError("Please provide a config file for sleeve presence check.")
                self.logger.log("Please provide a config file for sleeve presence check.", e, caller=self)
                raise e
            if centers_of_interest == []:
                e = Exception("No centers of interest found in config file.")
                self.logger.log("No centers of interest found in config file.", e, caller=self)
                raise e
        elif check=="diameter":
            if config_file:
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                        # Find the sleeve with matching ID
                        for sleeve in data.get("sleeves", []):
                            if sleeve.get("id") == id:
                                centers_of_interest = sleeve.get("diameter_measurement", [])
                                if centers_of_interest == []:
                                    e = Exception("No center of interest found for diameter measurement in the config file.")
                                    self.logger.log("No center of interest found for diameter measurement in the config file.", e, caller=self)
                                    raise e
                                break
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    self.logger.log(f"Error reading config file: {e}", e, caller=self)
                    raise Exception(f"Error reading config file: {e}")
            else:
                e = ValueError("Please provide a config file for diameter measurement.")
                self.logger.log("Please provide a config file for diameter measurement.", e, caller=self)
                raise e
        else:
            e = ValueError('Choose between check="single" or "presence"')
            self.logger.log('Choose between check="single" or "presence"', e, caller=self)
            raise e
        for center in centers_of_interest:
            try:
                xmin = center[0] - 200
                xmax = center[0] + 200
                ymin = center[1] - 200
                ymax = center[1] + 200
                region_of_interest = [int(xmin), int(xmax), int(ymin), int(ymax)]
                regions_of_interest.append(region_of_interest)
            except Exception as e:
                self.logger.log(f"Error processing center {center}:", e, caller=self)
                raise e
        if regions_of_interest == []:
            e = Exception("Failed to create \"regions_of_interest\" from \"centers_of_interest\".")
            self.logger.log("Failed to create \"regions_of_interest\" from \"centers_of_interest\".", e, caller=self)
            raise e
        return regions_of_interest

    def check_for_sleeve_presence(
        self,
        sleeve_rack_id:str|int,
        config_file:str,
        buffer:None|_Buffer=None,
        plot:bool=False,
        margin:float = 100,
        lower_threshold:float=600,
        # upper_threshold:float=10000, # TODO: unused
        min_sleeve_diameter:float=100,
        max_sleeve_diameter:float=300,
        return_image:bool=False
    )-> tuple[list[float] | None, list[np.ndarray] | None]:
        """Check for sleeve presence in buffer at positions provided in `config_file`
        Args:
            sleeve_rack_id (str): ID of the sleeve to check for.
            config_file (str): `.json` file containing the calibrated sleeve positions.
            plot (bool): If True, plot the results.
            margin (int): Margin for slicing the buffer.
            lower_threshold (float): Lower threshold passed to `get_closest_surface()`.
            upper_threshold (float): Unused parameter.
            min_sleeve_diameter (float): Minimum sleeve diameter to consider.
            max_sleeve_diameter (float): Maximum sleeve diameter to consider.
            return_image (bool): If True, return the image with fitted circles.
        Returns:
            tuple: (diameters, image) where:
                - diameters (list): List of detected sleeve diameters for each region of interest.
                - images (list or None): List of np.ndarray images with fitted circles for each region. None if `return_image` is False.
        """
        self.logger.log("Starting sleeve presence check", caller=self)
        if buffer is None:
            try:
                buffer = self.capture_buffer()
            except Helios2Error as e:
                self.logger.log(f"Error capturing buffer:", e, caller=self)
                raise e
        # if sleeve_rack_id is None: # utilities/Helios2.py|1476-1478 col 13-20 note| Type analysis indicates code is unreachable
        #     e = ValueError("ID must be provided to check for sleeve presence")
        #     self.logger.log("ID must be provided to check for sleeve presence", e, caller=self)
        #     raise e
        # if config_file is None or not isinstance(config_file, str): # utilities/Helios2.py|1480-1482 col 13-20 note| Type analysis indicates code is unreachable
        #     e = ValueError("Config file must be provided as a string path to check for sleeve presence")
        #     self.logger.log("Config file must be provided as a string path to check for sleeve presence", e, caller=self)
        #     raise e
        try:
            regions_of_interest = self.get_regions_of_interest_from_config(sleeve_rack_id, config_file, check="presence")
        except ValueError as e:
            self.logger.log(f"ValueError for {config_file} and ID {sleeve_rack_id}.", e, caller=self)
            raise e
        except Exception as e:
            self.logger.log(f"Exception occurred while getting regions of interest from config file {config_file} for ID {sleeve_rack_id}.", e, caller=self)
            raise e
        try:
            closest_surface = self.get_closest_surface(buffer=buffer, lower_threshold=lower_threshold, filter_zeros=True)
        except Exception as e:
            self.logger.log("Exception occurred while getting closest surface.", e, caller=self)
            raise e
        if closest_surface is None:
            self.logger.log("No closest surface found.", caller=self)
            return None, None
        try:
            slice_points = self.slice(buffer=buffer, z=closest_surface, margin=margin, plot=plot)
        except Exception as e:
            self.logger.log("Exception occurred while slicing points.", e, caller=self)
            raise e
        projected_points = slice_points[:, :2]

        projected_points = np.array(projected_points)
        regions = np.array(regions_of_interest)  # shape: (N, 4)
        x_min = regions[:, 0][:, None]
        x_max = regions[:, 1][:, None]
        y_min = regions[:, 2][:, None]
        y_max = regions[:, 3][:, None]

        mask = (
            (projected_points[:, 0] >= x_min) & (projected_points[:, 0] <= x_max) &
            (projected_points[:, 1] >= y_min) & (projected_points[:, 1] <= y_max)
        )

        diameters = []
        images = []
        for i, region_mask in enumerate(mask):
            roi_points = projected_points[region_mask]
            if roi_points.shape[0] == 0:
                print(f"No points found in region: {regions_of_interest[i]}")
                diameters.append(None)
                continue
            labels, unique_labels = self.get_clusters(roi_points, plot=plot)
            if return_image:
                outer_diameters, _, region_image = self.circle_fitting(roi_points, labels, unique_labels, plot=plot, od_percentile=98, id_percentile=1, return_image=return_image)
                images.append(region_image)
            else:
                outer_diameters, _, _ = self.circle_fitting(roi_points, labels, unique_labels, plot=plot, od_percentile=98, id_percentile=1, return_image=return_image)
            if outer_diameters:
                min_outer_diameter = min(outer_diameters)
                if min_outer_diameter > max_sleeve_diameter*1.2 or min_outer_diameter < min_sleeve_diameter*0.8:
                    print("No sleeve detected.")
                    diameters.append(None)
                    continue
                else:
                    print(f"Sleeve detected with size {min_outer_diameter}")
                    diameters.append(min_outer_diameter)
            else:
                print("No sleeve detected.")
                diameters.append(None)
        if return_image:
            self.logger.log("Sleeve presence check completed successfully", caller=self)
            return diameters, images
        else:
            self.logger.log("Sleeve presence check completed successfully", caller=self)
            return diameters, None

    # NOTE: Checks for sleeve presence without checking diameter, using point cloud comparison.
    def check_for_sleeve_presence2(
        self,
        sleeve_rack_id:str|int,
        config_file:str,
        points:np.ndarray|None = None,
        plot:bool = False,
        number_of_points_threshold:int = 900,
        margin:float = 100,
        # lower_threshold:float = 600, # TODO: not accessed
        # upper_threshold:float = 10000, # TODO: notaccessed
        # min_sleeve_diameter:float = 100, # TODO: not accessed
        max_sleeve_diameter:float = 300,
        # return_image:bool = False # TODO: not accessed
    )-> tuple[ list[bool]|None, np.ndarray|None ]:
        """Check for sleeve presence using point count thresholding method.
        
        Args:
            sleeve_rack_id (str|int): ID of the sleeve rack to check
            config_file (str): Path to configuration file with sleeve positions
            plot (bool): Whether to display plots for debugging
            number_of_points_threshold (int): Minimum number of points to consider sleeve present
            margin (float): Margin for 3D region extraction
            lower_threshold (float): Lower threshold for filtering depth values
            upper_threshold (float): Upper threshold for filtering depth values (unused)
            min_sleeve_diameter (float): Minimum sleeve diameter to consider
            max_sleeve_diameter (float): Maximum sleeve diameter to consider
            return_image (bool): Whether to return visualization images (unused)
            
        Returns:
            list[bool] | None: List of boolean values indicating sleeve presence for each center of interest
        """
        image = None # TODO: return image
        self.logger.log("Starting sleeve presence check (method 2)", caller=self)
        
        try:
            centers_of_interest = self.get_centers_of_interest_from_config(
                sleeve_rack_id, config_file, check="presence"
            )
        except Exception as e:
            self.logger.log(f"Error getting centers of interest: {e}", e, caller=self)
            return None, None
        
        if not centers_of_interest:
            self.logger.log("No centers of interest found", caller=self)
            return None, None
        
        sleeve_presence:list[bool] = []
        
        for i, center in enumerate(centers_of_interest):
            try:
                x, y = center
                self.logger.log(f"Checking center {i+1}/{len(centers_of_interest)}: ({x}, {y})", caller=self)  
                
                region_size = max_sleeve_diameter + margin
                xmin = x - region_size / 2
                xmax = x + region_size / 2
                ymin = y - region_size / 2
                ymax = y + region_size / 2
                zmin = 600
                zmax = 8300

                region_points = self.extract_3d_region(
                    points=points,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    zmin=zmin,
                    zmax=zmax,
                    plot=plot
                )
                
                num_points = len(region_points) if region_points is not None else 0

                self.logger.log(f"Center {i+1}: Found {num_points} points (threshold: {number_of_points_threshold})", caller=self)
                
                if num_points > number_of_points_threshold:
                    sleeve_presence.append(True)
                    self.logger.log(f"Center {i+1}: Sleeve PRESENT ({num_points} points)", caller=self)
                else:
                    sleeve_presence.append(False)
                    self.logger.log(f"Center {i+1}: Sleeve ABSENT ({num_points} points)", caller=self)
                    
            except Exception as e:
                self.logger.log(f"Error processing center {i+1}: {e}", e, caller=self)
                raise e
        
        self.logger.log(f"Sleeve presence check completed. Results: {sleeve_presence}", caller=self)
        return sleeve_presence, image

    def check_for_sleeve_presence3(
        self,
        sleeve_rack_id:str|int,
        config_file:str,
        points:np.ndarray|None = None,
        plot:bool = False,
        number_of_points_threshold:int = 900,
        margin:float = 100,
        # lower_threshold:float = 600, # TODO: not accessed
        # upper_threshold:float = 10000, # TODO: notaccessed
        # min_sleeve_diameter:float = 100, # TODO: not accessed
        max_sleeve_diameter:float = 300,
        # return_image:bool = False # TODO: not accessed
    )-> tuple[ list[bool]|None, np.ndarray|None ]:
        """Check for sleeve presence using cylindrical point count thresholding method.
        
        Args:
            sleeve_rack_id (str|int): ID of the sleeve rack to check
            config_file (str): Path to configuration file with sleeve positions
            plot (bool): Whether to display plots for debugging
            number_of_points_threshold (int): Minimum number of points to consider sleeve present
            margin (float): Margin for 3D cylindrical region extraction
            lower_threshold (float): Lower threshold for filtering depth values
            upper_threshold (float): Upper threshold for filtering depth values (unused)
            min_sleeve_diameter (float): Minimum sleeve diameter to consider
            max_sleeve_diameter (float): Maximum sleeve diameter to consider
            return_image (bool): Whether to return visualization images (unused)
            
        Returns:
            list[bool] | None: List of boolean values indicating sleeve presence for each center of interest
        """
        image = None # TODO: return image
        self.logger.log("Starting sleeve presence check (method 3 - cylindrical)", caller=self)
        
        try:
            centers_of_interest = self.get_centers_of_interest_from_config(
                sleeve_rack_id, config_file, check="presence"
            )
        except Exception as e:
            self.logger.log(f"Error getting centers of interest: {e}", e, caller=self)
            return None, None
        
        if not centers_of_interest:
            self.logger.log("No centers of interest found", caller=self)
            return None, None
        
        sleeve_presence:list[bool] = []
        
        for i, center in enumerate(centers_of_interest):
            try:
                x, y = center
                self.logger.log(f"Checking center {i+1}/{len(centers_of_interest)}: ({x}, {y})", caller=self)  
                
                radius = (max_sleeve_diameter + margin) / 2
                zmin = 600
                zmax = 8300

                region_points = self.extract_3d_region_cylindrical(
                    points=points,
                    center_x=x,
                    center_y=y,
                    radius=radius,
                    zmin=zmin,
                    zmax=zmax,
                    plot=plot  # Only plot for first region to avoid too many plots
                )
                
                num_points = len(region_points) if region_points is not None else 0
                
                self.logger.log(f"Center {i+1}: Found {num_points} points (threshold: {number_of_points_threshold})", caller=self)
                
                if num_points > number_of_points_threshold:
                    sleeve_presence.append(True)
                    self.logger.log(f"Center {i+1}: Sleeve PRESENT ({num_points} points)", caller=self)
                else:
                    sleeve_presence.append(False)
                    self.logger.log(f"Center {i+1}: Sleeve ABSENT ({num_points} points)", caller=self)
                    
            except Exception as e:
                self.logger.log(f"Error processing center {i+1}: {e}", e, caller=self)
                raise e
        
        self.logger.log(f"Sleeve presence check completed. Results: {sleeve_presence}", caller=self)
        return sleeve_presence, image

    def extract_3d_region(
        self,
        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        zmin:float,
        zmax:float,
        buffer:None|_Buffer=None,
        points=None,
        plot:bool=False,
        save_path:None|str=None
    )-> np.ndarray:
        """
        Removes all points outside of a 3D region from the buffer or provided points array.
        Returns a numpy array of shape (N, 3) with points inside the region, applying coordinate correction for camera orientation.
        
        Args:
            xmin, xmax, ymin, ymax, zmin, zmax: Region boundaries
            buffer: Buffer containing 3D coordinate data (assumes Coord3D_ABCY16 format). 
                   Used only if points is None.
            points: Numpy array of shape (N, 3) with pre-existing 3D points [x, y, z].
                   If provided, buffer parameter is ignored.
            plot (bool): If True, show 3D visualization of extracted points
            save_path (str): Optional path to save the extracted points as PLY file
        Returns:
            np.ndarray: Numpy array of shape (N, 3) with points inside the specified 3D region (corrected for camera orientation).
        Raises:
            Helios2Error: If nodemap is not available or buffer pixel format is incorrect (buffer mode only).
            PointsProcessingError: If there is an error processing the points.
            Exception: If there is an error visualizing/saving the points.
        """
        if points is not None:
            points = np.array(points) if not isinstance(points, np.ndarray) else points
            
            outside_points = []
            filtered_points = []
            
            try:
                for point in points:
                    x, y, z = point[0], point[1], point[2]
                    if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
                        filtered_points.append([x, y, z])
                    elif plot:
                        outside_points.append([x, y, z])
                
                points = np.array(filtered_points)
            except Exception as e:
                self.logger.log(f"Error filtering provided points: {e}", e, caller=self)
                raise PointsProcessingError(f"Error filtering provided points: {e}") from e
        else:
            if buffer is None:
                try:
                    buffer = self.capture_buffer()
                except Helios2Error as e:
                    self.logger.log(f"Error capturing buffer:", e, caller=self)
                    raise e
            
            width = 640
            height = 480
            num_pixels = width * height

            nodemap = self.nodemap if hasattr(self, "nodemap") else None
            if nodemap is not None:
                nodemap['Scan3dCoordinateSelector'].value = 'CoordinateA'
                scale_x = nodemap['Scan3dCoordinateScale'].value
                offset_x = nodemap['Scan3dCoordinateOffset'].value
                nodemap['Scan3dCoordinateSelector'].value = 'CoordinateB'
                scale_y = nodemap['Scan3dCoordinateScale'].value
                offset_y = nodemap['Scan3dCoordinateOffset'].value
                nodemap['Scan3dCoordinateSelector'].value = 'CoordinateC'
                scale_z = nodemap['Scan3dCoordinateScale'].value
                offset_z = nodemap['Scan3dCoordinateOffset'].value
            else:
                e = Helios2Error("Nodemap is not available. Cannot get scale and offset values.")
                self.logger.log("Nodemap is not available. Cannot get scale and offset values.", e, caller=self)
                raise e
            
            outside_points = []
            points = []
            
            try: 
                pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))
            except Exception as e:
                self.logger.log("Error casting buffer data to ctypes. Ensure buffer is of type Coord3D_ABCY16.", e, caller=self)
                raise e
            try:
                for i in range(num_pixels):
                    x = int(pdata_16bit[i * 4 + 0]) * scale_x + offset_x
                    y = int(pdata_16bit[i * 4 + 1]) * scale_y + offset_y
                    z = int(pdata_16bit[i * 4 + 2]) * scale_z + offset_z

                    if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
                        points.append([x, y, z])
                    elif plot:
                        outside_points.append([x,y,z])

                points = np.array(points)
            except Exception as e:
                self.logger.log(f"Error extracting points from buffer: {e}", e, caller=self)
                raise PointsProcessingError(f"Error extracting points from buffer: {e}") from e
        if len(points) == 0:
            e = PointsProcessingError("No points found in the specified 3D region.")
            self.logger.log("No points found in the specified 3D region.", e, caller=self)
        #             raise e

        print(f"Extracted {len(points)} points from 3D region (corrected)")

        # Visualization
        if plot and (len(points) > 0 or len(outside_points) > 0):
            try:
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color([0, 0.651, 0.929])
                else:
                    pcd = None

                outside_points = np.array(outside_points)
                if len(outside_points) > 0:
                    pcd_outside = o3d.geometry.PointCloud()
                    pcd_outside.points = o3d.utility.Vector3dVector(outside_points)
                    pcd_outside.paint_uniform_color([1, 0, 0])
                else:
                    pcd_outside = None

                bbox_points = np.array([
                    [xmin, ymin, zmin], [xmax, ymin, zmin],
                    [xmin, ymax, zmin], [xmax, ymax, zmin],
                    [xmin, ymin, zmax], [xmax, ymin, zmax],
                    [xmin, ymax, zmax], [xmax, ymax, zmax]
                ])
                bbox_lines = [
                    [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
                    [4, 5], [5, 7], [7, 6], [6, 4],  # top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
                ]
                bbox = o3d.geometry.LineSet()
                bbox.points = o3d.utility.Vector3dVector(bbox_points)
                bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
                bbox.paint_uniform_color([1, 0, 0])

                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

                print("Showing 3D visualization of extracted region...")
                geometries = [bbox, coord_frame]
                if pcd is not None:
                    geometries.insert(1, pcd) 
                if pcd_outside is not None:
                    geometries.insert(1, pcd_outside)
                o3d.visualization.draw_geometries( # pyright: ignore[reportAttributeAccessIssue]
                    geometries, #type: ignore
                    window_name="Extracted 3D Region",
                    width=800, height=600
                )
            except Exception as e:
                self.logger.log(f"Error during visualization: {e}", e, caller=self)
                raise Exception(f"Error during visualization: {e}") from e

        if save_path and len(points) > 0:
            try:
                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(save_path, pcd_save)
                print(f"Extracted points saved to {save_path}")
            except Exception as e:
                self.logger.log(f"Error saving extracted points: {e}", e, caller=self)
                raise Exception(f"Error saving extracted points: {e}") from e
        try:
            plt.clf()
        except Exception as e:
            self.logger.log("Failed to close figures", e, caller=__name__)
            raise e
        return points

    def extract_3d_region_cylindrical(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        zmin: float,
        zmax: float,
        buffer: None | _Buffer = None,
        points=None,
        plot: bool = False,
        save_path: None | str = None
    ) -> np.ndarray:
        """
        Removes all points outside of a cylindrical 3D region from the buffer or provided points array.
        Returns a numpy array of shape (N, 3) with points inside the cylindrical region.
        
        Args:
            center_x, center_y: Center coordinates of the cylinder base
            radius: Radius of the cylinder
            zmin, zmax: Height boundaries of the cylinder
            buffer: Buffer containing 3D coordinate data (assumes Coord3D_ABCY16 format). 
                   Used only if points is None.
            points: Numpy array of shape (N, 3) with pre-existing 3D points [x, y, z].
                   If provided, buffer parameter is ignored.
            plot (bool): If True, show 3D visualization of extracted points
            save_path (str): Optional path to save the extracted points as PLY file
        Returns:
            np.ndarray: Numpy array of shape (N, 3) with points inside the specified cylindrical region.
        Raises:
            Helios2Error: If nodemap is not available or buffer pixel format is incorrect (buffer mode only).
            PointsProcessingError: If there is an error processing the points.
            Exception: If there is an error visualizing/saving the points.
        """
        if points is not None:
            points = np.array(points) if not isinstance(points, np.ndarray) else points
            
            outside_points = []
            filtered_points = []
            
            try:
                for point in points:
                    x, y, z = point[0], point[1], point[2]
                    distance_from_axis = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance_from_axis <= radius and zmin <= z <= zmax:
                        filtered_points.append([x, y, z])
                    elif plot:
                        outside_points.append([x, y, z])
                
                points = np.array(filtered_points)
            except Exception as e:
                self.logger.log(f"Error filtering provided points: {e}", e, caller=self)
                raise PointsProcessingError(f"Error filtering provided points: {e}") from e
        else:
            if buffer is None:
                try:
                    buffer = self.capture_buffer()
                except Helios2Error as e:
                    self.logger.log(f"Error capturing buffer:", e, caller=self)
                    raise e
            
            width = 640
            height = 480
            num_pixels = width * height

            nodemap = self.nodemap if hasattr(self, "nodemap") else None
            if nodemap is not None:
                nodemap['Scan3dCoordinateSelector'].value = 'CoordinateA'
                scale_x = nodemap['Scan3dCoordinateScale'].value
                offset_x = nodemap['Scan3dCoordinateOffset'].value
                nodemap['Scan3dCoordinateSelector'].value = 'CoordinateB'
                scale_y = nodemap['Scan3dCoordinateScale'].value
                offset_y = nodemap['Scan3dCoordinateOffset'].value
                nodemap['Scan3dCoordinateSelector'].value = 'CoordinateC'
                scale_z = nodemap['Scan3dCoordinateScale'].value
                offset_z = nodemap['Scan3dCoordinateOffset'].value
            else:
                e = Helios2Error("Nodemap is not available. Cannot get scale and offset values.")
                self.logger.log("Nodemap is not available. Cannot get scale and offset values.", e, caller=self)
                raise e
            
            outside_points = []
            points = []
            
            try: 
                pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))
            except Exception as e:
                self.logger.log("Error casting buffer data to ctypes. Ensure buffer is of type Coord3D_ABCY16.", e, caller=self)
                raise e
            try:
                for i in range(num_pixels):
                    x = int(pdata_16bit[i * 4 + 0]) * scale_x + offset_x
                    y = int(pdata_16bit[i * 4 + 1]) * scale_y + offset_y
                    z = int(pdata_16bit[i * 4 + 2]) * scale_z + offset_z

                    distance_from_axis = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance_from_axis <= radius and zmin <= z <= zmax:
                        points.append([x, y, z])
                    elif plot:
                        outside_points.append([x, y, z])

                points = np.array(points)
            except Exception as e:
                self.logger.log(f"Error extracting points from buffer: {e}", e, caller=self)
                raise PointsProcessingError(f"Error extracting points from buffer: {e}") from e

        if len(points) == 0:
            e = PointsProcessingError("No points found in the specified cylindrical region.")
            self.logger.log("No points found in the specified cylindrical region.", e, caller=self)

        print(f"Extracted {len(points)} points from cylindrical region (center: ({center_x}, {center_y}), radius: {radius}, z: [{zmin}, {zmax}])")

        # Visualization
        if plot and (len(points) > 0 or len(outside_points) > 0):
            try:
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color([0, 0.651, 0.929])
                else:
                    pcd = None

                outside_points = np.array(outside_points)
                if len(outside_points) > 0:
                    pcd_outside = o3d.geometry.PointCloud()
                    pcd_outside.points = o3d.utility.Vector3dVector(outside_points)
                    pcd_outside.paint_uniform_color([1, 0, 0])
                else:
                    pcd_outside = None

                # Create cylinder wireframe for visualization
                cylinder_points = []
                cylinder_lines = []
                
                # Create circles at top and bottom of cylinder
                theta = np.linspace(0, 2*np.pi, 32)
                for z_level in [zmin, zmax]:
                    circle_start_idx = len(cylinder_points)
                    for angle in theta:
                        x_circle = center_x + radius * np.cos(angle)
                        y_circle = center_y + radius * np.sin(angle)
                        cylinder_points.append([x_circle, y_circle, z_level])
                    
                    # Connect circle points
                    for i in range(len(theta) - 1):
                        cylinder_lines.append([circle_start_idx + i, circle_start_idx + i + 1])
                    # Close the circle
                    cylinder_lines.append([circle_start_idx + len(theta) - 1, circle_start_idx])
                
                # Connect top and bottom circles with vertical lines
                for i in range(len(theta)):
                    cylinder_lines.append([i, i + len(theta)])

                cylinder_wireframe = o3d.geometry.LineSet()
                cylinder_wireframe.points = o3d.utility.Vector3dVector(cylinder_points)
                cylinder_wireframe.lines = o3d.utility.Vector2iVector(cylinder_lines)
                cylinder_wireframe.paint_uniform_color([1, 0, 0])

                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

                print("Showing 3D visualization of extracted cylindrical region...")
                geometries = [cylinder_wireframe, coord_frame]
                if pcd is not None:
                    geometries.insert(1, pcd) 
                if pcd_outside is not None:
                    geometries.insert(1, pcd_outside)
                o3d.visualization.draw_geometries( # pyright: ignore[reportAttributeAccessIssue]
                    geometries, #type: ignore
                    window_name="Extracted Cylindrical 3D Region",
                    width=800, height=600)

            except Exception as e:
                self.logger.log(f"Error during visualization: {e}", e, caller=self)
                raise Exception(f"Error during visualization: {e}") from e

        if save_path and len(points) > 0:
            try:
                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(save_path, pcd_save)
                print(f"Extracted points saved to {save_path}")
            except Exception as e:
                self.logger.log(f"Error saving extracted points: {e}", e, caller=self)
                raise Exception(f"Error saving extracted points: {e}") from e
        
        # Clean up plots
        try:
            plt.clf()
        except Exception as e:
            self.logger.log("Failed to close figures", e, caller=__name__)
            raise e
        
        return points

    def create_depth_image_from_points(
        self,
        projected_points,
        z_values,
    )-> np.ndarray | None:
        """
        Create a grayscale depth image from projected points with Z values, and does not make use of alphashapes.
        Args:
            projected_points: Numpy array of shape (N, 2) containing X and Y coordinates of points.
            z_values: Numpy array of shape (N,) containing Z coordinates (depth values) of points.
        Returns:
            depth_image: Grayscale depth image with pixel values based on Z-range (0-255).
        """
        if len(projected_points) == 0:
            print("No projected points provided.")
            return None

        if len(z_values) != len(projected_points):
            print("Number of Z values must match number of projected points.")
            return None

        projected_points = np.array(projected_points)
        z_values = np.array(z_values)

        x_min, x_max = np.min(projected_points[:, 0]), np.max(projected_points[:, 0])
        y_min, y_max = np.min(projected_points[:, 1]), np.max(projected_points[:, 1])

        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)

        z_min, z_max = np.min(z_values), np.max(z_values)
        z_range = z_max - z_min

        depth_image = np.zeros((height, width), dtype=np.uint8)

        # for each pixel, get the average Z value
        for i, point in enumerate(projected_points):
            x, y = int(point[0] - x_min), int(point[1] - y_min)
            if 0 <= x < width and 0 <= y < height:
                # Normalize Z value to 0-255 range
                normalized_z = int((z_values[i] - z_min) / z_range * 255)
                depth_image[y, x] = normalized_z
        return depth_image

    def template_match_SQDIFF(
        self,
        image:np.ndarray,
        template:np.ndarray,
        plot:bool=False,
        template_id:None|str|int=None
    )-> tuple[float, np.ndarray]:
        """Template matching using `SQDIFF_NORMED` method.
        Args:
            image: Input image (grayscale)
            template: Template image to match against
            plot: Whether to display debugging plots
            template_id: ID of the template for display purposes
        Returns:
            Tuple of (min_val, result) where:
                - min_val: Minimum match score (lower is better)
                - result: Resulting match image
        """
        result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(result)

        if plot:
            plt.imshow(result, cmap='gray')
            title = f'Template Match Result for ID {template_id}' if template_id else 'Template Match Result'
            plt.title(title)
            plt.colorbar(label='Match Score (lower is better)')
            plt.show()
        return min_val, result

    def template_match_CCOEFF_NORMED(
        self,
        image:np.ndarray,
        template:np.ndarray,
        plot:bool=False,
        template_id:None|str|int=None
    )-> tuple[float, np.ndarray]:
        """Template matching using `CCOEFF_NORMED` method.
        Args:
            image: Input image (grayscale)
            template: Template image to match against
            plot: Whether to display debugging plots
            template_id: ID of the template for display purposes
        Returns:
            Tuple of (max_val, result) where:
                - max_val: Maximum match score (higher is better)
                - result: Resulting match image
        """
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if plot:
            plt.imshow(result, cmap='gray')
            title = f'Template Match Result for ID {template_id}' if template_id else 'Template Match Result'
            plt.title(title)
            plt.colorbar(label='Match Score (higher is better)')
            plt.show()
        return max_val, result

    def template_match_gripper(
        self,
        image:np.ndarray,
        template_set:GripperTemplateSet,
        method:Literal["CCOEFF_NORMED", "SQDIFF_NORMED"] = "CCOEFF_NORMED", # TODO: add methods ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR','TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        match_score_threshold:float=0.9,
        attached:bool=False,
        plot:bool=False
    )-> tuple[None|float, None|str|int]:
        """ Match image against gripper templates using specified method.
        Args:
            image: Input image (grayscale)
            template_set: GripperTemplateSet containing templates to match against
            method: Template matching method to use.
            match_score_threshold: Threshold for considering a match (default: 0.9). Match score is automatically inverted for methods that return low values for good match results.
            attached: Whether to use attached templates (default: False)
            plot: Whether to display debugging plots (default: False)
        Returns:
            Tuple of (best_match_score, best_match_id) where:
                - best_match_score: Best match score found (None if no match)
        """
        best_match = None
        inverted_threshold = False
        try:
            for gripper_id in template_set.get_id_list(attached=attached):
                template = template_set.get_template(gripper_id, attached=attached)
                if template is None:
                    print(f"No template found for ID {gripper_id}. Skipping.")
                    continue

                match method:# TODO: add ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR','TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
                    case 'SQDIFF':
                        self.logger.log("Method SQDIFF is not implemented yet", caller=self)
                        raise NotImplementedError("Method SQDIFF is not implemented yet")
                    case "SQDIFF_NORMED":
                        # Perform template matching using SQDIFF_NORMED (lower is better)
                        match_score_threshold = 1 - match_score_threshold
                        inverted_threshold = True
                        val, result = self.template_match_SQDIFF(image, template, plot=False, template_id=gripper_id)
                        if best_match is None or val < best_match[0]:
                            best_match = (val, gripper_id)
                    case 'CCOEFF':
                        self.logger.log("Method CCOEFF is not implemented yet", caller=self)
                        raise NotImplementedError("Method CCOEFF is not implemented yet")
                    case "CCOEFF_NORMED":
                        # perform template matching using CCOEFF_NORMED (higher is better)
                        val, result = self.template_match_CCOEFF_NORMED(image, template, plot=False, template_id=gripper_id)
                        if best_match is None or val > best_match[0]:
                            best_match = (val, gripper_id)
                    case 'CCORR':
                        self.logger.log("Method CCOR is not implemented yet", caller=self)
                        raise NotImplementedError("Method CCOR is not implemented yet")
                    case 'CCORR_NORMED':
                        self.logger.log("Method CCOR_NORMED is not implemented yet", caller=self)
                        raise NotImplementedError("Method CCOR_NORMED is not implemented yet")
                    case _:
                        self.logger.log(f"Unsupported matching method: {method}", caller=self)
                        raise ValueError(f"Unsupported matching method: {method}")
                self.logger.log(f"Template matching for ID {gripper_id} completed with score {val}", caller=self)
                if plot: 
                    plt.imshow(result, cmap='gray')
                    plt.title(f'Template Match Result for ID {gripper_id}')
                    plt.colorbar(label='Match Score (lower is better)')
                    plt.show()

        except Exception as e:
            self.logger.log(f"Error during template matching: {e}", e, caller=self)
            raise e

        match_found = False
        if best_match is not None:
            match inverted_threshold:
                case True:
                    # For SQDIFF_NORMED, lower is better
                    match_found = best_match[0] <= match_score_threshold
                case False:
                    # For CCOEFF_NORMED, higher is better
                    match_found = best_match[0] >= match_score_threshold
        else:
            self.logger.log("No matches found during template matching.", caller=self)
            return None, None
        if match_found:
            print(f"Best match found: {best_match[1]} with score {best_match[0]}")
            self.logger.log(f"Best match found: {best_match[1]} with score {best_match[0]}", caller=self)
            return best_match
        else:
            print(f"No match found with score better than {match_score_threshold}. Best score: {best_match[0] if best_match else None}")
            self.logger.log(f"No match found with score better than {match_score_threshold}. Best score: {best_match[0] if best_match else None}", caller=self)
            return None, None

    def feature_match_gripper(
        self,
        image:np.ndarray,
        template_set:GripperTemplateSet,
        plot:bool=False
    )-> tuple[float, str|int|None] | tuple[None, None]:
        """
        Match image features against the loaded templates using SIFT and return best matches.
        Args:
            img: Input image (grayscale)
            plot: Whether to display debugging plots
        Returns:
            List of (template_name, score) tuples sorted by match quality
        """
        # Ensure the image is grayscalel TODO: test
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) != 2:
            print("Input image must be grayscale or BGR format")
            return None, None
        sift = cv2.SIFT_create( #type:ignore
            nfeatures=500,
            contrastThreshold=0.01,
            edgeThreshold=20,
            sigma=1.6
        )

        kp_img, des_img = sift.detectAndCompute(image, None)

        if des_img is None or len(kp_img) < 5:
            print("Not enough features found in the input image")
            return None, None

        # Initialize ratio test threshold (Lowe's ratio test)
        ratio_threshold = 0.75

        # Use FLANN matcher for faster matching with many templates
        FLANN_INDEX_KDTREE = 1
        index_params:dict[str, Union[bool, int, float, str]] = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params:dict[str, Union[bool, int, float, str]] = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Store results as (template_name, match_score) tuples
        match_results = []

        if plot:
            img_with_kp = cv2.drawKeypoints(image, kp_img, None,   # type: ignore
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(img_with_kp)
            plt.title(f"Input Image: {len(kp_img)} keypoints")

        for template_id in template_set.get_id_list():
            template = template_set.get_template(template_id)
            kp_template, des_template = sift.detectAndCompute(template, None)

            if des_template is None or len(kp_template) < 3:
                print(f"Not enough features in template {template_id}")
                continue

            if len(des_template) >= 2 and len(des_img) >= 2:
                matches = flann.knnMatch(des_template, des_img, k=2)

                # Apply ratio test to filter good matches
                good_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)

                match_score = len(good_matches) / max(len(kp_template), 1)
                match_results.append((template_id, match_score, good_matches, kp_template, kp_img))

                if plot and len(good_matches) > 0:
                    print(f"Template '{template_id}': {len(good_matches)}/{len(kp_template)} matches, score: {match_score:.3f}")

        # Sort results by match score (descending)
        match_results.sort(key=lambda x: x[1], reverse=True)

        if plot and match_results:
            best_match = match_results[0]
            template_id, score, good_matches, kp_template, kp_img = best_match
            template = template_set.get_template(template_id)

            plt.subplot(132)
            temp_with_kp = cv2.drawKeypoints(
                template, kp_template, None,   # type: ignore
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(temp_with_kp, cmap='gray')
            plt.title(f"Template '{template_id}': {len(kp_template)} keypoints")

            plt.subplot(133)
            match_img = cv2.drawMatches(template, kp_template, image, kp_img, # type: ignore
                                        good_matches[:30], None, flags=2) # type: ignore
            plt.imshow(match_img)
            plt.title(f"Best Match: {template_id} (Score: {score:.3f})")
            plt.tight_layout()
            plt.show()

        if match_results:
            best_match = match_results[0]
            return best_match[1], best_match[0]  # Return score and template ID
        else:
            print("No matches found")
            return 0, None        

    def check_gripper(
        self,
        buffer:_Buffer,
        template_set:GripperTemplateSet | None,
        xy:tuple[float, float],
        z:float,
        plot:bool=False,
        capture_type:Literal["intensity", "depth"] = "intensity"
    )-> tuple[
        None | tuple[None|float,None|str|int],
        None | np.ndarray,
        None | np.ndarray, 
        None | tuple[int, int, int, int]
    ]:
        """
        Checks for the presence and position of a gripper in a specified 3D region of an image buffer using template matching.
        Args:
            buffer: The image buffer containing intensity and depth data.
            template_set (GripperTemplateSet): The set of gripper templates to use for matching.
            xy (tuple): The (x, y) coordinates of the center of the region to check.
            z (float): The z-coordinate (depth) of the center of the region to check.
            plot (bool, optional): Whether to plot the masked image. Defaults to False.
            type (str, optional): The type of image to use for matching, either "intensity" or "depth". Defaults to "intensity".
        Returns:
            tuple:
                - best_match: The result of the template matching (or None if no template set is provided).
                - binary_image (np.ndarray): Binary mask of the region of interest (shape: (buffer.height, buffer.width)).
                - image_masked (np.ndarray): The masked image (shape: (buffer.height, buffer.width)), type depends on `type` argument.
                - mask_bounding_box (tuple): Bounding box of the masked region as (xmin, xmax, ymin, ymax).
        """
        best_match, binary_image, image_masked, mask_bounding_box = None, None, None, None
        xymargin = 100
        x, y = xy
        xmin, xmax, ymin, ymax = x - xymargin, x + xymargin, y - xymargin, y + xymargin
        zmin, zmax = z - 50, z + 50
        #intensity_image = self.get_intensity_image(buffer, logarithmic=False, multiplier=1,  plot=plot)

        try:
            match capture_type:
                case "intensity":
                    image = self.get_intensity_image(buffer, logarithmic=False, multiplier=1, plot=plot)
                case "depth":
                    image = self.get_depth_image(buffer, plot=plot, zmin=zmin, zmax=zmax)
                case _:
                    raise ValueError(f"Invalid capture type '{capture_type}'. Supported types are 'intensity' and 'depth'.")
            if image is None:
                raise ValueError(f"Image of type '{capture_type}' could not be retrieved from buffer.")
        except Exception as e:
            self.logger.log(f"Error getting {capture_type} image from buffer: {e}", e, caller=self)
            raise Exception(f"Error getting {capture_type} image from buffer: {e}") from e
        # Create mask of pixels that are within the specified 3D region
        try:
            mask = self.create_region_mask(buffer, xmin, xmax, ymin, ymax, zmin, zmax)
            # get bounding box of the masked area
            mask_xmin = np.min(np.where(mask.any(axis=0))[0])
            mask_xmax = np.max(np.where(mask.any(axis=0))[0])
            mask_ymin = np.min(np.where(mask.any(axis=1))[0])
            mask_ymax = np.max(np.where(mask.any(axis=1))[0])
            mask_bounding_box = (int(mask_xmin), int(mask_xmax), int(mask_ymin), int(mask_ymax))
        except ValueError as e:
            self.logger.log(f"No points found in mask region.", e, caller=self)
            mask= None
        except Exception as e:
            self.logger.log(f"Error creating region mask: {e}", e, caller=self)
            raise Exception(f"Error creating region mask: {e}") from e
        if mask is None:
            print("No mask created for the specified 3D region.")
            mask_bounding_box = None
            return best_match, binary_image, image_masked, mask_bounding_box
        image_masked = np.zeros_like(image)
        image_masked[mask] = image[mask]
        binary_image = (image_masked > 0).astype(np.uint8)

        if capture_type == "depth":
            if zmax - zmin > 0:
                norm_image = np.zeros_like(image_masked, dtype=np.uint8)
                valid = mask & (image_masked > 0)
                norm_image[valid] = ((image_masked[valid] - zmin) / (zmax - zmin) * 255).clip(0, 255).astype(np.uint8)
                image_masked = norm_image

        if template_set is not None:
            best_match = self.template_match_gripper(image_masked, template_set, plot=plot, match_score_threshold=0.1)
        else:
            best_match = None
            print("No template set provided for gripper matching.")

        if plot:
            plt.imshow(image_masked.reshape((buffer.height, buffer.width)), cmap='gray')
            plt.title('Masked Intensity Image')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
        return best_match, binary_image, image_masked, mask_bounding_box

    def check_gripper_attached(
        self,
        buffer:None|_Buffer,
        template_set:GripperTemplateSet | None,
        d:float,
        max_diameter_gripper:float,
        max_length_gripper:float,
        diameter_margin:float=50,
        method:Literal["CCOEFF_NORMED", "SQDIFF_NORMED"] = "CCOEFF_NORMED", # TODO: add methods ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR','TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        min_points:int= 80,
        plot:bool=False,
        capture_type:Literal["intensity", "depth"] = "intensity",
        gripper_base_z:float=150
    )->tuple[
        None | tuple[None|float, None|str|int],
        None | np.ndarray,
        None | np.ndarray,
        None | tuple[int, int, int, int]
    ]:
        """
        Checks for the presence and position of a gripper in a specified 3D region of an image buffer using template matching.
        Args: 
            buffer: The image buffer containing intensity and depth data.
            template_set: The set of gripper templates to use for matching.
            d: Distance between the camera and gripper centerline (in mm).
            max_diameter_gripper: Maximum diameter of the gripper (in mm).
            max_length_gripper: Maximum length of the gripper (in mm).
            diameter_margin: Margin to add around the gripper dimensions (in mm).
            method: The template matching method to use ("CCOEFF_NORMED" or "SQDIFF_NORMED").
            min_points: Minimum number of 3D points required in the region to perform template matching.
            plot: Whether to plot the masked image.
            capture_type: Either "intensity" or "depth" to specify which image type to use for matching.
            gripper_base_z: The z-coordinate of the base of the gripper (in mm).
        Returns:
            tuple:
                - best_match: The result of the template matching (or None if no template set is provided).
                - binary_image (np.ndarray): Binary mask of the region of interest (shape: (buffer.height, buffer.width)).
                - image_masked (np.ndarray): The masked image (shape: (buffer.height, buffer.width)), type depends on `type` argument.
                - mask_bounding_box (tuple): Bounding box of the masked region as (xmin, xmax, ymin, ymax).
        """
        best_match, binary_image, image_masked, mask_bounding_box = None, None, None, None
        if method not in ["CCOEFF_NORMED", "SQDIFF_NORMED"]:
            raise ValueError(f"Invalid method '{method}'. Supported methods are 'CCOEFF_NORMED' and 'SQDIFF_NORMED'.")
        if buffer is None:
            try:
                buffer = self.capture_buffer()
            except Exception as e:
                self.logger.log(f"Error capturing buffer.", e, caller=self)
                raise Exception(f"Error capturing buffer: {e}") from e
        matchscore, binary_image = None, None
        x, y = d, 0
        radius = max_diameter_gripper / 2
        xmin, xmax = x-radius-diameter_margin, x+radius+diameter_margin
        ymin, ymax = y-radius-diameter_margin, y+radius+diameter_margin
        zmin, zmax = (gripper_base_z - diameter_margin), (gripper_base_z + max_length_gripper + diameter_margin) 
        print(xmin, xmax)
        print(ymin, ymax)
        print(zmin, zmax)
        
        try:
            match capture_type:
                case "intensity":
                    image = self.get_intensity_image(buffer, logarithmic=False, multiplier=1, plot=False)
                case "depth":
                    image = self.get_depth_image(buffer, plot=plot, zmin=zmin, zmax=zmax)
                case _:
                    self.logger.log(f"Unsupported capture type: {capture_type}", caller=self)
                    raise ValueError(f"Unsupported capture type: {capture_type}")
            if image is None:
                raise ValueError(f"Failed to get {capture_type} image from buffer.")
        except Exception as e:
            self.logger.log(f"Error getting {capture_type} image from buffer: {e}", e, caller=self)
            raise Exception(f"Error getting {capture_type} image from buffer: {e}") from e
        # Create mask of pixels that are within the specified 3D region
        try:
            mask = self.create_region_mask(buffer, xmin, xmax, ymin, ymax, zmin, zmax, keep_enclosed_pixels=True)
            # # TODO: remove this  vvvv
            # _ = self.extract_3d_region(xmin,xmax,ymin,ymax,zmin,zmax,buffer,plot=True)
            # # TODO: remove this  ^^^^ 
            if mask is None or mask.size == 0:
                self.logger.log(f"Mask creation returned None for region ({xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax})", caller=self)
                return  matchscore, binary_image, None, None
            # get bounding box of the masked area
            mask_xmin = np.min(np.where(mask.any(axis=0))[0])
            mask_xmax = np.max(np.where(mask.any(axis=0))[0])
            mask_ymin = np.min(np.where(mask.any(axis=1))[0])
            mask_ymax = np.max(np.where(mask.any(axis=1))[0])
            mask_bounding_box = (int(mask_xmin), int(mask_xmax), int(mask_ymin), int(mask_ymax))
        except ValueError as e:
            self.logger.log(f"No points found in mask region.", e, caller=self)
            mask = None
        except Exception as e:
            self.logger.log(f"Error creating region mask: {e}", e, caller=self)
            raise Exception(f"Error creating region mask: {e}") from e
        if mask is None:
            print("No mask created for the specified 3D region.")
            mask_bounding_box = None
            return best_match, binary_image, image_masked, mask_bounding_box
        # apply the mask to the image (both have shape (480, 640))
        image_masked = np.zeros_like(image)
        image_masked[mask] = image[mask]
        binary_image = (image_masked > 0).astype(np.uint8)

        if capture_type == "depth":
            # scale the depth image coloring to 0-255, with zero being zmin and 255 being zmax
            if zmax - zmin > 0:
                norm_image = np.zeros_like(image_masked, dtype=np.uint8)
                valid = mask & (image_masked > 0)
                norm_image[valid] = ((image_masked[valid] - zmin) / (zmax - zmin) * 255).clip(0, 255).astype(np.uint8)
                image_masked = norm_image

        if template_set is not None:
            # best_match = self.template_match_gripper(image_masked, template_set, plot=plot, match_score_threshold=0.8) 
            best_match = self.template_match_gripper(image_masked, template_set, plot=plot, method=method, attached=True, match_score_threshold=0.9) # 0.1 for sqdiff_normed
            # best_match = self.feature_match_gripper(image_masked, template_set, match_score_threshold=0.35, plot=plot)
        # else: #  Type analysis indicates this code is unreachable
        #     best_match = None
        #     print("No template set provided for gripper matching.")
        if plot:
            plt.imshow(image_masked.reshape((buffer.height, buffer.width)), cmap='gray')
            plt.title('Masked Intensity Image')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
        if best_match == (None, None):
            # No matching template was found, check if there is some object in place of the gripper at all
            region_points = self.extract_3d_region(buffer=buffer, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, plot=False)
            if region_points is not None and len(region_points) > min_points:
                # there is likely something on the tool, but its not recognized as a calibrated gripper
                print("Unrecognized object detected on tool.")
                return (None, None, None, None)
            else:
                # there is likely nothing on the tool. Return with best_match = ("empty", None)
                print("Tool is likely empty and ready for gripper attachment.")
                best_match = (None, "empty")
            
        return best_match, binary_image, image_masked, mask_bounding_box

    def create_region_mask(
        self,
        buffer:_Buffer,
        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        zmin:float,
        zmax:float,
        keep_enclosed_pixels:bool=False
    )-> np.ndarray:
        """Returns a boolean mask of shape (height, width) where True indicates the pixel is inside the specified 3D region.
        Args:
            buffer: Buffer containing 3D coordinate data
            xmin, xmax, ymin, ymax, zmin, zmax: Region boundaries in mm.
            keep_enclosed_pixels: If True, keeps pixels that are enclosed by other pixels in the mask even if they are outside the exact boundaries.
        Returns:
            mask: Boolean mask of shape (height, width) indicating the specified 3D region.
        Raises:
            Helios2Error: If nodemap is not available or buffer pixel format is incorrect.
            ValueError: If no points are found in the specified 3D region.
        """
        width = buffer.width
        height = buffer.height
        num_pixels = width * height

        # Get scale and offset for each coordinate from nodemap
        nodemap = self.nodemap if hasattr(self, "nodemap") else None
        if nodemap is not None:
            nodemap['Scan3dCoordinateSelector'].value = 'CoordinateA'
            scale_x = nodemap['Scan3dCoordinateScale'].value
            offset_x = nodemap['Scan3dCoordinateOffset'].value
            nodemap['Scan3dCoordinateSelector'].value = 'CoordinateB'
            scale_y = nodemap['Scan3dCoordinateScale'].value
            offset_y = nodemap['Scan3dCoordinateOffset'].value
            nodemap['Scan3dCoordinateSelector'].value = 'CoordinateC'
            scale_z = nodemap['Scan3dCoordinateScale'].value
            offset_z = nodemap['Scan3dCoordinateOffset'].value
        else:
            e = Helios2Error("Nodemap is not available. Cannot get scale and offset values.")
            self.logger.log("Error creating mask: ", e, caller=self)
            raise e

        import ctypes
        import numpy as np
        pdata_16bit = ctypes.cast(buffer.pdata, ctypes.POINTER(ctypes.c_uint16))

        # Prepare arrays for all coordinates
        x_vals = np.empty(num_pixels, dtype=np.float32)
        y_vals = np.empty(num_pixels, dtype=np.float32)
        z_vals = np.empty(num_pixels, dtype=np.float32)

        for i in range(num_pixels):
            x_vals[i] = int(pdata_16bit[i * 4 + 0]) * scale_x + offset_x
            y_vals[i] = int(pdata_16bit[i * 4 + 1]) * scale_y + offset_y
            z_vals[i] = int(pdata_16bit[i * 4 + 2]) * scale_z + offset_z

        # Create mask
        mask = (
            (x_vals >= xmin) & (x_vals <= xmax) &
            (y_vals >= ymin) & (y_vals <= ymax) &
            (z_vals >= zmin) & (z_vals <= zmax)
        )
        if keep_enclosed_pixels:
            # Reshape mask to 2D for processing
            mask_2d = mask.reshape((height, width))
            # Fill enclosed pixels using binary closing
            from scipy.ndimage import binary_closing
            mask_filled = binary_closing(mask_2d, structure=np.ones((3, 3))).astype(mask_2d.dtype)
            mask = mask_filled.flatten()
        try:
            mask = mask.reshape((height, width))
            if not np.any(mask):
                self.logger.log("Mask is empty, returning empty np.ndarray", caller=self)
                return np.empty((height, width), dtype=bool)
                # raise ValueError(f"No points found for mask in the specified 3D region: x=({xmin},{xmax}), y=({ymin},{ymax}), z=({zmin},{zmax})")
        except ValueError as e:
            raise Exception(f"Error reshaping mask: {e}. Check if the xmin, xmax, ymin, ymax, zmin, zmax are correct.")
        return mask

    def get_calibration_values(
        self,
        calibration_file:str,
        sleeve_rack_id:str|int
    )-> dict | None:
        """Gets calibration values for a specific sleeve ID from the calibration file.
        Args:
            calibration_file (str): Path to the calibration file in JSON format.
            sleeve_rack_id (str): Unique identifier for the sleeve.
        Returns:
            calibration_values (dict): Dictionary containing calibration values for the sleeve.
        """
        if calibration_file is not None:
            try:
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                    # Find the sleeve with matching ID
                    for sleeve in data.get("sleeves", []):
                        if sleeve.get("id") == sleeve_rack_id:
                            return sleeve
                    return None
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.log(f"Error reading config file: {e}", e, caller=self)
                raise Exception(f"Error reading config file: {e}") from e

    def calibrate_sleeve_positions(
        self,
        sleeve_rack_id:str|int,
        slice_points: NDArray[np.float64],  # shape: (N, 2)
        calibration_file:str,
        mode:Literal["multiple", "single"]
    ):
        """Calibrates sleeve positions by selecting calibration points. In case `calibration file` is not found, it will be created.

        Args:
            sleeve_rack_id (str): Unique identifier for the sleeve rack.
            slice_points (NDArray[np.float64]): Array of slice points with shape (N, 2) where N is the number of points.
            calibration_file (str): Path to the JSON file where calibration data will be saved.
            mode (str): Calibration mode, either "multiple" or "single". If "multiple", multiple coordinates can be provided. If "single", a single coordinate is expected.
        Returns:
            None
        Raises:
            ValueError: If the mode is not `multiple` or "single`.
            Exception: If there is an error reading or writing the calibration file.
        """
        if mode not in ["multiple", "single"]:
            e = ValueError("Invalid mode \"{mode}\". Choose either 'multiple' or 'single'.")
            self.logger.log("Invalid mode \"{mode}\". Choose either 'multiple' or 'single'.", e, caller=self)
            raise e
        figsize = (10, 7)
        plt.figure(figsize=figsize)
        slice_points_np = np.array(slice_points)
        plt.scatter(slice_points_np[:, 0], slice_points_np[:, 1], c='b', label="Slice Points", s=10)
        plt.title("Click to select calibration coordinates (close window when done)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')

        # Invert X-axis to match image coordinate system
        plt.gca().invert_xaxis()

        plt.legend()
        plt.show(block=False)
        print("Please click on the plot to select calibration points. Close the plot window when finished.")
        calibration_coordinates = plt.ginput(n=-1, timeout=0)  # n=-1 allows unlimited points until window is closed
        if calibration_coordinates is None or len(calibration_coordinates) == 0:
            print("No calibration coordinates provided.")
            return
        plt.close()
        print("Calibration coordinates:", calibration_coordinates)

        # write the coordinates to a json file with the id
        # First check if the calibration file already exists
        try:
            with open(calibration_file, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {"sleeves": []}
        except FileNotFoundError:
            existing_data = {"sleeves": []}

        # Initialize sleeves array if it doesn't exist
        if "sleeves" not in existing_data:
            existing_data["sleeves"] = []

        # Check if a sleeve with this ID already exists.
        sleeve_index = None
        for i, sleeve in enumerate(existing_data["sleeves"]):
            if sleeve.get("id") == sleeve_rack_id:
                sleeve_index = i
                break
        # If the sleevecart does not exist, create a new one
        if sleeve_index is None:
            sleeve_index = len(existing_data["sleeves"])
            existing_data["sleeves"].append({"id": sleeve_rack_id})

        if "sleeve_coordinates" not in existing_data["sleeves"][sleeve_index]:
            existing_data["sleeves"][sleeve_index]["sleeve_coordinates"] = []
        if "diameter_measurement" not in existing_data["sleeves"][sleeve_index]:
            existing_data["sleeves"][sleeve_index]["diameter_measurement"] = []

        # Create the new sleeve data
        match mode:
            case "multiple":
                new_sleeve_data = {
                    "id": sleeve_rack_id,
                    "sleeve_coordinates": calibration_coordinates,
                    "diameter_measurement": existing_data["sleeves"][sleeve_index]["diameter_measurement"],
                }
            case "single":
                if len(calibration_coordinates) == 1:
                    print(existing_data)
                    new_sleeve_data = {
                        "id": sleeve_rack_id,
                        "sleeve_coordinates": existing_data["sleeves"][sleeve_index]["sleeve_coordinates"],
                        "diameter_measurement": calibration_coordinates,
                    }
                else:
                    e = ValueError("Invalid mode or calibration coordinates. Please provide a single coordinate for single mode.")
                    self.logger.log("Invalid mode or calibration coordinates. Please provide a single coordinate for single mode.", e, caller=self)
                    raise e
            case _:
                e = ValueError("Invalid mode. Choose either 'multiple' or 'single'.")
                self.logger.log("Invalid mode. Choose either 'multiple' or 'single'.", e, caller=self)
                raise e        # Update or add the sleeve data
        if sleeve_index is not None:
            existing_data["sleeves"][sleeve_index] = new_sleeve_data
        else:
            existing_data["sleeves"].append(new_sleeve_data)

        # Write the updated data back to the file
        try:
            with open(calibration_file, 'w') as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            self.logger.log(f"Error writing calibration data to file:", e, caller=self)
            raise Exception(f"Error writing calibration data to file: {e}") from e
        self.logger.log(f"Calibration data for sleeve positions saved to {calibration_file}", caller=self)
        print(f"Calibration data saved to {calibration_file}")

    def calibrate_sleeve_positions2(
        self,
        sleeve_rack_id:str|int,
        coordinates:list[list[float]],
        calibration_file:str,
        mode:Literal["diameter", "presence"]="diameter"
    ):
        """Calibrates sleeve positions by saving coordinates to a JSON file.

        Args:
            sleeve_rack_id (str): Unique identifier for the sleeve.
            coordinates (list): List of coordinates for the sleeve.
            calibration_file (str): Path to the JSON file where calibration data will be saved.
            mode (str): Calibration mode, either "presence" or "diameter". If "presence", multiple coordinates can be provided. If "diameter", a single coordinate is expected.
        Returns:
            None
        """
        try:
            with open(calibration_file, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {"sleeves": []}
        except FileNotFoundError:
            existing_data = {"sleeves": []}

        # Initialize sleeves array if it doesn't exist
        if "sleeves" not in existing_data:
            existing_data["sleeves"] = []

        # Check if a sleeve with this ID already exists.
        sleeve_index = None
        for i, sleeve in enumerate(existing_data["sleeves"]):
            if sleeve.get("id") == sleeve_rack_id:
                sleeve_index = i
                break
        # If the sleeve index does not exist, create a new one
        if sleeve_index is None:
            sleeve_index = len(existing_data["sleeves"])
            existing_data["sleeves"].append({"id": sleeve_rack_id})

        if "sleeve_coordinates" not in existing_data["sleeves"][sleeve_index]:
            existing_data["sleeves"][sleeve_index]["sleeve_coordinates"] = []
        if "diameter_measurement" not in existing_data["sleeves"][sleeve_index]:
            existing_data["sleeves"][sleeve_index]["diameter_measurement"] = []


        match mode:
            case "presence":
                new_sleeve_data = {
                    "id": sleeve_rack_id,
                    "sleeve_coordinates": coordinates,
                    "diameter_measurement": existing_data["sleeves"][sleeve_index]["diameter_measurement"],
                }
            case "diameter":
                if len(coordinates) != 1:
                    e = Exception("Please provide a single coordinate for \"diameter\" calibration.")
                    self.logger.log("Please provide a single coordinate for \"diameter\" calibration.", e, caller=self)
                    raise e
                print(existing_data)
                new_sleeve_data = {
                    "id": sleeve_rack_id,
                    "sleeve_coordinates": existing_data["sleeves"][sleeve_index]["sleeve_coordinates"],
                    "diameter_measurement": coordinates,
                }
            case _:
                e = Exception("Please provide a mode for calibration (\"presence\" or \"diameter\").")
                self.logger.log("Please provide a mode for calibration (\"presence\" or \"diameter\")", e, caller=self)
                raise e
        # Update or add the sleeve data
        if sleeve_index is not None:
            existing_data["sleeves"][sleeve_index] = new_sleeve_data
        else:
            existing_data["sleeves"].append(new_sleeve_data)

        # Write the updated data back to the file
        with open(calibration_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

        print(f"Calibration data saved to {calibration_file}")

    def get_sleeve_face_angle(
        self,
        x:float,
        y:float,
        zmargin:float,
        xymargin:float,
        buffer:None|_Buffer=None,
        points:None|np.ndarray=None,
        plot:bool=False,
        return_image:bool=False
    )-> tuple[float, None|np.ndarray]:
        """Calculates the sleeve face angle based on points in a specified region.
        Args:
            x (float): X-coordinate of the center of the region.
            y (float): Y-coordinate of the center of the region.
            zmargin (float): Margin in Z direction to consider for slicing points.
            xymargin (float): Margin in XY direction to consider for the region.
            buffer (Buffer, optional): The image buffer containing intensity and depth data. If None, it will be captured.
            points (np.ndarray, optional): Pre-extracted points to use instead of capturing a new buffer. If None, it will be extracted from the buffer.
            plot (bool, optional): Whether to display debugging plots. Defaults to False.
            return_image (bool, optional): Whether to return the plot as an image. Defaults to False.
        Returns:
            tuple: A tuple containing the sleeve face angle in degrees and an optional image of the plot.
        Raises:
            ValueError: If neither buffer nor points are provided.
            PointsProcessingError: If no points are extracted from the specified region.
        """
        self.logger.log(f"Calculating sleeve face angle at ({x}, {y}) with zmargin={zmargin} and xymargin={xymargin}", caller=self)
        image = None
        angle = None
        if buffer is None and points is None:
            try:
                buffer = self.capture_buffer()
            except Exception as e:
                self.logger.log(f"Error getting buffer: {e}", e, caller=self)
                raise Helios2Error(f"Error getting buffer: {e}") from e
        elif buffer is not None and points is None:
            try:
                points = self.buffer_to_numpy(buffer)
            except Exception as e:
                self.logger.log(f"Error converting buffer to numpy array: {e}", e, caller=self)
                raise e
        try:
            closest_surface, points = self.get_closest_surface_from_region(
                (x, y),
                buffer=buffer,
                sensitivity=50,
                margin=xymargin,
                plot=plot)
            if closest_surface is None:
                e = PointsProcessingError("No closest surface found in the specified region.")
                self.logger.log("No closest surface found in the specified region.", e, caller=self)
                raise e
            points = self.slice(
                points=points,
                z=closest_surface,
                margin=zmargin,
                plot=plot
            )
        except Exception as e:
            self.logger.log(f"Error getting closest surface and slicing points:", e, caller=self)
            raise e
        if points is None or len(points) == 0:
            e = PointsProcessingError("No points extracted from the specified region.")
            self.logger.log("No points extracted from the specified region.", e, caller=self)
            raise e
        xz_projection = points[:, [0, 2]]
        bin_size = 5.0  # mm
        x_bins = np.arange(xz_projection[:, 0].min(), xz_projection[:, 0].max() + bin_size, bin_size)
        indices = np.digitize(xz_projection[:, 0], x_bins)
        min_z_points = []
        for i in np.unique(indices):
            bin_points = xz_projection[indices == i]
            if len(bin_points) > 0:
                x_val = bin_points[:, 0].mean()
                z_val = np.sort(bin_points[:, 1])[:10].mean()
                min_z_points.append([x_val, z_val])
        min_z_points = np.array(min_z_points)

        # Fit a line to the (X, min Z) points
        from sklearn.linear_model import LinearRegression
        try:
            _x = min_z_points[:, 0].reshape(-1, 1)
            _z = min_z_points[:, 1]
            model = LinearRegression()
            model.fit(_x, _z)
            slope = model.coef_[0]
            angle = np.arctan(slope) * (180 / np.pi)
        except Exception as e:
            self.logger.log("Error fitting line to points", e)
            raise e
        if plot or return_image:
            try:
                plt.figure(figsize=(8, 6))
                plt.scatter(xz_projection[:, 0], xz_projection[:, 1], label='All Points', s=10, alpha=0.3)
                plt.scatter(min_z_points[:, 0], min_z_points[:, 1], color='orange', label='Lower Envelope', s=20)
                plt.plot(_x, model.predict(_x), color='red', label='Fitted Tangent Line')
                plt.title(f'Sleeve Face Angle (lower envelope): {angle:.2f} degrees')
                plt.xlabel('X')
                plt.ylabel('Z')
                plt.legend()
                plt.gca().set_aspect('equal', adjustable='box')
                if plot:
                    plt.show()
            except:
                e = Exception("Error plotting the results. Ensure matplotlib is properly configured.")
                self.logger.log("Error plotting the results. Ensure matplotlib is properly configured.", e, caller=self)
                raise e
            if return_image:
                try:
                    buf = plt.gcf().canvas.print_to_buffer() # type: ignore
                    image = np.frombuffer(buf[0], dtype=np.uint8)
                    image = image.reshape(buf[1][::-1] + (4,))
                    image = image[:, :, :3]
                except Exception as e:
                    self.logger.log("Error creating image", e)
                    raise e
        try:
            plt.close("all")
        except Exception as e:
            print(f"Error closing plots: {e}")
        self.logger.log(f"Sleeve face angle calculated (at {closest_surface} mm): {angle:.2f} degrees", caller=self)
        return angle, image

    def estimate_sleeve_length(
        self,
        sleeve_face_angle:float,
        alignment_rack_elevation:float,
        outer_diameter_sleeve:float,
        inner_diameter_sleeve:float
    )-> float:
        """Estimates sleeve length based on diameter and angle in alignment rack

        Args:
            sleeve_face_angle (float): Angle of the sleeve face in degrees.
            alignment_rack_elevation (float): Elevation of the alignment rack in mm.
            outer_diameter_sleeve (float): Outer diameter of the sleeve in mm.
            inner_diameter_sleeve (float): Inner diameter of the sleeve in mm.
        Returns:
            float: Estimated sleeve length in mm.
        """
        base_distance = 250 # TODO:, placeholder value, distance sleevepin to resting point alignment rack in unelevated state 
        d = alignment_rack_elevation - (base_distance - (inner_diameter_sleeve + (outer_diameter_sleeve-inner_diameter_sleeve)/2))
        d = 100 # TODO: placeholder value, remove
        sleeve_length = d / np.sin(np.radians(sleeve_face_angle))
        return sleeve_length

    def measure_attached_sleeve_diameter(
        self,
        buffer:None|_Buffer=None,
        d:float=-400,
        max_sleeve_length:float=1500,
        max_sleeve_diameter:float=300,
        plot:bool=False,
        return_image:bool=False
    )-> tuple[float, None|np.ndarray]:
        """Measures the diameter of the sleeve in front of the tool.
        Args:
            buffer: Buffer containing 3D coordinate data.
            d (float): Distance in mm from the camera axis to the center of the sleeve.
            max_sleeve_length (float): Maximum length of the sleeve in mm.
            max_sleeve_diameter (float): Maximum diameter of the sleeve in mm.
        Returns:
            float: Estimated sleeve diameter in mm.
        Raises:
            PointsProcessingError: If no points are extracted from the specified region.
            Exception: If there is an error fitting the arc to the projected points.
        """

        # extract region in front of tool
        # project to xy plane
        # fit arc
        # return diameter

        self.logger.log("measure_attached_sleeve_diameter was called", caller=self)
        if buffer is None:
            try:
                buffer = self.capture_buffer()
            except Exception as e: 
                self.logger.log("Error capturing buffer",e)
                raise e

        x, y = d, 0
        radius = max_sleeve_diameter / 2
        someconstant = 150 # constant to adjust the z-range to the base of the gripper # TODO: add parameter
        somemargin = 50 # margin to add to the gripper length and diameter # TODO: add parameter
        xmin, xmax = x-radius-somemargin, x+radius+somemargin
        ymin, ymax = y-radius-somemargin, y+radius+somemargin
        zmin, zmax = (someconstant - somemargin), (someconstant + max_sleeve_length + somemargin)

        # TODO: remove start slice test
        z_avg = (zmin + zmax) / 2
        zmin = z_avg - 20# Adjust zmin to be below the average
        zmax = z_avg + 20  # Adjust zmax to be above the average

        # TODO: remove end slice test
        # self, buffer, xmin, xmax, ymin, ymax, zmin, zmax, plot=False, save_path=None):

        points = self.extract_3d_region(
            buffer=buffer,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            plot=plot,
        )
        if points is None or len(points) == 0:
            e = PointsProcessingError("No points extracted from the specified region.")
            self.logger.log("No points extracted from the specified region.", e, caller=self)
            raise e
        # project points to the xy plane
        projected_points = points[:, [0, 1]]
        # the points will describe a partial circle.  Fit a circular arc to the projected points
        try:
            from scipy.optimize import curve_fit

            def circle_equation(x, a, b, r):
                return np.sqrt(r**2 - (x - a)**2) + b

            # Initial guess for the parameters: center (a, b) and radius r
            x_mean = np.mean(projected_points[:, 0])
            y_mean = np.mean(projected_points[:, 1])
            r_guess = np.max(np.linalg.norm(projected_points - np.array([x_mean, y_mean]), axis=1))
            initial_guess = [x_mean, y_mean, r_guess]

            # Fit the circle equation to the projected points
            params, _ = curve_fit(circle_equation, projected_points[:, 0], projected_points[:, 1], p0=initial_guess)
            a, b, r = params

            # Calculate the diameter
            diameter = 2 * r
            self.logger.log(f"Estimated sleeve diameter: {diameter:.2f} mm", caller=self)
            image = None
            if plot or return_image:
                try:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(projected_points[:, 0], projected_points[:, 1], label='Projected Points', s=10, alpha=0.5)
                    theta = np.linspace(0, 2 * np.pi, 100)
                    x_circle = a + r * np.cos(theta)
                    y_circle = b + r * np.sin(theta)
                    plt.plot(x_circle, y_circle, color='red', label='Fitted Circle')
                    plt.title(f'Sleeve Diameter Estimation: {diameter:.2f} mm')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.legend()
                    if plot:
                        plt.show()
                    if return_image:
                        buf = plt.gcf().canvas.print_to_buffer() #type: ignore
                        image = np.frombuffer(buf[0], dtype=np.uint8)
                        image = image.reshape(buf[1][::-1] + (4,))
                        image = image[:, :, :3]
                    plt.close("all")
                    plt.clf()
                except Exception as e: self.logger.log("Error plotting the results", e, caller=self)
            return diameter, image
        except Exception as e:
            self.logger.log("Error fitting arc to projected points", e, caller=self)
            raise e 

    def measure_attached_sleeve_length(
        self,
        buffer:None|_Buffer=None,
        d:float=-400,
        max_sleeve_length:float=1500,
        max_sleeve_diameter:float=300,
        plot:bool=False,
        return_image:float=False
    )-> tuple[float, None|np.ndarray]:
        image, sleeve_length = None, None
        def placeholder():
            self.logger.log("measure_attached_sleeve_length was called", caller=self)
            image = None
            length = 100 # TODO: placeholder value
            return length, image
        try:
            if buffer is None:
                try:
                    buffer = self.capture_buffer()
                except Exception as e: 
                    self.logger.log("Error capturing buffer", e)
                    raise e
            x, y = d, 0
            radius = max_sleeve_diameter / 2
            someconstant = 150 # constant to adjust the z-range to the base of the gripper # TODO: add parameter
            somemargin = 50 # margin to add to the gripper length and diameter # TODO: add parameter
            xmin, xmax = x-radius-somemargin, x+radius+somemargin
            ymin, ymax = y-radius-somemargin, y+radius+somemargin
            zmin, zmax = (someconstant - somemargin), (someconstant + max_sleeve_length + somemargin)

            points = self.extract_3d_region(
                buffer=buffer,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                zmin=zmin,
                zmax=zmax,
                plot=plot,
            )
            z_values = points[:, 2]
            if z_values is None or len(z_values) == 0:
                e = PointsProcessingError("No points extracted from the specified region.")
                self.logger.log("No points extracted from the specified region.", e, caller=self)
                raise e
            if plot or return_image:
                try:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=10, alpha=0.5)
                    plt.colorbar(label='Z (mm)')
                    plt.title('3D Points in Specified Region')
                    plt.xlabel('X (mm)')
                    plt.ylabel('Y (mm)')
                    plt.gca().set_aspect('equal', adjustable='box')
                    if plot:
                        plt.show()
                    if return_image:
                        buf = plt.gcf().canvas.print_to_buffer() #type: ignore
                        image = np.frombuffer(buf[0], dtype=np.uint8)
                        image = image.reshape(buf[1][::-1] + (4,))
                        image = image[:, :, :3]
                    plt.close("all")
                    plt.clf()
                except Exception as e:
                    self.logger.log("Error plotting the results", e, caller=self)
            furthest_point = np.percentile(z_values, 95)
            sleeve_length = float(furthest_point - someconstant)
            return sleeve_length, image
        except Exception as e:
            self.logger.log("placeholder", e, caller=self)
            return placeholder()

class PointProcessor:
    def __init__(self):
        self.logger = Logger("Helios2.log", name=__name__, verbose=False)

    def ply_to_numpy(self, ply_dir:str)-> np.ndarray:
        """Reads a PLY file and converts it to a numpy array.
        Args:
            ply_dir (str): Path to the PLY file.
        Returns:
            np.ndarray: Numpy array of shape (N, 3) where N is the number of points.
        """
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_dir)
        points = np.asarray(pcd.points)
        if points.shape[1] != 3:
            e = ValueError("PLY file must contain 3D points with shape (N, 3).")
            raise e
        return points 

    def euler_to_R_mat(
        self,
        r:float,
        p:float,
        y:float,
    )-> np.ndarray:
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])
        Ry = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])
        Rz = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx  # Matrix multiplication in the order ZYX
        return R

    def create_T_vec(
        self,
        x:float,
        y:float,
        z:float,
    )-> np.ndarray:
        """Creates a translation matrix for a given translation vector.
        Args:
            x (float): Translation in the X direction.
            y (float): Translation in the Y direction.
            z (float): Translation in the Z direction.
        Returns:
            np.ndarray: Translation matrix of shape (3,).
        """
        T = np.array([x, y, z])
        return T

    def create_H_mat(
        self,
        R:np.ndarray,
        T:np.ndarray
    )-> np.ndarray:
        """Creates a homogeneous transformation matrix from a rotation matrix and a translation vector.
        Args:
            R (np.ndarray): Rotation matrix of shape (3, 3).
            T (np.ndarray): Translation vector of shape (3,).
        Returns:
            np.ndarray: Homogeneous transformation matrix of shape (4, 4).
        """
        if R.shape != (3, 3) or T.shape != (3,):
            raise ValueError("R must be of shape (3, 3) and T must be of shape (3,).")
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T
        return H

    def rotate(
        self,
        points:np.ndarray,
        R_mat:np.ndarray
    )-> np.ndarray:
        """Rotates points using matrix of shape (3, 3):
        Args:
            points (np.ndarray): Points to rotate, shape (N, 3) where N is the number of points.
            R_mat (np.ndarray): Rotation matrix of shape (3, 3):
                [[r11, r12, r13],
                 [r21, r22, r23],
                 [r31, r32, r33]]
        Returns:
            np.ndarray: Rotated points of shape (N, 3).
        """
        if R_mat.shape != (3, 3):
            e = ValueError("R_mat must be of shape (3, 3).")
            raise e
        if points.shape[1] != 3:
            e = ValueError("Points must be of shape (N, 3) where N is the number of points.")
            raise e
        points_rotated = points @ R_mat.T
        return points_rotated

    def translate(
        self,
        points:np.ndarray,
        T_mat:np.ndarray
    )-> np.ndarray:
        """Translates points using translation vector of shape (3,):
        Args:
            points (np.ndarray): Points to translate, shape (N, 3) where N is the number of points.
            T_mat (np.ndarray): Translation vector of shape (3,):
                [t1, t2, t3]
        Returns:
            np.ndarray: Translated points of shape (N, 3).
        """
        if T_mat.shape != (3,):
            e = ValueError("T_mat must be of shape (3,).")
            raise e
        if points.shape[1] != 3:
            e = ValueError("Points must be of shape (N, 3) where N is the number of points.")
            raise e
        points_translated = points + T_mat
        return points_translated

    def h_transform(
        self,
        points:np.ndarray,
        H_mat:np.ndarray
    )-> np.ndarray:
        """Applies a homogeneous transformation to points using matrix of shape (4, 4):
        Args:
            points (np.ndarray): Points to transform, shape (N, 3) where N is the number of points.
            H_mat (np.ndarray): Homogeneous transformation matrix of shape (4, 4):
                [[r11, r12, r13, t1],
                 [r21, r22, r23, t2],
                 [r31, r32, r33, t3],
                 [0, 0, 0, 1]]
        Returns:
            np.ndarray: Transformed points of shape (N, 3).
        """
        if H_mat.shape != (4, 4):
            e = ValueError("H_mat must be of shape (4, 4).")
            raise e
        if points.shape[1] != 3:
            e = ValueError("Points must be of shape (N, 3) where N is the number of points.")
            raise e
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = points_homogeneous @ H_mat.T
        return transformed_points[:, :3]

    def visualize_points(self, points:np.ndarray|list[np.ndarray])-> None:
        """Visualizes the points using open3d. In case of a list, it colors each point cloud differently.
        """
        if isinstance(points, list):
            try:
                pcd = o3d.geometry.PointCloud()
                for point_set in points:
                    pcd_i = o3d.geometry.PointCloud()
                    pcd_i.points = o3d.utility.Vector3dVector(point_set)
                    color = np.random.rand(3)
                    pcd_i.paint_uniform_color(color)
                    pcd += pcd_i
                # Add coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
                o3d.visualization.draw_geometries( # pyright: ignore[reportAttributeAccessIssue]
                    [pcd, coord_frame], #type: ignore
                    window_name="Extracted 3D Region",
                    width=800, height=600)
            except Exception as e:
                print(f"Error occurred while plotting points: {e}")
                self.logger.log(f"There was an error plotting the results.", e, caller=self)
        else:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color([0, 0.651, 0.929])
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
                o3d.visualization.draw_geometries( # pyright: ignore[reportAttributeAccessIssue]
                    [pcd, coord_frame], #type: ignore
                    window_name="Extracted 3D Region",
                    width=800, height=600)
            except Exception as e:
                print(f"Error occurred while plotting points: {e}")
                self.logger.log(f"There was an error plotting the results.", e, caller=self)

class GripperTemplateMaker:
    """Class to create and save gripper templates using a Helios2 camera.
    Attributes:
        save_dir (str): Directory where templates will be saved.
        type (str): Type of template to create, either "intensity" or "depth".
        camera (Helios2): Instance of the Helios2 camera.
    """
    def __init__(
        self,
        save_dir:str,
        capture_type:Literal["intensity", "depth"]="intensity",
        camera:None|Helios2=None
    ):
        self.logger = Logger("Helios2.log", __name__)
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno

        if camera is not None:
            self.camera = camera
        else:
            self.camera = Helios2()
        self.save_dir = save_dir
        if capture_type not in ["intensity", "depth"]:
            raise ValueError("Type must be either 'intensity' or 'depth'.")
        else:
            self.capture_type:Literal["intensity", "depth"] = capture_type
        self.logger.log(f"GripperTemplateMaker object {id(self)} with save_dir \'{self.save_dir}\' created in {caller_file} at line {caller_line}", caller=self)

    def __del__(self):
        self.logger.log(f"GripperTemplateMaker instance {id(self)} deleted.", caller=self)

    def capture_template_gripperrack(
        self,
        xy:tuple[float, float],
        z:float,
        plot:bool=False
    )-> None|np.ndarray:
        """Creates a template image of a specified XY region at depth Z (in cartesian coordinates w.r.t. the camera in mm).

        Args:
            xy (tuple): The (x, y) coordinates of the gripper center position (in mm).
            z (float): The z depth perpendicular to the camera plane (in mm).
            plot (bool): Enable/disable plotting of intermediate results.

        Returns:
            template (np.ndarray): Template image with depth or intensity values depending on `self.capture_type` value.
        """
        try:
            buffer = self.camera.capture_buffer()
            _, _, template, mask_bounding_box = self.camera.check_gripper(
                buffer=buffer,
                template_set=None,
                xy=xy,
                z=z,
                plot=plot,
                capture_type=self.capture_type
            )
            if mask_bounding_box is None or template is None:
                self.logger.log("No gripper detected in the captured image.", caller=self)
                return None
            xmin, xmax, ymin, ymax = mask_bounding_box
            # keep only the bounding box area
            template = template[ymin:ymax, xmin:xmax]
            self.logger.log("Template captured succesfully", caller=self)
            return template
        except Exception as e:
            self.logger.log("Error capturing template", e, caller=self)
            raise e

    def capture_template_attached(
        self,
        d:float,
        max_diameter_gripper:float,
        max_length_gripper:float,
        plot:bool=False
    )-> None|np.ndarray:
        """Creates a template image of a specified XY region at depth Z (in cartesian coordinates w.r.t. the camera in mm).

        Args:
            xy (tuple): The (x, y) coordinates of the gripper center position (in mm).
            z (float): The z depth perpendicular to the camera plane (in mm).
            plot (bool): Enable/disable plotting of intermediate results.

        Returns:
            template (np.ndarray): Template image with depth or intensity values depending on `self.capture_type` value.
        """
        try:
            buffer = self.camera.capture_buffer()
            _, _, template, mask_bounding_box = self.camera.check_gripper_attached(
                buffer,
                template_set=None,
                d=d,
                max_diameter_gripper=max_diameter_gripper,
                max_length_gripper=max_length_gripper,
                plot=plot
            )
            if mask_bounding_box is None or template is None:
                self.logger.log("No gripper detected in the captured image.", caller=self)
                return None
            xmin, xmax, ymin, ymax = mask_bounding_box
            # keep only the bounding box area
            template = template[ymin:ymax, xmin:xmax]
            self.logger.log("Template captured succesfully", caller=self)
            return template
        except Exception as e:
            self.logger.log("Error capturing template", e, caller=self)
            raise e

    def save_template_attached(
        self,
        template:np.ndarray,
        gripper_id:str | int,
        check_if_exists:bool=True
    ):
        """Saves a template image to `grippers.json` in `self.save_dir`.

        Args:
            template (np.ndarray): The template image to save.
            xy (tuple): The (x, y) coordinates of the gripper center position w.r.t. the camera (in mm).
            z (float): The z coordinate of the gripper front w.r.t. the camera (in mm).
            gripper_id (str): Identifier for the gripper template.
            check_if_exists (bool, optional): If True, checks for existing ID and prompts the user. Default is True.

        Returns:
            None
        """
        if self.capture_type not in ["intensity", "depth"]:
            e = ValueError("self.capture_type has to be set to either depth or intensity")
            self.logger.log("Error in save_template_gripperrack", e, caller=self)
            raise e
        # if grippers.json does not exsit in the save_dir, create it
        templates_file = os.path.join(self.save_dir, "grippers.json")
        if not os.path.exists(templates_file):
            with open(templates_file, 'w') as f:
                json.dump({}, f)
        # Load existing templates, handle empty file
        with open(templates_file, 'r') as f:
            content = f.read().strip()
            if not content:
                templates = {}
            else:
                templates = json.loads(content)
        if check_if_exists:
            while gripper_id in templates and "attached_template" in templates[gripper_id]:
                response = input(f"Template for attached gripper with ID '{gripper_id}' already exists. Overwrite? [Y/N]:\n> ")
                match response.lower():
                    case 'y':
                        print(f"Overwriting template with ID '{gripper_id}'.")
                        break
                    case 'n':
                        print("Template not saved.")
                        gripper_id = input("Please provide a new identifier for the gripper.\n> ID: ")
                        # Continue the while loop to check if the new ID also exists
                    case _:
                        print("Please respond 'Y' or 'N'.")
        # Save the updated templates to the file
        if gripper_id not in templates:
            templates[gripper_id] = {}
        templates[gripper_id]["attached_template"] = f"{gripper_id}_attached.jpg"
        try:
            with open(templates_file, 'w') as f:
                json.dump(templates, f, indent=4)
        except Exception as e:
            self.logger.log("Error writing to grippers.json", e, caller=self)
            raise e
        # save the template as {gripper_id}_attached.jpg in the save_dir
        try:
            template_path = os.path.join(self.save_dir, f"{gripper_id}_attached.jpg")
            match self.capture_type:
                case "intensity":
                    cv2.imwrite(template_path, template)
                case "depth":
                    # Convert depth image to 8-bit grayscale for saving
                    template_8bit = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)# type: ignore
                    cv2.imwrite(template_path, template_8bit)
            self.logger.log("Template image saved succesfully.", caller=self)
        except Exception as e:
            self.logger.log("Error saving template image", e, caller=self)
            raise e

    def save_template_gripperrack(
        self,
        template: np.ndarray,
        xy: tuple[float, float],
        z: float,
        gripper_id: str | int,
        check_if_exists:bool=True
    ): # TODO: currently overwrites existing attached gripper template
        """Saves a template image to `grippers.json` in `self.save_dir`.

        Args:
            template (np.ndarray): The template image to save.
            xy (tuple): The (x, y) coordinates of the gripper center position w.r.t. the camera (in mm).
            z (float): The z coordinate of the gripper front w.r.t. the camera (in mm).
            gripper_id (str): Identifier for the gripper template.
            check_if_exists (bool, optional): If True, checks for existing ID and prompts the user. Default is True.

        Returns:
            None
        """
        if self.capture_type not in ["intensity", "depth"]:
            e = ValueError("self.capture_type has to be set to either depth or intensity")
            self.logger.log("Error in save_template_gripperrack", e, caller=self)
            raise e
        # if grippers.json does not exsit in the save_dir, create it
        templates_file = os.path.join(self.save_dir, "grippers.json")
        if not os.path.exists(templates_file):
            with open(templates_file, 'w') as f:
                json.dump({}, f)
        # Load existing templates, handle empty file
        with open(templates_file, 'r') as f:
            content = f.read().strip()
            if not content:
                templates = {}
            else:
                templates = json.loads(content)
        if check_if_exists:
            # Check if the ID already exists
            while gripper_id in templates and "gripperrack_template" in templates[gripper_id]: 
                response = input(f"Template with ID '{gripper_id}' already exists. Overwrite? [Y/N]:\n> ")
                match response.lower():
                    case 'y':
                        print(f"Overwriting template with ID '{gripper_id}'.")
                        break
                    case 'n':
                        print("Template not saved.")
                        gripper_id = input("Please provide a new identifier for the gripper.\n> ID: ")
                        # Continue the while loop to check if the new ID also exists
                    case _:
                        print("Please respond 'Y' or 'N'.")
        # Save the updated templates to the file
        current_json = templates.get(gripper_id, {})
        new_json = {
            "id": gripper_id,
            "type": self.capture_type,
            "center_position": {
                "xy": xy,
                "z": z,
            },
            "gripperrack_template": f"{gripper_id}.jpg"
        }
        merged_json = {**current_json, **new_json}
        templates[gripper_id] = merged_json
        try:
            with open(templates_file, 'w') as f:
                # merge existing json with new one, giving priority to the new one
                json.dump(templates, f, indent=4)
        except Exception as e:
            self.logger.log("Error writing to grippers.json", e, caller=self)
            raise e
        # save the template as {gripper_id}.jpg in the save_dir
        try:
            template_path = os.path.join(self.save_dir, f"{gripper_id}.jpg")
            match self.capture_type:
                case "intensity":
                    cv2.imwrite(template_path, template)
                case "depth":
                    # Convert depth image to 8-bit grayscale for saving
                    template_8bit = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)# type: ignore
                    cv2.imwrite(template_path, template_8bit)
                case _:
                    e = ValueError("self.capture_type has to be set to either depth or intensity")
                    self.logger.log("Error in save_template_gripperrack", e, caller=self)
                    raise e
            self.logger.log("Template image saved succesfully.", caller=self)
        except Exception as e:
            self.logger.log("Error saving template image", e, caller=self)
            raise e

    def create_templates_gripperrack(self):
        """Interactive method to create and save gripper templates using the connected camera.
            1. Prompt user to position the camera and enter X, Y coordinates.
            2. Ask whether to use the closest surface for Z or enter it manually.
            3. Capture and display the template image.
            4. Ask the user whether to save or discard the template.
            5. If saved, prompt for an identifier and store the template and metadata.

        Returns:
            None
        """
        stop = False
        while not stop:
            input("Move camera to detection position and press Enter.")
            try:
                x = float(input("Relative X coordinate of center gripper (in mm)\n> "))
                y = float(input("Relative Y coordinate of center gripper (in mm)\n> "))
            except ValueError as e: 
                self.logger.log("Invalid input for X or Y coordinate", e, caller=self)
                print("Invalid input. Please enter valid numbers for X and Y coordinates.")
                continue
            xy = (x, y)
            while True:
                response = input("Treat closest surface as Z? [Y/N]\n> ")
                match response.lower():
                    case 'y':
                        z = self.camera.get_closest_surface(filter_zeros=True)
                        break
                    case 'n':
                        try:
                            z = float(input("Relative Z coordinate of gripper front (in mm)\n> "))
                        except ValueError: 
                            print("Invalid input. Please enter a valid number for Z.")
                            continue
                        break
                    case _:
                        print("Please respond 'Y' or 'N'.")
            if z is None:
                print("No Z value provided. Please try again.")
                continue
            try:
                template = self.capture_template_gripperrack(xy, z)
            except Exception as e:
                self.logger.log("Failed to capture template", e, caller=self)
                print(f"Failed to capture template: {str(e)}")
                continue

            # plot the captured template
            if template is not None:
                plt.imshow(template, cmap='gray')
                plt.title('Captured Template')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.colorbar(label='Depth (normalized)' if self.capture_type == "depth" else 'Intensity')
                plt.show()

                restart = False
                gripper_id = ""
                while True:
                    response = input("Save template? [Y/N]\n> ")
                    match response.lower():
                        case 'n':
                            restart = True
                            print("Template discarded.\n")
                            break
                        case 'y':
                            gripper_id = input("Please provide an identifier for the gripper.\n> ID: ")
                            if gripper_id is None or gripper_id.strip() == "":
                                print("Identifier cannot be empty. Please provide a valid identifier.")
                                continue
                            else:
                                break
                        case _:
                            print("Please respond 'Y' or 'N'.")
                if restart:
                    continue

                try:
                    self.save_template_gripperrack(template, xy, z, gripper_id) 
                except Exception as e:
                    self.logger.log("Error saving template", e)
                    print(f"Error saving template: {str(e)}")
                    continue

    def create_templates_attached(
        self,
        d:float=150,
        max_diameter_gripper:float|None=150,
        max_length_gripper:float|None=150,
        plot:bool=True
    ): 
        """Interactive method to create and save templates of attached grippers using the camera.
            1. Prompt user to position the camera and enter X, Y coordinates.
            2. Ask whether to use the closest surface for Z or enter it manually.
            3. Capture and display the template image.
            4. Ask the user whether to save or discard the template.
            5. If saved, prompt for an identifier and store the template and metadata.

        Args:
            d (int): Distance from the centerline of the gripper to the camera axis in mm. Default is 150.  # TODO: replace with measurement
            max_diameter_gripper (int): Maximum diameter of the gripper in mm. Default is 150 mm.           # TODO: replacd with reasonable value
            max_length_gripper (int): Maximum length of the gripper in mm. Default is 150 mm.               # TODO: replace with reasonable value

        Returns:
            None
        """
        stop = False
        stop = not yn_prompt("Start gripper template creation? [Y/N]\n> ")
        while not stop:
            input("Attach gripper and press Enter.")
            # x = d
            # y = 0 # TODO: add y offset
            try:
                if max_diameter_gripper is None:
                    max_diameter_gripper = float(input("Max diameter gripper: "))
                if max_length_gripper is None:
                    max_length_gripper = float(input("Max length gripper: "))
            except:
                self.logger.log("Invalid input for max diameter or length", caller=self)
                print("Please provide numerical input.")
                continue
            try:
                template = self.capture_template_attached(
                    d=d,
                    max_diameter_gripper=max_diameter_gripper,
                    max_length_gripper=max_length_gripper,
                    plot=plot)
            except Exception as e:
                self.logger.log("Failed to capture template", e, caller=self)
                print(f"Failed to capture template: {str(e)}")
                continue

            # plot the captured template
            if template is not None:
                plt.imshow(template, cmap='gray')
                plt.title('Captured Template')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.colorbar(label='Depth (normalized)' if self.capture_type == "depth" else 'Intensity')
                plt.show()

                restart = False
                gripper_id = ""

                response = yn_prompt("Save template? [Y/N] ") 
                match response:
                    case False:
                        print("Template discarded.\n")
                        continue
                    case True:
                        gripper_id = str_prompt("Please provide an identifier for the gripper.\n> ID: ") 
                try:
                    self.save_template_attached(template, gripper_id) 
                except Exception as e:
                    self.logger.log("Error saving template", e)
                    print(f"Error saving template: {str(e)}")
                    continue
                stop = not yn_prompt("Create another template? [Y/N]\n> ")
                
class Sleeve_collection:
    """Collection of sleeves loaded from a `.json` file.

    Attributes:
        sleeve_dict (dict): Dictionary containing sleeve data indexed by their repeat values.
    """
    def __init__(self, json_path:str):
        self.logger = Logger("Helios2.py", __name__)
        self.sleeve_dict = {}
        self.load_sleeve_data(json_path)

    def __del__(self):
        self.logger.log(f"Sleeve_collection instance {id(self)} deleted.", caller=self)

    def load_sleeve_data(self, json_path:str):
        """Load sleeve data from a `.json` file and populate the sleeve dictionary.

        Args:
            json_path (str): Path to the `.json` file containing sleeve data.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find file: {json_path}")

        try:
            with open(json_path, 'r') as file:
                sleeve_data = json.load(file)

            for sleeve in sleeve_data:
                repeat = sleeve.get("Repeat")
                if repeat is not None:
                    self.sleeve_dict[repeat] = {
                        "BCD": sleeve.get("BCD"),
                        "OD": sleeve.get("OD"),
                        "Wall Thickness": sleeve.get("Wall Thickness"),
                        "Length": sleeve.get("Length")
                    }
            print(f"Loaded {len(self.sleeve_dict)} sleeves from {json_path}")
        except json.JSONDecodeError as e:
            self.logger.log(f"Failed to parse JSON file {json_path}", e, caller=self)
            raise ValueError(f"Failed to parse JSON file {json_path}: {str(e)}") from e

    def match_diameter(self, diameter:float)-> dict | None:
        """Find the closest matching diameter sleeve in the sleeve dictionary.

        Args:
            diameter (float): The diameter to find the closest match for.

        Returns:
            dict: The sleeve data with the closest matching diameter.
        """
        if not self.sleeve_dict:
            print("No sleeves available in the dictionary.")
            return None

        closest_match = None
        min_difference = float('inf')

        for _, sleeve_data in self.sleeve_dict.items():
            sleeve_diameter = sleeve_data.get("OD")
            if sleeve_diameter is not None:
                difference = abs(sleeve_diameter - diameter)
                if difference < min_difference:
                    min_difference = difference
                    closest_match = sleeve_data

        if closest_match:
            print(f"Closest match found with OD: {closest_match['OD']}")
        else:
            print("No matching sleeve found.")

        return closest_match

class SceneConstructor:
    def __init__(self, camera:Helios2):
        self.camera = camera
        self.scene_points = np.empty((0, 3))
        self.logger = Logger("Helios2.log", __name__)

    def capture(self, cam_position)-> None | np.ndarray:
        """Captures points from the camera and transforms them to the camera position.
        Args:
            cam_position (tuple): The camera position in the format (roll, pitch, yaw, x, y, z) where:
                - roll, pitch, yaw are the Euler angles in radians,
                - x, y, z are the camera position in mm.
        Returns:
            np.ndarray: Points captured from the camera, transformed to the camera position.
        """
        buffer = self.camera.capture_buffer()
        try:
            points = self.camera.buffer_to_numpy(buffer)
        except Exception as e:
            self.logger.log("Error capturing points", e, caller=self)
            raise e
        if points is not None and points.size > 0:
            try:
                pp = PointProcessor()
                # Rotate and translate points to the camera position
                R = pp.euler_to_R_mat(cam_position[0], cam_position[1], cam_position[2])
                T = pp.create_T_vec(cam_position[3], cam_position[4], cam_position[5])
                points = pp.h_transform(points, pp.create_H_mat(R, T))
            except Exception as e:
                self.logger.log("Error processing points", e, caller=self)
                raise e
            return points
        else:
            return None

    def add_points(
        self,
        points:np.ndarray,
    ):
        if self.scene_points.size == 0:
            self.scene_points= points
        else:
            if points is not None and points.size > 0:
                self.scene_points = np.vstack((self.scene_points, points))
            else:
                self.logger.log("No points to add.", caller=self)
    
    def reconstruct_pointcloud(
        self,
        voxel_size:float=100.0
    )-> None:
        """Simplifies pointcloud using voxel downsampling and removes outliers.
        """
        import open3d as o3d
        if self.scene_points.size == 0:
            self.logger.log("No points to reconstruct.", caller=self)
            return None
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.scene_points)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)
            return pcd
        except Exception as e:
            self.logger.log("Error reconstructing point cloud", e, caller=self)
            raise e

    def show_scene(self)-> None:
        """Visualizes the reconstructed point cloud using open3d.
        """
        if self.scene_points.size == 0:
            self.logger.log("No points to visualize.", caller=self)
            return
        try:
            pcd = self.scene_points
            if isinstance(pcd, np.ndarray):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.scene_points)
            elif not isinstance(pcd, o3d.geometry.PointCloud):
                raise TypeError("Points must be a numpy array or an open3d PointCloud object.")
            # Use Visualizer for custom camera settings
            vis = o3d.visualization.Visualizer() # type:ignore
            vis.create_window(window_name="Points", width=800, height=600, visible=True)
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            render_option = vis.get_render_option()
            render_option.point_size = 0.1
            ctr = vis.get_view_control()
            ctr.set_front([1.0, 0.0, 0.0])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.01)
            ctr.camera_local_translate(0, 0, 1000)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.destroy_window()
        except Exception as e:
            self.logger.log("Error visualizing scene", e, caller=self)
            raise e

    def show_points(self, points:np.ndarray):
        if points.size == 0:
            self.logger.log("No points to visualize.", caller=self)
            return
        try:
            import open3d as o3d
            pcd = points
            if isinstance(pcd, np.ndarray):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
            elif not isinstance(pcd, o3d.geometry.PointCloud):
                raise TypeError("Points must be a numpy array or an open3d PointCloud object.")
            # Use Visualizer for custom camera settings
            vis = o3d.visualization.Visualizer() # type:ignore
            vis.create_window(window_name="Points", width=800, height=600, visible=True)
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            render_option = vis.get_render_option()
            render_option.point_size = 0.1
            ctr = vis.get_view_control()
            ctr.set_front([1.0, 0.0, 0.0])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.01)
            ctr.camera_local_translate(0, 0, 1000)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.destroy_window()
        except Exception as e:
            self.logger.log("Error visualizing scene", e, caller=self)
            raise e


    def compare_with_scene(
        self,
        captured_points:np.ndarray,
        margin:float=50, #mm
        voxel_size:float=50, #mm
        show_voxels:bool=False,
        plot:bool=False,
        return_image:bool=False,
    )-> tuple[ np.ndarray, np.ndarray | None ]:
        """Compares captured points with the scene points and returns the points that are not in the scene.
        1. Voxelizes `self.scene_points`.
        2. Compares the captured points with the voxelized scene points.
        3. If `plot is True`, visualizes the captured points and the voxelized scene points.
        4. Returns the points that are not inside of the voxels.
        Args:
            captured_points (np.ndarray): Points captured from the camera.
            margin (float): Margin in mm to consider for comparison.
        Returns:
            np.ndarray: Points that are not in the scene.
        """
        image = None
        import open3d as o3d
        if self.scene_points.size == 0:
            self.logger.log("No scene points to compare with.", caller=self)
            return captured_points, None
        try:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(self.scene_points)
            scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)
            captured_pcd = o3d.geometry.PointCloud()
            captured_pcd.points = o3d.utility.Vector3dVector(captured_points)
            distances = captured_pcd.compute_point_cloud_distance(scene_pcd)
            mask = np.array(distances) > margin  # Points outside the margin
            outside_points = captured_points[mask]

            if plot or return_image:
                # Visualize the captured points and the scene points
                colors = np.zeros((captured_points.shape[0], 3))
                colors[mask] = [1, 0, 0]  # Red for outside points
                colors[~mask] = [0, 1, 0]  # Green for inside points
                captured_pcd.colors = o3d.utility.Vector3dVector(colors)
                
                geometries = [captured_pcd, scene_pcd]
                
                if show_voxels:
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(scene_pcd, voxel_size=voxel_size)
                    for voxel in voxel_grid.get_voxels():
                        voxel.color = [0.7, 0.7, 1.0]  # Light blue
                    geometries.append(voxel_grid)
                if plot:
                    o3d.visualization.draw_geometries(geometries, # type:ignore
                                                    window_name="Captured Points vs Scene",
                                                    width=800, height=600,
                                                    point_show_normal=False,
                                                    mesh_show_back_face=True)
                if return_image:
                    vis = o3d.visualization.Visualizer() # type:ignore
                    vis.create_window(visible=False)
                    vis.add_geometry(captured_pcd)
                    vis.update_geometry(captured_pcd)
                    # set point size
                    render_option = vis.get_render_option()
                    render_option.point_size = 0.1
                    ctr = vis.get_view_control()
                    # set position to [10, 10, 10]
                    ctr.set_front([1.0, 0.0, 0.0])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                    ctr.set_up([0.0, 0.0, 1.0])
                    ctr.set_zoom(0.01)
                    ctr.camera_local_translate(0, 0, 1000)
                    vis.poll_events()
                    vis.update_renderer()
                    image = vis.capture_screen_float_buffer(do_render=True)
                    image = np.asarray(image)
                    # Convert from float [0,1] to uint8 [0,255] if needed
                    image = (image * 255).astype(np.uint8)
                    # rotate -90 degrees
                    image = np.rot90(image, -1)
                    vis.destroy_window()
            return outside_points, image
        except Exception as e:
            self.logger.log("Error comparing with scene", e, caller=self)
            raise e

    def save_scene(self, filename:str):
        """Saves the current scene points to a file.
        Args:
            filename (str): The name of the file to save the scene points to.
        """
        if not filename.endswith('.npy'):
            filename += '.npy'
        if self.scene_points.size == 0:
            self.logger.log("No scene points to save.", caller=self)
            return
        try:
            np.save(filename, self.scene_points)
            self.logger.log(f"Scene points saved to {filename}.", caller=self)
        except Exception as e:
            self.logger.log(f"Error saving scene points to {filename}.", e, caller=self)

    def load_scene(self, filename:str):
        """Loads scene points from a file.
        Args:
            filename (str): The name of the file to load the scene points from.
        """
        if not filename.endswith('.npy'):
            filename += '.npy'
        if not os.path.exists(filename):
            self.logger.log(f"File {filename} does not exist.", caller=self)
            return
        try:
            self.scene_points = np.load(filename)
            self.logger.log(f"Scene points loaded from {filename}.", caller=self)
        except Exception as e:
            self.logger.log(f"Error loading scene points from {filename}.", e, caller=self)
            raise e

    def reset_scene(self):
        self.scene_points = np.empty((0, 3))

    def export_scene(self, type:str|None=None, filename:str="scene.ply"):
        """Exports the current scene points to a file in the specified format.
        Args:
            type (str): The format to export the scene points to. Supported formats are "ply" and "xyz".
            filename (str): The name of the file to save the scene points to.
        """
        options = ["ply", "xyz"]
        if type is None:
            for option in options:
                if filename.endswith(f".{option}"):
                    type = option
                    break
        if type not in options:
            raise ValueError(f"Unsupported export type. Supported types are: {', '.join(options)}")
        if filename.endswith(type) is False:
            filename += f".{type}"
        import open3d as o3d
        if self.scene_points.size == 0:
            self.logger.log("No scene points to export.", caller=self)
            return
        try:
            success = False
            write_failed = Exception("Export failed. Please check the file path and format.")
            match type:
                case "ply":
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(self.scene_points)
                    success = o3d.io.write_point_cloud(filename, pcd)
                    self.logger.log(f"Scene points exported to {filename} in PLY format.", caller=self)
                case "xyz":
                    print("TODO")
                    success = True
                case _:
                    raise ValueError(f"Unsupported export type: {type}. Supported types are: {', '.join(options)}")
            if success is False:
                self.logger.log("Write failed", write_failed, caller=self)
                raise write_failed
        except Exception as e:
            self.logger.log(f"Error exporting scene points to {filename}.", e, caller=self)
            raise e
        print("Sleeve rack capture finished.")

    def merge_scene(self, other_scene: "SceneConstructor"):
        """
        Merges the scene points from another SceneConstructor instance into this one.
        Args:
            other_scene (SceneConstructor): Another scene to merge.
        """
        if other_scene.scene_points.size == 0:
            self.logger.log("Other scene has no points to merge.", caller=self)
            return
        self.add_points(other_scene.scene_points)
        self.logger.log("Scenes merged.", caller=self)

    def ply_to_npy(self, filename:str):
        """Converts a PLY file to a NPY file.
        Args:
            filename (str): The name of the PLY file to convert.
        """
        if not filename.endswith('.ply'):
            filename += '.ply'
        if not os.path.exists(filename):
            self.logger.log(f"File {filename} does not exist.", caller=self)
            return
        try:
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            np.save(filename.replace('.ply', '.npy'), points)
            self.logger.log(f"Converted {filename} to {filename.replace('.ply', '.npy')}.", caller=self)
        except Exception as e:
            self.logger.log(f"Error converting {filename} to .npy.", e, caller=self)
            raise e