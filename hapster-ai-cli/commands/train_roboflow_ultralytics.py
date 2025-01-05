import os
import sys
import cv2
from ultralytics import YOLO
from roboflow import Roboflow

class TrainRoboflowUltralytics:
    def __init__(self, api_key=None, config=None):
        self.api_key    = api_key
        self.config     = config

        # Configuration
        self.rf_config      = self.config.get('roboflow')
        self.rf_location    = self.rf_config.get('location') or 'tmp'
        self.rf_workspace   = self.rf_config.get('workspace')
        self.rf_project     = self.rf_config.get('project')
        self.rf_version     = self.rf_config.get('version')
        self.rf_format      = self.rf_config.get('format')

        self.u_config   = self.config.get('ultralytics')
        self.model_file = self.u_config.get('model_file')
        self.epochs     = self.u_config.get('epochs')
        self.imgsz      = self.u_config.get('imgsz')
        self.batch      = self.u_config.get('batch')
        self.device     = self.u_config.get('device')

    def execute(self):
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.rf_workspace).project(self.rf_project)
        version = self.rf_version
        dataset = version.download(self.rf_format, location=self.rf_location)

        self.model = YOLO(self.model_file)

        self.model.train(
            data=f'{dataset.location}/data.yaml',
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device
        )

        print("Done.")
