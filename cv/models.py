"""
StrideIQ CV models.py
Production-ready Pydantic models for the CV subsystem.
"""

from __future__ import annotations 
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class Grade(str, Enum):
    EXCELLENT="Excellent"
    GOOD="Good"
    FAIR="Fair"
    NEEDS_IMPROVEMENT="Needs Improvement"
    HIGH_PRIORITY="High Priority"

class MetricName(str, Enum):
    SHOULDER_RELAXATION="Shoulder Relaxation"
    ARM_SWING="Arm Swing"
    ARM_CROSSOVER="Arm Crossover"
    TRUNK_LEAN="Trunk Lean"
    KNEE_DRIVE="Knee Drive"

class Landmark(BaseModel):
    model_config=ConfigDict(frozen=True)
    x: float
    y: float
    z: float
    visibility: float

class BodyPose(BaseModel):
    model_config=ConfigDict(frozen=True)
    nose: Landmark
    left_eye: Landmark
    right_eye: Landmark
    left_ear: Landmark
    right_ear: Landmark
    left_shoulder: Landmark
    right_shoulder: Landmark
    left_elbow: Landmark
    right_elbow: Landmark
    left_wrist: Landmark
    right_wrist: Landmark
    left_hip: Landmark
    right_hip: Landmark
    left_knee: Landmark
    right_knee: Landmark
    left_ankle: Landmark
    right_ankle: Landmark
    left_heel: Landmark
    right_heel: Landmark
    left_foot_index: Landmark
    right_foot_index: Landmark

class VideoMetadata(BaseModel):
    model_config=ConfigDict(frozen=True)
    filename:str
    filepath:str
    fps:float
    frame_count:int
    width:int
    height:int
    duration_seconds:float
    codec: Optional[str]=None

class FramePacket(BaseModel):
    model_config=ConfigDict(arbitrary_types_allowed=True)
    frame_index:int
    timestamp:float
    body:BodyPose
    pose_detected:bool
    inference_time_ms:float

# biomechanics.py imports "PoseFrame" but this class was always named
# FramePacket — alias added rather than renaming the class everywhere,
# since pose_estimator.py and other modules already reference FramePacket.
PoseFrame = FramePacket

class FeatureVector(BaseModel):
    features:Dict[str,float]=Field(default_factory=dict)

class Prediction(BaseModel):
    metric:MetricName
    score:float=Field(ge=0,le=100)
    grade:Grade
    confidence:float=Field(ge=0,le=1)

class MetricResult(BaseModel):
    metric:MetricName
    score:float=Field(ge=0,le=100)
    grade:Grade
    confidence:float=Field(ge=0,le=1)
    explanation:str

class RunningReport(BaseModel):
    schema_version:str="1.0"
    video:VideoMetadata
    overall_score:float=Field(ge=0,le=100)
    overall_grade:Grade
    metrics:List[MetricResult]
    strengths:List[str]=Field(default_factory=list)
    improvements:List[str]=Field(default_factory=list)
    wearable_goals:List[str]=Field(default_factory=list)
    llm_summary:Optional[str]=None

class CVPipelineOutput(BaseModel):
    metadata:VideoMetadata
    feature_vector:Optional[FeatureVector]=None
    predictions:Optional[List[Prediction]]=None
    report:Optional[RunningReport]=None 