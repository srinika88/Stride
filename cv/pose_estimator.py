"""StrideIQ pose_estimator.py"""
from __future__ import annotations
import time
import cv2
import mediapipe as mp
from cv.models import Landmark,BodyPose,FramePacket

class PoseEstimator:
    def __init__(self,static_image_mode=False,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5,draw=False):
        self.draw=draw
        self.mp_pose=mp.solutions.pose
        self.drawer=mp.solutions.drawing_utils
        self.pose=self.mp_pose.Pose(static_image_mode=static_image_mode,model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)
    def _lm(self,lms,i):
        p=lms[i]
        return Landmark(x=p.x,y=p.y,z=p.z,visibility=p.visibility)
    def _body(self,lms): 
        P=self.mp_pose.PoseLandmark
        return BodyPose(
            nose=self._lm(lms,P.NOSE.value),
            left_eye=self._lm(lms,P.LEFT_EYE.value),right_eye=self._lm(lms,P.RIGHT_EYE.value),
            left_ear=self._lm(lms,P.LEFT_EAR.value),right_ear=self._lm(lms,P.RIGHT_EAR.value),
            left_shoulder=self._lm(lms,P.LEFT_SHOULDER.value),right_shoulder=self._lm(lms,P.RIGHT_SHOULDER.value),
            left_elbow=self._lm(lms,P.LEFT_ELBOW.value),right_elbow=self._lm(lms,P.RIGHT_ELBOW.value),
            left_wrist=self._lm(lms,P.LEFT_WRIST.value),right_wrist=self._lm(lms,P.RIGHT_WRIST.value),
            left_hip=self._lm(lms,P.LEFT_HIP.value),right_hip=self._lm(lms,P.RIGHT_HIP.value),
            left_knee=self._lm(lms,P.LEFT_KNEE.value),right_knee=self._lm(lms,P.RIGHT_KNEE.value),
            left_ankle=self._lm(lms,P.LEFT_ANKLE.value),right_ankle=self._lm(lms,P.RIGHT_ANKLE.value),
            left_heel=self._lm(lms,P.LEFT_HEEL.value),right_heel=self._lm(lms,P.RIGHT_HEEL.value),
            left_foot_index=self._lm(lms,P.LEFT_FOOT_INDEX.value),right_foot_index=self._lm(lms,P.RIGHT_FOOT_INDEX.value)
        )
    def process(self,frame_packet):
        img=frame_packet.frame
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        t=time.perf_counter()
        res=self.pose.process(rgb)
        dt=(time.perf_counter()-t)*1000
        if not res.pose_landmarks:
            return None
        if self.draw:
            self.drawer.draw_landmarks(img,res.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
        body=self._body(res.pose_landmarks.landmark)
        return FramePacket(frame_index=frame_packet.frame_index,timestamp=frame_packet.timestamp,
                           body=body,pose_detected=True,inference_time_ms=dt)
    def close(self):
        self.pose.close()
