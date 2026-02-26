"""
VisualRenderer component for drawing body landmarks and connections on video frames.

This module provides visualization of MediaPipe Holistic detection results by drawing
landmarks and connections for pose and hands on video frames using MediaPipe's
drawing utilities.
"""

import mediapipe as mp
import numpy as np
from typing import Optional


class VisualRenderer:
    """
    Renders body landmarks and connections on video frames.
    
    This class uses MediaPipe's drawing utilities to visualize detection results
    with distinct colors for different body parts (pose, left hand, right hand).
    """
    
    def __init__(self):
        """
        Initialize the VisualRenderer with MediaPipe drawing utilities.
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define custom drawing specs for distinct colors
        self.pose_landmark_style = self.mp_drawing.DrawingSpec(
            color=(245, 117, 66), thickness=2, circle_radius=2
        )
        self.pose_connection_style = self.mp_drawing.DrawingSpec(
            color=(245, 66, 230), thickness=2
        )
        
        self.left_hand_landmark_style = self.mp_drawing.DrawingSpec(
            color=(121, 255, 121), thickness=2, circle_radius=2
        )
        self.left_hand_connection_style = self.mp_drawing.DrawingSpec(
            color=(121, 255, 121), thickness=2
        )
        
        self.right_hand_landmark_style = self.mp_drawing.DrawingSpec(
            color=(255, 121, 121), thickness=2, circle_radius=2
        )
        self.right_hand_connection_style = self.mp_drawing.DrawingSpec(
            color=(255, 121, 121), thickness=2
        )
    
    def render(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw all landmarks and connections on the frame.
        
        Args:
            frame: BGR image as numpy array (height, width, 3)
            results: MediaPipe Holistic results containing landmark data
            
        Returns:
            Annotated frame with landmarks and connections drawn
        """
        if results is None:
            return frame
        
        # Draw pose landmarks and connections
        frame = self.draw_pose(frame, results.pose_landmarks)
        
        # Draw hand landmarks and connections
        frame = self.draw_hands(
            frame,
            results.left_hand_landmarks,
            results.right_hand_landmarks
        )
        
        return frame
    
    def draw_pose(self, frame: np.ndarray, pose_landmarks) -> np.ndarray:
        """
        Draw pose landmarks and connections on the frame (excluding facial landmarks).
        
        Args:
            frame: BGR image as numpy array
            pose_landmarks: MediaPipe pose landmarks or None
            
        Returns:
            Frame with pose landmarks drawn
        """
        if pose_landmarks is not None:
            try:
                # Create a custom connection set that excludes facial landmarks
                # MediaPipe pose landmarks 0-10 are facial landmarks
                # We only want to draw body landmarks (11-32)
                pose_connections = [
                    connection for connection in self.mp_holistic.POSE_CONNECTIONS
                    if connection[0] >= 11 and connection[1] >= 11
                ]
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    pose_connections,
                    landmark_drawing_spec=self.pose_landmark_style,
                    connection_drawing_spec=self.pose_connection_style
                )
            except Exception as e:
                print(f"Error drawing pose landmarks: {e}")
        
        return frame
    
    def draw_hands(
        self,
        frame: np.ndarray,
        left_hand_landmarks,
        right_hand_landmarks
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections for both hands with distinct colors.
        
        Args:
            frame: BGR image as numpy array
            left_hand_landmarks: MediaPipe left hand landmarks or None
            right_hand_landmarks: MediaPipe right hand landmarks or None
            
        Returns:
            Frame with hand landmarks drawn
        """
        # Draw left hand in green
        if left_hand_landmarks is not None:
            try:
                self.mp_drawing.draw_landmarks(
                    frame,
                    left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.left_hand_landmark_style,
                    connection_drawing_spec=self.left_hand_connection_style
                )
            except Exception as e:
                print(f"Error drawing left hand landmarks: {e}")
        
        # Draw right hand in red
        if right_hand_landmarks is not None:
            try:
                self.mp_drawing.draw_landmarks(
                    frame,
                    right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.right_hand_landmark_style,
                    connection_drawing_spec=self.right_hand_connection_style
                )
            except Exception as e:
                print(f"Error drawing right hand landmarks: {e}")
        
        return frame
