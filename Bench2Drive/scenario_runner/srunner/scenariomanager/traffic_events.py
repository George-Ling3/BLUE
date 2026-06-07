#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of TrafficEvents
"""

from enum import Enum


class TrafficEventType(Enum):

    """
    This enum represents different traffic events that occur during driving.
    """

    NORMAL_DRIVING = 0
    COLLISION_STATIC = 1
    COLLISION_VEHICLE = 2
    COLLISION_PEDESTRIAN = 3
    ROUTE_DEVIATION = 4
    ROUTE_COMPLETION = 5
    ROUTE_COMPLETED = 6
    TRAFFIC_LIGHT_INFRACTION = 7
    WRONG_WAY_INFRACTION = 8
    ON_SIDEWALK_INFRACTION = 9
    STOP_INFRACTION = 10
    OUTSIDE_LANE_INFRACTION = 11
    OUTSIDE_ROUTE_LANES_INFRACTION = 12
    VEHICLE_BLOCKED = 13
    MIN_SPEED_INFRACTION = 14
    YIELD_TO_EMERGENCY_VEHICLE = 15
    SCENARIO_TIMEOUT = 16


class TrafficEvent(object):

    """
    TrafficEvent definition
    """

    def __init__(self, event_type, frame, message="", dictionary=None, agent_step=None):
        """
        Initialize object

        :param event_type: TrafficEventType defining the type of traffic event
        :param frame: frame in which the event happened 
        :param message: optional message to inform users of the event
        :param dictionary: optional dictionary with arbitrary keys and values
        :param agent_step: optional agent step number for mapping to metric_info.json
        """
        self._type = event_type
        self._frame = frame
        self._message = message
        self._dict = dictionary
        self._agent_step = agent_step
        
        # add timestamp information
        from srunner.scenariomanager.timer import GameTime
        self._timestamp = GameTime.get_time()
        self._frame_number = frame

    def get_type(self):
        """return the type"""
        return self._type

    def get_frame(self):
        """return the frame"""
        return self._frame

    def set_frame(self, frame):
        """return the frame"""
        self._frame = frame

    def set_message(self, message):
        """Set message"""
        self._message = message

    def get_message(self):
        """returns the message"""
        return self._message

    def set_dict(self, dictionary):
        """Set dictionary"""
        self._dict = dictionary

    def get_dict(self):
        """returns the dictionary"""
        return self._dict

    def get_timestamp(self):
        """return the timestamp"""
        return self._timestamp

    def get_frame_number(self):
        """return the frame number"""
        return self._frame_number

    def get_agent_step(self):
        """return the agent step number"""
        return self._agent_step

    def set_agent_step(self, agent_step):
        """set the agent step number"""
        self._agent_step = agent_step