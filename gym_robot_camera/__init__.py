from gym.envs.registration import register

register(
    id='robot-camera-v0',
    entry_point='gym_robot_camera.envs:RobotCameraEnvV0',
)

register(
    id='robot-camera-v1',
    entry_point='gym_robot_camera.envs:RobotCameraEnvV1',
)

register(
    id='track-point-v0',
    entry_point='gym_robot_camera.envs:TrackPointEnvV0',
)