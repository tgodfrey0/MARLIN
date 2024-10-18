from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.event_handlers import OnShutdown
from launch.substitutions import FindExecutable


def generate_launch_description():
  return LaunchDescription([
    Node(
        package = 'cmd_listener',
        executable = 'listener',
        name = 'listener'
    )
  ])
