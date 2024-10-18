import concurrent.futures
import math
import os.path
import time
from copy import deepcopy
from enum import Enum
from math import pi
from threading import Lock, Thread, RLock
from typing import Optional, List, Tuple, Dict, Union

import ray
import rclpy
import tf_transformations as transformations
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ray.rllib import Policy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

from src.llms.llm_move_gen import LLMMoveGen
from src.marl.common_marl import setup
from src.marl.corridor_env import CorridorEnv
from src.utils.scenarios import MazeLikeCorridor
from src.utils.utils import Utils

policy_path_root = "PATH TO TRAINED MODEL CHECKPOINTS DIR"
policy_path = (lambda is_hybrid, ep,
                      trial: f"{policy_path_root}{'hybrid' if is_hybrid else 'marl'}/episode_{ep}/trial_{trial}")
csv_path_root = "OUTPUT PATH DIR"
REPEATS = 5


def get_n_trials(is_hybrid, episode):
  return len(os.listdir(os.path.join(policy_path_root, f"{'hybrid' if is_hybrid else 'marl'}/episode_{episode}/")))


alice_ns: str = "/turtlebot_d3_23_00"
bob_ns: str = "/turtlebot_d3_21_93"

assert alice_ns == "" or alice_ns[0] == "/"
assert bob_ns == "" or bob_ns[0] == "/"


class Dir(Enum):
  NORTH = 0
  EAST = 1
  SOUTH = 2
  WEST = 3


alice_starting_dir: Dir = Dir.NORTH
bob_starting_dir: Dir = Dir.SOUTH

scan_mx: Lock = Lock()
scan_ranges = []

finished_mx: Lock = Lock()
finished: bool = False


# Final actions parsed in ROS node
class RobotAction(Enum):
  FORWARD = 0
  BACKWARD = 1
  CLOCKWISE = 2
  ANTICLOCKWISE = 3


# Actions that come from the MARL network
class Action(Enum):
  WAIT = 0
  FORWARD = 1
  BACKWARD = 2
  LEFT = 3
  RIGHT = 4


# Intermediary actions because I was getting too confused between forwards in the direction of the robot (RobotAction) or forwards towards north (Action)
class CardinalAction(Enum):
  NORTH = 0,
  SOUTH = 1,
  EAST = 2,
  WEST = 3


def action_to_string(action: Action) -> str:
  return "@" + action.name.upper()


def action_from_string(string: str) -> Action:
  a = Action.WAIT
  if string == "@FORWARD":
    a = Action.FORWARD
  elif string == "@BACKWARD":
    a = Action.BACKWARD
  elif string == "@LEFT":
    a = Action.LEFT
  elif string == "@RIGHT":
    a = Action.RIGHT
  elif string == "@WAIT":
    a = Action.WAIT

  return a


def action_to_cardinal_action(action: Action) -> CardinalAction:
  if action == Action.FORWARD:
    return CardinalAction.NORTH
  elif action == Action.BACKWARD:
    return CardinalAction.SOUTH
  elif action == Action.LEFT:
    return CardinalAction.WEST
  elif action == Action.RIGHT:
    return CardinalAction.EAST


def map_action(dir: Dir, action: Union[Action, int]) -> Tuple[List[Optional[RobotAction]], Dir]:
  if type(action == int):
    action = Action(action)

  if action == Action.WAIT:
    return [None], dir

  a = action
  action = action_to_cardinal_action(action)

  act_map: Dict[Dir, Dict[CardinalAction, Tuple[List[RobotAction], Optional[Dir]]]] = {
    Dir.NORTH: {
      CardinalAction.NORTH: ([RobotAction.FORWARD], None),
      CardinalAction.SOUTH: ([RobotAction.BACKWARD], None),
      CardinalAction.WEST:  ([RobotAction.ANTICLOCKWISE, RobotAction.FORWARD], Dir.WEST),
      CardinalAction.EAST:  ([RobotAction.CLOCKWISE, RobotAction.FORWARD], Dir.EAST)
    },
    Dir.EAST:  {
      CardinalAction.NORTH: ([RobotAction.ANTICLOCKWISE, RobotAction.FORWARD], Dir.NORTH),
      CardinalAction.SOUTH: ([RobotAction.CLOCKWISE, RobotAction.FORWARD], Dir.SOUTH),
      CardinalAction.WEST:  ([RobotAction.BACKWARD], None),
      CardinalAction.EAST:  ([RobotAction.FORWARD], None)
    },
    Dir.SOUTH: {
      CardinalAction.NORTH: ([RobotAction.BACKWARD], None),
      CardinalAction.SOUTH: ([RobotAction.FORWARD], None),
      CardinalAction.WEST:  ([RobotAction.CLOCKWISE, RobotAction.FORWARD], Dir.WEST),
      CardinalAction.EAST:  ([RobotAction.ANTICLOCKWISE, RobotAction.FORWARD], Dir.EAST)
    },
    Dir.WEST:  {
      CardinalAction.NORTH: ([RobotAction.CLOCKWISE, RobotAction.FORWARD], Dir.NORTH),
      CardinalAction.SOUTH: ([RobotAction.ANTICLOCKWISE, RobotAction.FORWARD], Dir.SOUTH),
      CardinalAction.WEST:  ([RobotAction.FORWARD], None),
      CardinalAction.EAST:  ([RobotAction.BACKWARD], None)
    }
  }

  new_actions, new_dir = act_map[dir][action]

  if new_dir is None:
    new_dir = dir

  print(
      f"Old Dir {dir.name}")
  print(
      f"Model action {action.name} ({a})")
  print(
      f"New Dir {new_dir.name}")
  print(type(new_actions))
  print(type(new_actions[0]))
  print(
      f"Robot actions {new_actions}")
  # input()
  return new_actions, new_dir


def get_scan_ranges():
  with scan_mx:
    return scan_ranges


def set_scan_ranges(ranges):
  global scan_ranges
  with scan_mx:
    scan_ranges = ranges


def is_finished():
  with finished_mx:
    b = finished
  return b


def set_finished(b):
  global finished
  with finished_mx:
    finished = b


# Movement parameters
LINEAR_SPEED = 0.15  # m/s
ANGULAR_SPEED = math.radians(15)  # rads / s
LIDAR_THRESHOLD_M = 0.25
LIDAR_THRESHOLD_MIN = 0.0
COLLISION_DELAY_S = 10
GRID_SQUARE_SIZE_M = 0.5
EPSILON = 1e-10


class ScanSubscriber(Node):
  def __init__(self):
    super().__init__("scan_subscriber")

    qos = QoSProfile(
        reliability = ReliabilityPolicy.BEST_EFFORT,
        history = HistoryPolicy.KEEP_LAST,
        depth = 10
    )

    self.subscription = self.create_subscription(
        LaserScan,
        "/scan",
        self.listener_callback,
        qos_profile = qos
    )
    self.subscription

  def listener_callback(self, msg: LaserScan) -> None:
    # LIDAR feeds in with msg. ros2 topic echo to get this. Pick a useful arc

    start_angle = -3.12159 / 4
    end_angle = 3.12159 / 4

    start_idx = int((start_angle - msg.angle_min) / msg.angle_increment)
    end_idx = int((end_angle - msg.angle_min) / msg.angle_increment)

    set_scan_ranges(msg.ranges[start_idx:end_idx + 1])

  def run(self):
    self.get_logger().info("Scan subscriber started")
    while not is_finished():
      rclpy.spin_once(self)


class VelocityPublisher(Node):
  def __init__(self):
    super().__init__("velocity_publisher")
    self.raw_actions = None
    self.cmd_vel_publishers = {
      "alice": self.create_publisher(Twist, alice_ns + "/cmd_vel",
                                     10),
      "bob":   self.create_publisher(Twist, bob_ns + "/cmd_vel",
                                     10)
    }

    qos = QoSProfile(
        reliability = ReliabilityPolicy.BEST_EFFORT,
        history = HistoryPolicy.KEEP_LAST,
        depth = 10
    )

    self.odom_subscribers = {
      "alice": self.create_subscription(Odometry, alice_ns + "/odom", lambda msg: self.odom_callback(msg, "alice"),
                                        qos_profile = qos),
      "bob":   self.create_subscription(Odometry, bob_ns + "/odom", lambda msg: self.odom_callback(msg, "bob"),
                                        qos_profile = qos)
    }
    self.odom_subscribers["alice"]
    self.odom_subscribers["bob"]

    self.last_distance_difference = 0

    self.action_fn_map = {
      RobotAction.FORWARD:       self.pub_forwards,
      RobotAction.BACKWARD:      self.pub_backwards,
      RobotAction.CLOCKWISE:     self.pub_clockwise,
      RobotAction.ANTICLOCKWISE: self.pub_anticlockwise
    }

    self.dirs = {
      "alice": Dir.NORTH,
      "bob":   Dir.SOUTH
    }
    self.ids = ["alice", "bob"]
    self.locks = {i: RLock() for i in self.ids}
    self.odom_info = {i: {"position": (0., 0., 0.), "orientation": (0., 0., 0., 0.)} for i in self.ids}
    self.subscriber_thread = Thread(target = self.run_subscriber)
    self.subscriber_thread.start()

  def odom_callback(self, msg: Odometry, name: str):
    with self.locks[name]:
      self.odom_info[name] = {
        "position":    (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
        "orientation": (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w)
      }
      # self.get_logger().info(
      #     f"{tuple(map(math.degrees, transformations.euler_from_quaternion(self.get_odom(name)['orientation'])))}")
      # self.get_logger().info(f"{list(map(math.degrees, quaternion_to_euler(*self.get_odom(name)['orientation'])))}")

  def get_odom(self, name):
    with self.locks[name]:
      return self.odom_info[name]

  def run_subscriber(self):
    rclpy.spin(self)

  def run_movement(self, actions: List[Dict[str, Action]]):
    # assert (self.dirs["alice"] == Dir.NORTH and self.dirs["bob"] == Dir.SOUTH)

    self.dirs = {
      "alice": Dir.NORTH,
      "bob":   Dir.SOUTH
    }

    self.raw_actions = actions
    if self.raw_actions is not None:
      for action_pair in self.raw_actions:
        # input("Press key for next move")
        self.get_logger().info(f"Executing action pair {action_pair}")

        actions = {}
        for i in self.ids:
          try:
            print(f"{self.dirs}")
            print(f"{i}")
            actions[i], self.dirs[i] = map_action(self.dirs[i], action_pair[i])
          except KeyError:
            actions[i] = [None]

        if len(set(map(len, actions.values()))) == 1:  # Both turning -- no issue of collision
          assert list(actions.keys()) == list(self.ids)
          self.parallel_exec_actions(actions)

        else:  # One is turning -- risk of collision. Must do one at a time.
          min_num_acts = 100000000
          for i in self.ids:
            min_num_acts = min(min_num_acts, len(actions[i]))

          acts_left = {}
          for i in self.ids:
            if len(actions[i]) == min_num_acts:
              self.exec_action(i, actions[i])
            else:
              acts_left[i] = actions[i]

          self.parallel_exec_actions(acts_left)

  def parallel_exec_actions(self, actions: Dict[str, List[RobotAction]]):
    self.get_logger().info("Starting movement")
    with concurrent.futures.ThreadPoolExecutor(max_workers = len(self.ids)) as executor:
      futures = {executor.submit(self.exec_action, i, actions[i]): i for i in actions.keys()}
      for future in concurrent.futures.as_completed(futures):
        future.result()
    self.get_logger().info("Movement finished")

  def exec_action(self, name: str, actions: List[RobotAction]):
    if actions is not None:
      for a in actions:
        if a is not None:
          self.action_fn_map[a](name)

  # Returns true if the agent can move forwards
  def area_clear(self, distance_to_check) -> bool:
    ranges = get_scan_ranges()

    # if any(list(map(lambda r: LIDAR_THRESHOLD_M > r > LIDAR_THRESHOLD_MIN, ranges))):
    while any(list(map(lambda r: distance_to_check > r > LIDAR_THRESHOLD_MIN, ranges))):
      info(f"Possible obstruction! Delaying for {COLLISION_DELAY_S} seconds")
      self.delay(COLLISION_DELAY_S)
      ranges = get_scan_ranges()

    return True
    # if any(list(map(lambda r: LIDAR_THRESHOLD_M > r > LIDAR_THRESHOLD_MIN, ranges))):
    #  info(f"Obstruction still present")
    # robot_interrupt_event.set()

  def delay(self, t_target):
    t0 = self.get_clock().now()
    while self.get_clock().now() - t0 < rclpy.duration.Duration(seconds = t_target):
      pass
    self.get_logger().info(f"Delayed for {t_target} seconds")

  def _publish_cmd(self, msg: Twist, name: str):
    self.cmd_vel_publishers[name].publish(msg)
    # self.get_logger().info(f"Publishing command for {name}")

  def _publish_zero(self, name: str):
    # self.get_logger().info(f"Zero velocity requested")
    msg = Twist()

    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0

    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0

    self._publish_cmd(msg, name)

  def _pub_linear(self, dir: int, dist: float, name: str):
    if dist == 0:
      return

    msg = Twist()

    msg.linear.x = dir * LINEAR_SPEED
    msg.linear.y = 0.0
    msg.linear.z = 0.0

    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0

    axis = 0 if self.dirs[name] == Dir.NORTH or self.dirs[name] == Dir.SOUTH else 1

    pos = self.get_odom(name)["position"]
    rpy = transformations.euler_from_quaternion(self.get_odom(name)['orientation'])

    dx = dir * dist * math.cos(rpy[2])
    dy = dir * dist * math.sin(rpy[2])

    # goal_pos = list(pos)
    # goal_pos[0] += dx
    # goal_pos[1] += dy
    # goal_pos = tuple(goal_pos)

    new_pos = self.get_odom(name)["position"]
    new_rpy = transformations.euler_from_quaternion(self.get_odom(name)['orientation'])

    diff_x = new_pos[0] - pos[0]
    diff_y = new_pos[1] - pos[1]
    d = math.sqrt(diff_x ** 2 + diff_y ** 2)

    # self.get_logger().info(f"dx {dx}; diff_x {diff_x}; dy {dy}; diff_y {diff_y}")

    while d <= dist:
      new_pos = self.get_odom(name)["position"]
      new_rpy = transformations.euler_from_quaternion(self.get_odom(name)['orientation'])

      diff_x = new_pos[0] - pos[0]
      diff_y = new_pos[1] - pos[1]
      d = math.sqrt(diff_x ** 2 + diff_y ** 2)

      # self.get_logger().info(
      #     f"Start {pos}, {rpy}; Current position: {new_pos}, {new_rpy};")
      # self.get_logger().info(f"d {d}; diff_x {diff_x}; diff_y {diff_y}")
      self._publish_cmd(msg, name)
      time.sleep(0.1)
    self._publish_zero(name)
    # self.get_logger().info(
    #     f"Start {pos}, {rpy}; Current position: {new_pos}, {new_rpy};")

  def _pub_rotation(self, dir: float, angle: float, name: str):
    if angle == 0:
      return

    msg = Twist()

    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0

    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = dir * ANGULAR_SPEED

    q_orig = self.get_odom(name)["orientation"]

    quat_error = transformations.quaternion_multiply(
        self.get_odom(name)['orientation'],
        transformations.quaternion_inverse(q_orig))
    euler_error = transformations.euler_from_quaternion(quat_error)

    # self.get_logger().info(f"orig rpy {transformations.euler_from_quaternion(q_orig)}")
    # self.get_logger().info(f"euler error: {euler_error[2]}")
    # self.get_logger().info(f"euler error w dir: {euler_error[2] * dir}")
    # self.get_logger().info(f"q_orig_inv {q_orig_inv}")
    # self.get_logger().info(f"q_rot {q_rot}")

    while (euler_error[2] * dir) <= angle:
      quat_error = transformations.quaternion_multiply(
          self.get_odom(name)['orientation'],
          transformations.quaternion_inverse(q_orig))

      # transformations.quaternion_multiply(q_orig, self.get_odom(name)['orientation'])
      euler_error = transformations.euler_from_quaternion(quat_error)
      # self.get_logger().info(f"orig rpy {transformations.euler_from_quaternion(q_orig)}")
      # self.get_logger().info(f"curr rpy {transformations.euler_from_quaternion(self.get_odom(name)['orientation'])}")
      # self.get_logger().info(f"euler error: {euler_error[2]}")
      # self.get_logger().info(f"euler error w dir: {euler_error[2] * dir}")
      self._publish_cmd(msg, name)
      time.sleep(0.1)
    self._publish_zero(name)

  def pub_forwards(self, name: str):  # m
    dist = GRID_SQUARE_SIZE_M
    self.get_logger().info(f"{name.capitalize()}: Forwards {dist} m command")
    self._pub_linear(1, abs(dist), name)

  def pub_backwards(self, name: str):  # m
    dist = GRID_SQUARE_SIZE_M
    self.get_logger().info(f"{name.capitalize()}: Backwards {dist} m command")
    self._pub_linear(-1, abs(dist), name)

  def pub_anticlockwise(self, name: str):  # rads
    angle = -pi / 2
    self.get_logger().info(f"{name.capitalize()}: Anticlockwise {angle} rad command")
    self._pub_rotation(1, abs(angle), name)

  def pub_clockwise(self, name: str):  # rads
    angle = pi / 2
    self.get_logger().info(f"{name.capitalize()}: Clockwise {angle} rad command")
    self._pub_rotation(-1, abs(angle), name)

  def stop_all(self):
    for i in self.ids:
      self._publish_zero(i)


def get_actions_from_model(path):
  moves = []

  setup()

  policy: Policy = Policy.from_checkpoint(path)

  # algo = Algorithm.from_checkpoint(
  #     "/home/toby/projects/uni/internship/Hybrid_LLM_MARL/output/CentralizedCritic_2024-08-21_12-09-09/CentralizedCritic_corridor_c9bf8_00000_0_2024-08-21_12-09-10/checkpoint_000008",
  #     policy_ids = ["pol1"],
  #     policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "pol1"
  # )

  env = CorridorEnv({
    "csv_path":            "none",
    "csv_filename":        "none",
    "episode_step_limit":  50,
    "env_change_rate_eps": 0,  # 0 for no env change
    "scenario_name":       "Maze_Like_Corridor",
    "worker_index":        1
  })

  done = {'__all__': False}
  total_reward = 0
  observations = env.reset()[0]
  print(observations)
  steps = 0
  moves = []

  while not done["__all__"]:
    # action = algo.compute_actions(observations, policy_id = "pol1")
    action = {}
    for i in env.get_agent_ids():
      action[i] = policy.compute_single_action(observations[i])[0]
    print(f"actions: {action}")
    observations, reward, done, trunc, info = env.step(action)
    act_pair = {}
    for i in env.get_agent_ids():
      try:
        if info[i]["valid_move"]:
          act_pair[i] = action[i]
        else:
          act_pair[i] = 0
      except KeyError:
        act_pair[i] = 0
    moves.append(act_pair)
    steps += 1
    print(f"observations: {observations}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"trunc: {trunc}")
    print(f"info: {info}")
    total_reward += sum(reward.values())
    print(f"total_reward {total_reward}")
    print(f"steps: {steps}")

  loc_list = [(env.agent_pos[i], env.agent_goal_pos[i], env.agent_starting_pos[i]) for i in env.get_agent_ids()]
  avg_perf = Utils.calc_multiagent_avg_perf(loc_list)
  print(avg_perf)
  print(moves)

  Utils.write_csv(
      "ros",
      ["episode", "alice_start", "alice_end", "alice_goal", "bob_start", "bob_end", "bob_goal", "performance",
       "scenario", "env_change_rate"],
      [path.split("/")[-2].split("_")[1], env.agent_starting_pos["alice"], env.agent_pos["alice"],
       env.agent_goal_pos["alice"],
       env.agent_starting_pos["bob"], env.agent_pos["bob"], env.agent_goal_pos["bob"],
       avg_perf, env.scenario.name, env.env_change_rate_eps])

  return moves


def get_actions_from_llm():
  s = MazeLikeCorridor()
  g = LLMMoveGen(s,
                 ["alice", "bob"],
                 s.valid_pos,
                 {"alice": (1, 0), "bob": (1, 7)},
                 {"alice": (1, 7), "bob": (1, 0)},
                 "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 0,
                 write_csv = True,
                 path_name = "ros",
                 write_conversation = True)
  moves_dict, perf = g.gen_moves(50, translate_moves = True, verbose = True)
  print(f"Plan perf: {perf}")
  return moves_dict


def main(args = None):
  rclpy.init()
  velocity_publisher = VelocityPublisher()
  ray.init()

  for is_hybrid in [0, 1, 2]: 
    for episode in [100, 350, 600, 850, 1100, 1350, 1600]:
      # is_hybrid = False
      # episode = 1600
      trial = 0

      input(f"Press to run {is_hybrid}/{episode}")

      # scan_subscriber = ScanSubscriber()
      Utils.set_csv_file("ros", csv_path_root,
                         f"{'hybrid' if is_hybrid == 0 else ('marl' if is_hybrid == 1 else 'llm')}_ros_results.csv")
      if is_hybrid == 0:
        actions = get_actions_from_model(policy_path(True, episode, trial))
      elif is_hybrid == 1:
        actions = get_actions_from_model(policy_path(False, episode, trial))
      else:
        actions = get_actions_from_llm()

      input("Press any key to execute")
      
      move_thread = Thread(target = velocity_publisher.run_movement, args = [actions])
      move_thread.start()
      
      move_thread.join()
      set_finished(True)

  velocity_publisher.destroy_node()
  # scan_subscriber.destroy_node()
  ray.shutdown()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
