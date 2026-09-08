MoveIt robot-isolation patch v2
===============================

Final frame policy:
  RB5 : MoveIt planning_frame=link0, group=rb5_arm
  UR5e: MoveIt planning_frame=world, group=ur5e_arm
        (workflow equipment base_frame may remain base_link; world->base_link
         is the fixed identity joint in the UR MoveIt model)

Important v2 behavior:
  - If /move_group already exists AND matches the selected robot, REUSE it.
  - Do not kill the correct MoveGroup launched during Connect.
  - Only terminate/relaunch when the active MoveGroup is for the wrong robot.
  - After startup, verify robot model, SRDF group, planning frame, and tip link
    before sending IK / planning-scene requests.

This fixes the v1 failure:
  "failed to remove stale move_group ... /move_group"
which occurred because v1 tried to remove even the currently correct MoveGroup.

Python/YAML only; no colcon rebuild is required.
