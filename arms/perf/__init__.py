"""Keep the control loops on time on the Jetson.

Three separate answers to one problem: the teleop process is a single CPython
interpreter, so every hot thread -- the piper_sdk CAN decoder, ``control_loop``,
``ik_solver``, ``joint_state``, the quest reader -- serialises on one GIL.

Nothing here is imported for its own sake; each module exists because a measured
source of control-loop jitter needed removing. Deliberately NOT re-exported at
package level: ``cv_threads`` and ``gil_tuning`` must be callable cheaply at
process start, and importing them should not drag in ``piper_process``.
"""
