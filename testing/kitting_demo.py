    # # Iterate over Subtasks
    # for subtask in range(subtask):
        
    #     # Get Current Image

    #     # Send Image Into Generative Model

    #     # Infer Frames and get Trajectory

    #     # Execute Trajectory

    #     # Grip or ungrip based on the subtask.

    # Currently, Just test for a single subtask

import torch
from visionik.infer import ARTDepthInference
from aloha_scripts.robot_utils import move_grippers # requires aloha
from aloha_scripts.real_env import make_real_env # requires aloha

# Left finger position limits (qpos[7])
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

checkpoint_path = "/home/weixun/testing/vision-ik/testing/models/truncated_shufflenet/more_distortion_less_freq/checkpoint_epoch_444_step_880000.pth"
first_subtask_gif_path = "/home/weixun/testing/vision-ik/testing/data/first_seg_first_frame_out.gif"
env = make_real_env(init_node=True, setup_robots=True, setup_base=False)
ts = env.reset()
image_list = []

model = ARTDepthInference(checkpoint_path)

# Grip or ungrip based on the subtask.
move_grippers([env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open grippers

# Get Current Image
obs = ts.observation
if 'images' in obs:
    image_list.append(obs['images'])
else:
    image_list.append({'main': obs['image']})

# Grab first subtask GIF, infer Frames and get Trajectory
with torch.inference_mode():
    infer_outputs = model.infer_gif(first_subtask_gif_path)
infer_outputs = np.array(infer_outputs)
print("Infer Outputs Shape: ", infer_outputs.shape)

exit(0)
# Execute Trajectory
ts = env.step(infer_outputs[:-1], base_action=None) # What is returned here?

# Grip or ungrip based on the subtask.
move_grippers([env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)  # open grippers




            curr_image = self.get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))
            infer_image_list.append(np.copy(obs['images']['top']))

            if t == 0:
                # warm up
                for _ in range(10):
                    raw_action = self.policy(curr_image)
                print('network warm up done')
                time1 = time.time()

            while True:
                raw_action = self.policy(curr_image)
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:] 

                if real_robot:
                    ts = env.step(target_qpos, base_action)

                if real_robot:
                    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open grippers
