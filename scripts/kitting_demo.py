    # # Iterate over Subtasks
    # for subtask in range(subtask):
        
    #     # Get Current Image

    #     # Send Image Into Generative Model

    #     # Infer Frames and get Trajectory

    #     # Execute Trajectory

    #     # Grip or ungrip based on the subtask.

    # Currently, Just test for a single subtask

import torch
from visionik.infer import VisionIKInference
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

model = VisionIKInference(checkpoint_path)

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



# sample_per_seq = 7
# model_checkpoint_path = "/home/weixun/testing/vision-ik/testing/models/truncated_shufflenet/more_distortion_less_freq/checkpoint_epoch_444_step_880000.pth"
# model = VisionIKInference(model_checkpoint_path)
# tuple_list = [(first_qpos, "first_seg_first_frame_out.gif"), 
#               (second_qpos, "second_seg_first_frame_out.gif"),
#                 (third_qpos, "third_seg_first_frame_out.gif"),
#                 (fourth_qpos, "fourth_seg_first_frame_out.gif"),
#                 (fifth_qpos, "fifth_seg_first_frame_out.gif"),
#                 (sixth_qpos, "sixth_seg_first_frame_out.gif")
#               ]

# def infer_and_visualize(qpos_list, gif_path, plot_dir="./eval_plots/", ylim=(-1, 1), label_overwrite=None):
    
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)
#     seg_name = gif_path[:-20]

#     # Way to sample from dataset after split:
#     N = len(qpos_list)  # Total number of frames in the sequence
#     t_axis = []
#     # Uniformly sample {self.sample_per_seq} frames from the sequence
#     for i in range(sample_per_seq-1):
#         t_axis.append(int(i*(N-1)/(sample_per_seq-1)))
#     t_axis.append(N-1)  # Ensure the last frame is always included

#     qpos_y = qpos_list[t_axis]

#     if label_overwrite:
#         label1, label2 = label_overwrite
#     else:
#         label1, label2 = 'State', 'Inference'

#     # Performing Inferenct on GIF
#     infer_outputs = model.infer_gif(gif_path)
#     infer_outputs = np.array(infer_outputs)

#     # Plotting Preparations
#     qpos = np.array(qpos_list) # ts, dim
#     num_ts, num_dim = qpos.shape
#     h, w = 2, num_dim
#     num_figs = num_dim
#     fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

#     # plot joint state
#     all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
#     for dim_idx in range(num_dim):
#         ax = axs[dim_idx]
#         ax.plot(qpos[:, dim_idx], label=label1)
#         ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
#         ax.legend()

#     # plot Joint state where comparisons matter
#     for dim_idx in range(num_dim):
#         ax = axs[dim_idx]
#         ax.scatter(t_axis, qpos_y[:, dim_idx], label=label1)
#         ax.legend()
    
#     # Scatter Plotting Inference Points
#     for dim_idx in range(num_dim):
#         ax = axs[dim_idx]
#         ax.scatter(t_axis, infer_outputs[:, dim_idx], label=label2)
#         ax.legend()

#     if ylim:
#         for dim_idx in range(num_dim):
#             ax = axs[dim_idx]
#             ax.set_ylim(ylim)

#     # Adding the super title
#     fig.suptitle(f'Inference Results for {seg_name}', fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust the rect to fit the suptitle

#     plot_path = os.path.join(plot_dir, f"infer_{seg_name}.png")
#     plt.savefig(plot_path)
#     print(f'Saved qpos plot to: {plot_path}')
#     plt.close()

# for seg, infer_gif in tuple_list:
#     infer_and_visualize(seg, infer_gif)