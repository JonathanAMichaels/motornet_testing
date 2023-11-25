import numpy as np
import json
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
from simple_task import CentreOutFFMinJerk
from simple_policy import Policy


def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    xw = x[0:w*rows].reshape((rows,cols)).mean(axis=1)
    return xw

def plot_training_log(log,loss_type,w=50,figsize=(10,3)):
    """
        loss_type: 'position_loss' or 'hidden_loss' or 'muscle_loss' or 'overall_loss'
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(log,dict):
       log = log[loss_type]
    logw = window_average(np.array(log),w=w)

    x = np.linspace(0,np.size(logw)-1,np.size(logw)) * w
    ax.semilogy(x, logw)

    ax.set_ylabel("Loss")
    ax.set_xlabel(f"Batch #")
    return fig, ax

  
def plot_simulations(xy, target_xy, figsize=(5,3)):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)

    ax.scatter(target_x, target_y)

    fig.tight_layout()
    return fig, ax

def plot_activation(all_hidden, all_muscles):
    n = np.shape(all_muscles)[0]
    fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(6,10))

    x = np.linspace(0, 1, 100)

    for i in range(n):
        ax[i,0].plot(x,np.array(all_muscles[i,:,:]))
        ax[i,1].plot(x,np.array(all_hidden[i,:,:]))
        
        ax[i,0].set_ylabel('muscle act (au)')
        ax[i,1].set_ylabel('hidden act (au)')
        ax[i,0].set_xlabel('time (s)')
        ax[i,1].set_xlabel('time (s)')
    fig.tight_layout()
    return fig, ax

def plot_kinematics(all_xy, all_tg, all_vel):
    n = np.shape(all_xy)[0]
    fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(6,10))

    x = np.linspace(0, 1, all_xy.size(dim=1))
    tgvel = np.diff(np.array(all_tg), axis=1)*100

    for i in range(n):
        ax[i,0].plot(x,np.array(all_tg[i,:,:]), '--')
        ax[i,0].plot(x,np.array(all_xy[i,:,:]), '-')
        ax[i,1].plot(x[1:],np.array(tgvel[i,:,:]), '--')
        ax[i,1].plot(x,np.array(all_vel[i,:,:]), '-')
        
        ax[i,0].set_ylabel('xy,tg')
        ax[i,1].set_ylabel('vel')
        ax[i,0].set_xlabel('time (s)')
        ax[i,1].set_xlabel('time (s)')
    fig.tight_layout()
    return fig, ax


def run_episode(env, policy, batch_size=1, catch_trial_perc=50, condition='train', ff_coefficient=None, test_mov_dur=0.500, detach=False):

  try:
    h = policy.module.init_hidden(batch_size=batch_size)
  except AttributeError:
    h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition        = condition,
                        catch_trial_perc = catch_trial_perc, 
                        ff_coefficient   = ff_coefficient, 
                        test_mov_dur     = test_mov_dur, 
                        options          = {'batch_size': batch_size})
  terminated = False

  # Initialize a dictionary to store lists
  data = {
      'xy': [],
      'tg': [],
      'vel': [],
      'all_actions': [],
      'all_hidden': [],
      'all_muscle': [],
      'all_force': [],
  }

  while not terminated:
      # Append data to respective lists
      data['all_hidden'].append(h[0, :, None, :])
      data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])

      action, h = policy(obs, h)
      obs, _, terminated, _, info = env.step(action=action)

      data['xy'].append(info["states"]["fingertip"][:, None, :])
      data['tg'].append(info["goal"][:, None, :])
      data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
      data['all_actions'].append(action[:, None, :])
      data['all_force'].append(info['states']['muscle'][:, 6, None, :])

  # Concatenate the lists
  for key in data:
      data[key] = th.cat(data[key], axis=1)

  if detach:
      # Detach tensors if needed
      for key in data:
          data[key] = th.detach(data[key])

  return data


def test(cfg_file, weight_file, ff_coefficient=None, test_mov_dur=0.500):

    device = th.device("cpu")

    # load configuration
    cfg = json.load(open(cfg_file, 'r'))

    if ff_coefficient is None:
        ff_coefficient=cfg['ff_coefficient']

    # environment
    name = cfg['name']
    # effector
    muscle_name = cfg['effector']['muscle']['name']
    timestep = cfg['effector']['dt']
    muscle = getattr(mn.muscle,muscle_name)()
    effector = mn.effector.RigidTendonArm26(muscle=muscle,timestep=timestep) 
    # delay
    proprioception_delay = cfg['proprioception_delay']*cfg['dt']
    vision_delay = cfg['vision_delay']*cfg['dt']
    # noise
    action_noise = cfg['action_noise'][0]
    proprioception_noise = cfg['proprioception_noise'][0]
    vision_noise = cfg['vision_noise'][0]
    # initialize environment
    max_ep_duration = cfg['max_ep_duration']
    env = CentreOutFFMinJerk(effector=effector,max_ep_duration=max_ep_duration,name=name,
               action_noise=action_noise,proprioception_noise=proprioception_noise,
               vision_noise=vision_noise,proprioception_delay=proprioception_delay,
               vision_delay=vision_delay)

    # network
    w = th.load(weight_file)
    num_hidden = int(w['gru.weight_ih_l0'].shape[0]/3)
    if 'h0' in w.keys():
        policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=True)
    else:
        policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=False)
    policy.load_state_dict(w)

    # Run episode
    data = run_episode(env, policy, 8, 0, 'test', ff_coefficient=ff_coefficient, test_mov_dur=test_mov_dur, detach=True)
    
    return data


def cal_loss(data, max_iso_force, dt, policy, test=False):

  # calculate losses

  # Jon's proposed loss function
  position_loss = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1))
  muscle_loss = th.mean(th.sum(data['all_force'], dim=-1))
  m_diff_loss = th.mean(th.sum(th.square(th.diff(data['all_force'], 1, dim=1)), dim=-1))
  hidden_loss = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
  diff_loss =  th.mean(th.sum(th.square(th.diff(data['all_hidden'], 1, dim=1)), dim=-1))

#  loss = position_loss + 1e-4*muscle_loss + 5e-5*hidden_loss + 3e-2*diff_loss + 1e-4*m_diff_loss
  loss = 2*position_loss + 1e-4*muscle_loss + 5e-5*hidden_loss + 3e-2*diff_loss + 1e-3*m_diff_loss
  
  angle_loss = None
  lateral_loss = None
  angle_loss = np.mean(calculate_angles_between_vectors(data['vel'].detach(), data['tg'].detach(), data['xy'].detach()))
  lateral_loss, _, _, _ = calculate_lateral_deviation(data['xy'].detach(), data['tg'].detach())
  lateral_loss = np.mean(lateral_loss)

  return loss, position_loss, muscle_loss, hidden_loss, angle_loss, lateral_loss


def calculate_angles_between_vectors(vel, tg, xy):
    """
    Calculate angles between vectors X2 and X3.

    Parameters:
    - vel (numpy.ndarray): Velocity array.
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - angles (numpy.ndarray): An array of angles in degrees between vectors X2 and X3.
    """

    tg = np.array(tg)
    xy = np.array(xy)
    vel = np.array(vel)
    
    # Compute the magnitude of velocity and find the index to the maximum velocity
    vel_norm = np.linalg.norm(vel, axis=-1)
    idx = np.argmax(vel_norm, axis=1)

    # Calculate vectors X2 and X3
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]
    X3 = xy[np.arange(xy.shape[0]), idx, :]

    X2 = X2 - X1
    X3 = X3 - X1
    
    cross_product = np.cross(X3, X2)
    # Calculate the sign of the angle
    sign = np.sign(cross_product)

    # Calculate the angles in degrees
    angles = sign*np.degrees(np.arccos(np.sum(X2 * X3, axis=1) / (1e-8+np.linalg.norm(X2, axis=1) * np.linalg.norm(X3, axis=1))))

    return angles


def calculate_lateral_deviation(xy, tg, vel=None):
    """
    Calculate the lateral deviation of trajectory xy from the line connecting X1 and X2.

    Parameters:
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - deviation (numpy.ndarray): An array of lateral deviations.
    """
    tg = np.array(tg)
    xy = np.array(xy)

    # Calculate vectors X2 and X1
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]

    # Calculate the vector representing the line connecting X1 to X2
    line_vector = X2 - X1
    line_vector2 = np.tile(line_vector[:,None,:],(1,xy.shape[1],1))

    # Calculate the vector representing the difference between xy and X1
    trajectory_vector = xy - X1[:,None,:]

    projection = np.sum(line_vector2 * trajectory_vector, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
    projection = line_vector2 * projection[:,:,np.newaxis]

    lateral_dev = np.linalg.norm(trajectory_vector - projection,axis=2)

    idx = np.argmax(lateral_dev,axis=1)

    max_laterl_dev = lateral_dev[np.arange(idx.shape[0]), idx]

    init = projection[np.arange(idx.shape[0]),idx,:]
    init = init+X1

    endp = xy[np.arange(idx.shape[0]),idx,:]


    cross_product = np.cross(endp-X1, X2-X1)
    # Calculate the sign of the angle
    sign = np.sign(cross_product)


    opt={'lateral_dev':np.mean(lateral_dev,axis=-1),
         'max_lateral_dev':max_laterl_dev,
         'lateral_vel':None}
    # speed 
    if vel is not None:
        vel = np.array(vel)
        projection = np.sum(line_vector2 * vel, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
        projection = line_vector2 * projection[:,:,np.newaxis]
        lateral_vel = np.linalg.norm(vel - projection,axis=2)
        opt['lateral_vel'] = np.mean(lateral_vel,axis=-1)

    return sign*max_laterl_dev, init, endp, opt


def save_model(env, policy, losses, model_name, quiet=False):
    weight_file = model_name + '_weights'
    log_file    = model_name + '_log.json'
    cfg_file    = model_name + '_cfg.json'

    # save model weights
    th.save(policy.state_dict(), weight_file)

    # save training history (log)
    with open(log_file, 'w') as file:
        json.dump({'losses':losses}, file)

    # save environment configuration dictionary
    cfg = env.get_save_config()
    with open(cfg_file, 'w') as file:
        json.dump(cfg, file)

    if (quiet==False):
        print(f"saved {weight_file}")
        print(f"saved {log_file}")
        print(f"saved {cfg_file}")

