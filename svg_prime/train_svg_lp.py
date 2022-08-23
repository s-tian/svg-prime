import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.utils.data import DataLoader
import utils
import progressbar
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--loss_fn', default='mse', help='loss function to use (mse | l1)')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--dataset_files', nargs='+', required=False, help='dataset files for perceptual_metrics dataset')
parser.add_argument('--camera_name', type=str, default="agentview_shift_2", help='camera name for robomimic')
parser.add_argument('--cache_mode', type=str, default="low_dim", help='perceptual_metrics dataset cache mode')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=512, help='dimensionality of hidden layer')
parser.add_argument('--M', type=int, default=1, help='scaling factor for LSTMs')
parser.add_argument('--K', type=int, default=1, help='scaling factor for encoder/decoder')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--a_dim', type=int, default=4, help='dimensionality of action vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='shallow_vgg', help='model type (dcgan | vgg | shallow_vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--wandb_project_name', type=str, default="svg_prime", help='project name to use when logging with wandb')
parser.add_argument('--disable_wandb', action='store_true', help='do not log to wandb')


opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)


import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim+opt.a_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
if opt.loss_fn == "mse":
    reconstruction_criterion = nn.MSELoss()
elif opt.loss_fn == "l1":
    reconstruction_criterion = nn.L1Loss()
else:
    raise NotImplementedError("Unknown loss function: %s" % opt.loss_fn)

def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# --------- load a dataset ------------------------------------
def prep_data(data_dict):
    video = data_dict["video"].permute(1, 0, 2, 3, 4).cuda()
    actions = data_dict["actions"].permute(1, 0, 2).cuda()
    return video, actions

if opt.dataset == "perceptual_metrics":
    from fitvid.data.robomimic_data import load_dataset_robomimic_torch
    train_loader = load_dataset_robomimic_torch(
        dataset_path=opt.dataset_files,
        batch_size=opt.batch_size,
        video_len=opt.n_past + opt.n_future,
        video_dims=(64, 64),
        phase="train",
        depth=False,
        normal=False,
        view=opt.camera_name,
        cache_mode=opt.cache_mode,
        seg=False,
    )
    test_loader= load_dataset_robomimic_torch(
        dataset_path=opt.dataset_files,
        batch_size=opt.batch_size,
        video_len=opt.n_past + opt.n_future,
        video_dims=(64, 64),
        phase="valid",
        depth=False,
        normal=False,
        view=opt.camera_name,
        cache_mode=opt.cache_mode,
        seg=False,
    )

else:
    train_data, test_data = utils.load_dataset(opt)

    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)

    prep_data = lambda x: x


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------
def plot(x, epoch, actions=None):
    nsample = 20
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0]
                z_t, _, _ = posterior(h_target)
                prior(h)
                if actions is not None:
                    tiled_action = actions[i-1]
                    frame_predictor(torch.cat([h, tiled_action, z_t], 1))
                else:
                    frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                if actions is not None:
                    tiled_action = actions[i-1]
                    h = frame_predictor(torch.cat([h, tiled_action, z_t], 1))
                else:
                    h = frame_predictor(torch.cat([h, z_t], 1))
                x_in = decoder([h, skip])
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    eval_mse = []
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            sample_eval_mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
                sample_eval_mse += reconstruction_criterion(gt_seq[t][i].data.cpu(), gen_seq[s][t][i].data.cpu())
            if mse < min_mse:
                min_mse = mse
                min_idx = s
        eval_mse.append(sample_eval_mse / opt.n_eval)
        s_list = [min_idx,
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)
    wandb.log({'eval/mse': np.mean(eval_mse)})
    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)
    log_wandb_gif(gifs, prefix="eval")


def recursive_stack(list_of_tensors):
    if torch.is_tensor(list_of_tensors):
        return list_of_tensors
    else:
        return torch.stack([recursive_stack(t) for t in list_of_tensors], 0)


def log_wandb_gif(gif, prefix):
    # Convert list of list of list of tensors into one tensor
    gif = recursive_stack(gif).detach().cpu()
    # Take ground truth and a single random sample to log
    gif = torch.cat((gif[:, :, 0], gif[:, :, -1]), dim=-1)
    gif = torch.cat(torch.unbind(gif, dim=1), dim=-2)
    gif = (gif.numpy() * 255).astype(np.uint8)
    wandb.log({f"{prefix}/video": wandb.Video(gif)})


def plot_rec(x, epoch, actions=None):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_past + opt.n_future):
        h = encoder(x[i - 1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h
        h_target = h_target
        z_t, _, _ = posterior(h_target)
        if i < opt.n_past:
            if actions is not None:
                tiled_action = actions[i-1]
                frame_predictor(torch.cat([h, tiled_action, z_t], 1))
            else:
                frame_predictor(torch.cat([h, z_t], 1))
            gen_seq.append(x[i])
        else:
            if actions is not None:
                tiled_action = actions[i-1]
                h_pred = frame_predictor(torch.cat([h, tiled_action, z_t], 1))
            else:
                h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip])
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past + opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x, actions=None, log_gif=False):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    mse = 0
    kld = 0

    if log_gif:
        final_gif = []

    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h = h[0]
        z_t, mu, logvar = posterior(h_target)
        _, mu_p, logvar_p = prior(h)
        if actions is not None:
            tiled_action = actions[i-1]
            h_pred = frame_predictor(torch.cat([h, tiled_action, z_t], 1))
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])
        if log_gif:
            final_gif.append(x_pred)
        mse += reconstruction_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

    loss = mse + kld*opt.beta
    loss.backward()
    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    if log_gif:
        final_gif = torch.stack(final_gif, dim=0).detach().cpu()
        gif_to_log = torch.stack((x[1:].detach().cpu(), final_gif), axis=2)
        gif_to_log = gif_to_log[:, :10] # Log at most 10 different batch items
        log_wandb_gif(gif_to_log, prefix="train")

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future-1), kld.data.cpu().numpy()/(opt.n_future+opt.n_past-1)


# Initialize wandb
if not opt.disable_wandb:
    import wandb
    wandb.init(
        project=opt.wandb_project_name,
        reinit=True,
        mode="online",
        settings=wandb.Settings(start_method="fork"),
    )

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i + 1)
        if opt.a_dim == 0:
            x = prep_data(next(training_batch_generator))
            actions = None
        else:
            x, actions = prep_data(next(training_batch_generator))
        # train frame_predictor
        mse, kld = train(x, actions, log_gif=(i == opt.epoch_size - 1))
        epoch_mse += mse
        epoch_kld += kld
        if not opt.disable_wandb:
            wandb.log({
                "train/mse": mse,
                "train/kld": kld,
            })

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (
        epoch, epoch_mse / opt.epoch_size, epoch_kld / opt.epoch_size, epoch * opt.epoch_size * opt.batch_size))

    # plot some stuff
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()

    if opt.a_dim == 0:
        x = prep_data(next(testing_batch_generator))
        actions = None
    else:
        x, actions = prep_data(next(testing_batch_generator))

    with torch.no_grad():
        plot(x, epoch, actions=actions)
        plot_rec(x, epoch, actions=actions)

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)

