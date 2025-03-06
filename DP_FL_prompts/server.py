import json
import logging
import os
import sys
import time
from datetime import datetime
import shutil
import datasets
import numpy as np
import torch
import transformers
from transformers import set_seed
import pandas as pd
from arguments import get_args
from private_transformers.privacy_engine import PrivacyEngine
from tasks.utils import *
import copy
import warnings
from privacy_utils.rdp_accountant import get_noise_multiplier, compute_rdp, get_privacy_spent
import utils
from helper_secagg import (mod_vector_m, get_actual_sentivity, postprocessing_client_gradient, \
    postprocessing_server_gradient, add_two_model_states, secure_aggregate_gradients, get_model_update,
                           get_trainable_param_size, get_seed_pair_hadamard)
from dp_discrete_gaussian_helpers import ddgauss_epsilon
import math


from per_example_grad_helpers import calc_sample_norms, calc_clipping_factors
from dp_discrete_gaussian_helpers import ddgauss_params
from discrete_noise_sampler import sample_discrete_gaussian


os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)



def get_clipped_gradients(trainer, args):
    norms = calc_sample_norms(trainer.named_params_grad_sample)
    clipping_factors = calc_clipping_factors(norms, flat_value=args.per_sample_max_grad_norm)
    for name, param in trainer.named_params_grad_sample.items():
        trainer.named_params_grad_sample[name] = clipping_factors.view(len(clipping_factors), *([1]*(param.ndim-1))) * param



def compute_epsilon(steps, sampling_probability, noise_multiplier, delta):
  """Computes epsilon value for given hyperparameters."""
  if noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp(
      q=sampling_probability,
      noise_multiplier=noise_multiplier,
      steps=int(steps),
      orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)

def get_checkpoint(resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    return checkpoint


def show_train_result(train_result):
    key = "metrics"
    if hasattr(trainer, key):
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    key = 'save_state'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        trainer.save_state()

    key = 'log_best_metrics'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        trainer.log_best_metrics()

    key = 'get_prv_epsilon'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        eps_prv = trainer.get_prv_epsilon()
    else:
        eps_prv = 0

    key = 'get_rdp_epsilon'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        eps_rdp = trainer.get_rdp_epsilon()
    else:
        eps_rdp = 0

    key = 'log'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })


def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = get_checkpoint(resume_from_checkpoint=resume_from_checkpoint, last_checkpoint=last_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    show_train_result(train_result=train_result)


def set_private_transformer_optimizer(trainer, resume_from_checkpoint=None, last_checkpoint=None, client_id=None):
    # checkpoint = get_checkpoint(resume_from_checkpoint=resume_from_checkpoint, last_checkpoint=last_checkpoint)

    # params that require the grad
    named_params = [(name, param) for name, param in trainer.model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.weight']
    # for name, param in named_params:
    #     print(name, param.shape)
    # prompt roberta-base: classifier weight (2, 768), classifier bias (2,), prefix_encoder.weight (10, 768)
    # prompt-infill roberta-base: lm_head: a lot, prefix_encoder.weight (pre_seq_length, 768)
    # prefix roberta-base: classifier weight (2, 768), classifier bias (2,), prefix_encoder.weight (10, 18432)

    # prompt roberta-large: classifier weight (2, 1024), classifier bias (2,), prefix_encoder.weight (10, 1024)
    # prefix roberta-large: classifier weight (2, 1024), classifier bias (2,), prefix_encoder.weight (10, 49152)

    # for aggregate
    if client_id is None:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
             'weight_decay': training_args.weight_decay},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = trainer.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )

        if training_args.lr_decay == 'yes':
            print('Apply default linear decay.')
            training_setup = trainer.get_training_setup()
            t_total = training_setup["t_total"]
            # `trainer.optimizer` is not None here, so no optimizer is created.
            trainer.create_optimizer_and_scheduler(num_training_steps=t_total)
        else:
            trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
    # for individual client
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = trainer.optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=client_training_args[client_id].learning_rate, momentum=0.0)
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)


    if client_id is not None:
        assert client_privacy_args[client_id].noise_multiplier == 0.0 # making sure we only use client's privacy engine for clipping
        client_train_batch_size = client_training_args[client_id].gradient_accumulation_steps * client_training_args[client_id].per_device_train_batch_size
        privacy_engine = PrivacyEngine(
            module=trainer.model,
            batch_size=min(client_train_batch_size, len(trainer.train_dataset)),
            sample_size=len(trainer.train_dataset),
            epochs=client_training_args[client_id].num_train_epochs,
            max_grad_norm=client_privacy_args[client_id].per_sample_max_grad_norm,
            noise_multiplier=client_privacy_args[client_id].noise_multiplier,
            # target_epsilon=privacy_args.target_epsilon, #we use the privacy engine to clip, not adding noise
            # target_delta=privacy_args.target_delta,
            # accounting_mode=privacy_args.accounting_mode,
            clipping_mode=client_privacy_args[client_id].clipping_mode,
            skip_checks=True,
        )
        # Originally, it could have been null.
        # privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        # privacy_args.target_delta = privacy_engine.target_delta


        print('privacy_engine.noise_multiplier: ', privacy_engine.noise_multiplier)
        print('privacy_engine.target_delta: ', privacy_engine.target_delta)

        print('privacy_args: ')
        print(json.dumps(client_privacy_args[client_id].__dict__, indent=4))
        privacy_engine.attach(optimizer)

    # Training

    # trainer.save_model()

    # show_train_result(train_result=train_result)


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    # Check trainer prediction step - no labels.
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return metrics


def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):

        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


def write_metrics(metrics, elapsed_time=0):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H:%M:%S")

    res_dir = f"tune_params_lr_grad_epochs_{data_args.dataset_name}.csv"
    if os.path.exists(res_dir):
        header = False
    else:
        header = True

    df = pd.DataFrame({"training_type": training_args.training_type,
                       "privacy_engine": training_args.privacy_engine,
                       "method_type": model_args.method_type,
                       "dataset_name": data_args.dataset_name,
                       "model_name_or_path": model_args.model_name_or_path,
                       "per_device_train_batch_size": training_args.per_device_train_batch_size,
                       "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                       "num_clients": collaboration_args.num_clients,
                       "sample_clients_ratio": collaboration_args.sample_clients_ratio,
                       "learning_rate": training_args.learning_rate,
                       "target_epsilon": privacy_args.target_epsilon,
                       "target_delta": privacy_args.target_delta,
                       "per_sample_max_grad_norm": privacy_args.per_sample_max_grad_norm,
                       "num_global_server_epochs": collaboration_args.num_global_server_epochs,
                       "pre_seq_len": model_args.pre_seq_len,
                       "eval_loss": metrics['eval_loss'],
                       "eval_accuracy": metrics['eval_accuracy'],
                       "elapsed_time": elapsed_time,
                       "elapsed_time_hour": elapsed_time/3600,
                       "current_time": current_time,
                       "seed": training_args.seed,
                       "max_seq_length": data_args.max_seq_length,
                       "freeze_non_prompt_layers": model_args.freeze_non_prompt_layers,
                       "shift_mask_pos_by_p_len": model_args.shift_mask_pos_by_p_len,
                       "use_discrete": privacy_args.use_discrete}, index=[0])
    df.to_csv(res_dir, mode='a', index=False, header=header)
    print(data_args.dataset_name)
    print('metrics file: ', res_dir)



def set_mapping_for_infilling(data_args):
    # from: https://arxiv.org/abs/2110.05679
    # TODO: Hacky mapping creation. Refactor this in the future.
    #  Currently gets replace if mapping_id and mapping_path is set.
    if data_args.dataset_name == "sst2":
        data_args.mapping = "{'negative':'terrible','positive':'great'}"
    elif data_args.dataset_name == "mnli":
        data_args.mapping = "{'contradiction': 'no', 'entailment': 'yes', 'neutral': 'maybe'}"
    elif data_args.dataset_name == "qnli":
        data_args.mapping = "{'not_entailment': 'no', 'entailment': 'yes'}"
    elif data_args.dataset_name == "qqp":
        data_args.mapping = "{'duplicate': 'yes', 'not_duplicate': 'no'}"  # 1 -- equivalent, 0 -- not equivalent.
    else:
        raise ValueError(f"Unknown task: {data_args.task_name}")


def set_template_for_infilling(data_args):
    if data_args.template is not None:
        print(f"use input template {data_args.template}")
        return
    else:
        if data_args.dataset_name == "sst2":
            data_args.template = "*cls**sent_0*_It_was*mask*.*sep+*"
        elif data_args.dataset_name == "mnli":
            data_args.template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"
        elif data_args.dataset_name == "qnli":
            data_args.template = "*cls**sent-_0**mask*,*+sentl_1**sep+*"
        elif data_args.dataset_name == "qqp":
            data_args.template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"  # 1 -- equivalent, 0 -- not equivalent.
        else:
            raise ValueError(f"Unknown task: {data_args.task_name}")


def save_model_state(args, aggregate_trainer, global_step, rewrite=True, trainable_only=True):
    _, _, _, _, _, _, collaboration_args, _, _ = args
    if not rewrite:
        if os.path.exists(f"{collaboration_args.global_output_dir}/aggregate_{global_step}.pt"):
            print(f"{collaboration_args.global_output_dir}/aggregate_{global_step}.pt already exists. skip saving")
            return

    if trainable_only:
        init_state = copy.deepcopy(aggregate_trainer.model.state_dict())
        trainable_params = [name for name, param in aggregate_trainer.model.named_parameters() if param.requires_grad]

        net_state = {}
        for param_name in trainable_params:
            net_state[param_name] = init_state[param_name]

    else:
        net_state = aggregate_trainer.model.state_dict()

    torch.save({'net': net_state,
                'optimizer': aggregate_trainer.optimizer.state_dict(),
                'scheduler': aggregate_trainer.lr_scheduler.state_dict()},
               f"{collaboration_args.global_output_dir}/temp_aggregate.pt")
    shutil.move(f"{collaboration_args.global_output_dir}/temp_aggregate.pt",
                f"{collaboration_args.global_output_dir}/aggregate_{global_step}.pt")
    print(f"saving {collaboration_args.global_output_dir}/aggregate_{global_step}.pt")


def load_model_state(args, aggregate_trainer, global_step):
    model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args, client_training_args, client_privacy_args = args
    state = torch.load(f"{collaboration_args.global_output_dir}/aggregate_{global_step}.pt", map_location=training_args.device)
    net_state = copy.deepcopy(aggregate_trainer.model.state_dict())

    net_state_loaded = state['net']
    for param_name in net_state_loaded:
        net_state[param_name] = net_state_loaded[param_name]

    aggregate_trainer.model.load_state_dict(net_state)
    aggregate_trainer.optimizer.load_state_dict(state['optimizer'])
    aggregate_trainer.lr_scheduler.load_state_dict(state['scheduler'])


def split_train_dataset(args, dataset=None):
    model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args, client_training_args, client_privacy_args = args
    if len(collaboration_args.client_data_split_ratio) == 1 and collaboration_args.client_data_split_ratio[0] == 1/collaboration_args.num_clients:
        len_train = len(dataset.train_dataset)
        len_client_train = len_train // collaboration_args.num_clients
        sequence = np.random.choice(len_train, size=len_train, replace=False)
        ind = [(j + 1) * len_client_train for j in range(len(sequence) // len_client_train)]
        sequence = np.split(sequence, ind)
        sequence = sequence[:collaboration_args.num_clients]
        if not os.path.exists(f"{collaboration_args.global_output_dir}/client_data_splitting.npy"):
            np.save(f"{collaboration_args.global_output_dir}/client_data_splitting.npy", sequence, allow_pickle=True)
        else:
            sequence = np.load(f"{collaboration_args.global_output_dir}/client_data_splitting.npy", allow_pickle=True)
        return sequence
    else:
        raise NotImplementedError(f"{collaboration_args.client_data_split_ratio} for num_clients {collaboration_args.num_clients} (split aggregate dataset) is not implemented")


def create_client_gradient_accumulation_steps(client_training_args, client_batch_size, client_id):
    print(f"client_batch_size:{client_batch_size}")
    print(f"client_training_args[client_id].gradient_accumulation_steps: {client_training_args[client_id].gradient_accumulation_steps}")
    assert client_batch_size % client_training_args[client_id].gradient_accumulation_steps == 0
    # need to ensure client_batch_size is divisible by their gradient_accumulation_steps. otherwise the steps doesn't match
    client_training_args[client_id].per_device_train_batch_size = client_batch_size // client_training_args[client_id].gradient_accumulation_steps
    client_batch_size = client_training_args[client_id].per_device_train_batch_size * client_training_args[client_id].gradient_accumulation_steps
    return client_batch_size

def create_client_batches(args, how="same_sampling_rate", sequence_all=None, replace=False, aggregate_trainer=None):
    """
    this method creates list of mini-batches for each client and set the client's training_args and privacy_args
    # Individual clients:
    # - privacy engine: clipping_norm is set, but noise_multiplier must be 0
    # - learning rate = 1.0 —> override by us into client_training_args[client_id]
               - this way the update after one step: (state_after - state_init) * client batch_size would be the sum of all gradient
    # - lr_scheduler = None
    # - optimizer = SGD without any momentum or decay
    """
    model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args, client_training_args, client_privacy_args = args
    for client_id in range(collaboration_args.num_clients):
        client_training_args[client_id].output_dir = os.path.join(collaboration_args.global_output_dir, f"client_{client_id}")
        client_training_args[client_id].learning_rate = 1.0
        client_training_args[client_id].weight_decay = 0.0
        client_training_args[client_id].lr_decay = 'no'
        client_training_args[client_id].num_train_epochs = 1
        client_privacy_args[client_id].noise_multiplier = 0.0
        client_privacy_args[client_id].target_epsilon = np.infty

    # calculate sampling rate here.
    batch_size_aggregate = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
    collaboration_args.sampling_rate = batch_size_aggregate / len(np.concatenate((sequence_all)))
    sampling_rate = collaboration_args.sampling_rate

    # Set up global steps from global epochs
    collaboration_args.num_global_server_steps = int(collaboration_args.num_global_server_epochs / sampling_rate)
    print(f"batch_size_aggregate: {batch_size_aggregate}")
    print(f"sampling rate: {sampling_rate}" )
    print(f"collaboration_args.num_global_server_steps: {collaboration_args.num_global_server_steps}")
    resultant_sequence = []

    # start splitting
    if how == 'same_sampling_rate':
        batch_size_total = 0
        list_sampling_rates = []
        for client_id, seq_client in enumerate(sequence_all):
            len_client_train = len(seq_client)
            batch_size = int(np.round(sampling_rate * len_client_train))
            batch_size = create_client_gradient_accumulation_steps(client_training_args, batch_size, client_id)
            list_sampling_rates.append(batch_size/len_client_train)
            batch_size_total += batch_size

            epochs = collaboration_args.num_global_server_epochs
            sequence = np.concatenate([np.random.choice(len_client_train, size=len_client_train, replace=replace)
                                       for i in range(epochs)])

            ind = [(j + 1) * batch_size for j in range(len(sequence) // batch_size)]
            sequence = np.split(sequence, ind)[:collaboration_args.num_global_server_steps]
            if len(sequence) < collaboration_args.num_global_server_steps:
                raise ValueError("number of minibatches is less than num_global_server_steps")
            assert len(sequence[-1]) == batch_size
            resultant_sequence.append(sequence)
        aggregate_trainer.gradient_accumulation_steps = 1
        aggregate_trainer.per_device_train_batch_size = batch_size_total
    else:
        raise NotImplementedError(f"how={how} not implemented")
    # exit()
    return resultant_sequence


def set_trainer_data_and_starting_state(trainer, dataset, training_indices, aggregate_trainer):
    init_state = copy.deepcopy(aggregate_trainer.model.state_dict())
    trainer.model.load_state_dict(init_state)
    trainer.train_dataset = dataset.train_dataset.select(training_indices)
    trainer.named_params_grad_sample = None



def get_noise(noise_multiplier, sensitivity, device, size, numel, dtype, args, discretization_granularity):
    if args.use_discrete:
        if args.use_our_approximated_discrete_gaussian:
            # noise_multiplier/discretization_granularity
            print("Please use use_our_approximated_discrete_gaussian=0")
            exit()
        else:
            print(f"noise size: {numel}")
            noise = sample_discrete_gaussian(math.ceil(noise_multiplier/discretization_granularity), size, torch.int64, device)
    else:
        noise = torch.normal(mean=0, std=noise_multiplier*sensitivity, size=size, device=device, dtype=dtype)
    return noise


def aggregate_model_states(args, aggregate_trainer, client_trainers,):
    model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args, client_training_args, client_privacy_args = args

    # params that require the grad
    aggregate_trainer.optimizer.zero_grad()
    for name, param in aggregate_trainer.model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param))

    init_state = copy.deepcopy(aggregate_trainer.model.state_dict())
    trainable_params = [name for name, param in aggregate_trainer.model.named_parameters() if param.requires_grad]

    list_client_gradient_update = {}
    gradient_noise = {}
    template = {}

    for param_name in trainable_params:
        gradient_noise[param_name] = torch.zeros_like(init_state[param_name])
        template[param_name] = (init_state[param_name].numel(), init_state[param_name].shape, init_state[param_name].dtype)

    seed_pair = get_seed_pair_hadamard()

    for client_id, trainer in client_trainers.items():
        get_clipped_gradients(trainer, privacy_args)
        client_gradient_update = trainer.named_params_grad_sample
        # client_gradient_update = get_model_update(trainer, pre_state=init_state, trainable_params=trainable_params, client_training_args_ind=client_training_args[client_id] )
        # divide the gradient by the granularity
        # flatten it
        # rounding it to int
        # mod by 2**B
        if privacy_args.use_discrete:
            client_gradient_update = postprocessing_client_gradient(client_gradient_update, template, seed_pair, privacy_args)
        else:
            for k, v in client_gradient_update.items():
                client_gradient_update[k] = v.sum(0)
        list_client_gradient_update[client_id] = client_gradient_update

    # summation and make it 2**B
    aggregate_gradient = secure_aggregate_gradients(list_client_gradient_update, privacy_args)

    if privacy_args.noise_multiplier > 0 and privacy_args.per_sample_max_grad_norm > 0:

        if isinstance(aggregate_gradient, dict):
            for param_name in trainable_params:
                noise = get_noise(privacy_args.noise_multiplier,
                              get_actual_sentivity(privacy_args),
                              aggregate_gradient[param_name].device,
                              aggregate_gradient[param_name].size(),
                                  aggregate_gradient[param_name].numel(),
                              aggregate_gradient[param_name].dtype,
                              privacy_args, privacy_args.discretization_granularity
                              )
                gradient_noise[param_name] = noise
        else:
            gradient_noise = get_noise(privacy_args.noise_multiplier,
                              get_actual_sentivity(privacy_args),
                              aggregate_gradient.device,
                              aggregate_gradient.size(),
                              aggregate_gradient.numel(),
                              aggregate_gradient.dtype,
                              privacy_args, privacy_args.discretization_granularity
                              )


        # for param_name in trainable_params:
        #     noise = get_noise(privacy_args.noise_multiplier,
        #                       get_actual_sentivity(privacy_args),
        #                       init_state[param_name].device,
        #                       init_state[param_name].size(),
        #                       init_state[param_name].dtype,
        #                       privacy_args, privacy_args.discretization_granularity
        #                       )
        #     gradient_noise[param_name] = noise
            # print(f"{init_state[param_name].dtype}")
            # exit()

    # noise mod 2**B
    if privacy_args.use_discrete:
        gradient_noise = mod_vector_m(gradient_noise, args=privacy_args)

    # add
    aggregate_gradient_plus_noise = add_two_model_states(aggregate_gradient, gradient_noise, param_names=trainable_params)

    if privacy_args.use_discrete:
        aggregate_gradient_plus_noise = mod_vector_m(aggregate_gradient_plus_noise, args=privacy_args) # the server sees this

        # this is what the server do to post-process the DP gradient
        aggregate_gradient_plus_noise = postprocessing_server_gradient(aggregate_gradient_plus_noise, template, seed_pair, args=privacy_args)


    for name, param in aggregate_trainer.model.named_parameters():
        if param.requires_grad:
            param.grad = aggregate_gradient_plus_noise[name].type(torch.float32).to(param.device)

    for name, param in aggregate_trainer.model.named_parameters():
        if param.requires_grad:
            param.grad /= (aggregate_trainer.per_device_train_batch_size * aggregate_trainer.gradient_accumulation_steps)



if __name__ == '__main__':
    print("start server")
    args = get_args()
    print('args: ', args)
    model_args, data_args, training_args, qa_args, privacy_args, aux_args, collaboration_args, client_training_args, client_privacy_args = args
    set_seed(training_args.seed)

    if data_args.infill:
        set_mapping_for_infilling(data_args)
        set_template_for_infilling(data_args)

    training_args.output_dir = collaboration_args.global_output_dir

    if data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer
    else:
        raise NotImplementedError(
            'Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    aggregate_trainer, predict_dataset, dataset_obj_temp, tokenizer_obj = get_trainer(args)
    dataset_obj = copy.deepcopy(dataset_obj_temp)
    set_private_transformer_optimizer(trainer=aggregate_trainer)
    print(f"trainable param size, padded trainable param size: {get_trainable_param_size(aggregate_trainer, privacy_args)}")

    # aggregate_state = copy.deepcopy(aggregate_trainer.model.state_dict())
    if collaboration_args.global_overwrite_output_dir:
        print(f"Overwriting existing output directory {collaboration_args.global_output_dir}")
        shutil.rmtree(collaboration_args.global_output_dir)

    if not os.path.exists(collaboration_args.global_output_dir):
        print(f"Creating new output directory {collaboration_args.global_output_dir}")
        os.makedirs(collaboration_args.global_output_dir)
    # exit()
    save_model_state(args, aggregate_trainer, 0, rewrite=False)

    last_checkpoint = 0
    last_checkpoint = utils.find_latest_checkpoint(collaboration_args.global_output_dir, "aggregate_", ".pt")


    temp_splited_clients_training_idx = split_train_dataset(args, dataset=dataset_obj)
    clients_training_batches = create_client_batches(args, how=collaboration_args.how_client_sample_data,
                                                     sequence_all=temp_splited_clients_training_idx, replace=False,
                                                     aggregate_trainer=aggregate_trainer)
    privacy_args.target_delta = privacy_args.target_delta if privacy_args.target_delta is not None else 1/len(dataset_obj.train_dataset)

    privacy_args.alpha = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    if privacy_args.target_epsilon is not None:
        if privacy_args.use_discrete:
            privacy_args.discretization_granularity, privacy_args.noise_multiplier = ddgauss_params(collaboration_args.sampling_rate, #
                                                                                    privacy_args.target_epsilon,
                                                                                    privacy_args.per_sample_max_grad_norm,
                                                                                    privacy_args.mod_num_bits,
                                                                                    aggregate_trainer.per_device_train_batch_size,
                                                                                    privacy_args.padded_trainable_param_size,
                                                                                    delta=privacy_args.target_delta,
                                                                                    beta=privacy_args.beta,
                                                                                    steps=collaboration_args.num_global_server_steps,
                                                                                    k=privacy_args.k_stddevs
                                                                                    )
            print(f"discritization granularity gamma: {privacy_args.discretization_granularity}")
            print(f"noise_multiplier sigma: {privacy_args.noise_multiplier}")
            print(f"variance for discrete gaussian (sigma/gamma)**2: {(privacy_args.noise_multiplier/privacy_args.discretization_granularity)**2}")
            print(
                f"std for discrete gaussian sigma/gamma: {(privacy_args.noise_multiplier / privacy_args.discretization_granularity)}")
        else:
            aggregate_noise_multiplier = get_noise_multiplier(privacy_args.target_epsilon,
                                                              privacy_args.target_delta,
                                                              collaboration_args.sampling_rate,
                                                              collaboration_args.num_global_server_epochs,
                                                              privacy_args.alpha)
            privacy_args.noise_multiplier = aggregate_noise_multiplier
    else:
        assert privacy_args.noise_multiplier is not None

    print(
        f"aggregate: \n"
        f"target eps: {privacy_args.target_epsilon}, \n"
        f"target delta: {privacy_args.target_delta}, \n"
        f"sampling rate: {collaboration_args.sampling_rate}, \n"
        f"noise multiplier: {privacy_args.noise_multiplier}, \n"
        f"num epoches {collaboration_args.num_global_server_epochs}, \n"
        f"num steps {collaboration_args.num_global_server_steps}")


    client_trainers = {}
    for client_id in range(collaboration_args.num_clients):
        trainer, _, _, _ = get_trainer(args, client_id=client_id,
                                       dataset_obj=copy.deepcopy(dataset_obj),
                                       tokenizer_obj=tokenizer_obj)
        trainer.model = trainer.model.to('cpu')
        set_private_transformer_optimizer(trainer=trainer, client_id=client_id)

        client_trainers[client_id] = trainer

    elapsed_time = 0
    if training_args.do_train:
        start = time.time()
        for global_step in range(last_checkpoint, collaboration_args.num_global_server_steps):
            if global_step < last_checkpoint:
                print(f"skip training from {global_step} steps")
                continue
            if global_step == last_checkpoint:
                load_model_state(args, aggregate_trainer, global_step)

            print(f"training from {global_step} steps")
            aggregate_trainer.optimizer.zero_grad()
            aggregate_trainer.model = aggregate_trainer.model.to("cpu")

            for client_id in range(collaboration_args.num_clients):
                train_indices = clients_training_batches[client_id][global_step]
                trainer = client_trainers[client_id]
                torch.cuda.empty_cache()
                trainer.model = trainer.model.to(client_training_args[client_id].device)
                print(f"client device: {client_training_args[client_id].device}")
                set_trainer_data_and_starting_state(trainer, copy.deepcopy(dataset_obj), train_indices, aggregate_trainer)
                if training_args.training_type == 'private' and collaboration_args.private_type == 'bound_datapoint':
                    if training_args.privacy_engine == 'private_transformers':
                        trainer.named_params_grad_sample = None
                        train_result = trainer.train()

                    else:
                        raise Exception(f"Unsupported privacy engine: {training_args.privacy_engine}.")
                else:
                    train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
                trainer.model = trainer.model.to('cpu')
            stop = time.time()
            elapsed_time = stop - start

            aggregate_trainer.model = aggregate_trainer.model.to("cpu")
            aggregate_model_states(args, aggregate_trainer, client_trainers, )
            aggregate_trainer.model = aggregate_trainer.model.to(training_args.device)

            aggregate_trainer.optimizer.step()
            aggregate_trainer.lr_scheduler.step()

            if (global_step + 1) % collaboration_args.save_freq_aggregate == 0 or (global_step + 1) == collaboration_args.num_global_server_steps:
                save_model_state(args, aggregate_trainer, global_step+1)
                print(f"saving step: {global_step+1}")
                metrics = evaluate(aggregate_trainer)
                if privacy_args.use_discrete:

                    eps, opt_order = ddgauss_epsilon(privacy_args.discretization_granularity, privacy_args.noise_multiplier,
                                          privacy_args.per_sample_max_grad_norm,
                                          privacy_args.beta, privacy_args.padded_trainable_param_size, collaboration_args.sampling_rate,
                                          (global_step + 1),
                                          privacy_args.target_delta)
                    print(f'For delta={privacy_args.target_delta}, sigma={privacy_args.noise_multiplier}, '
                          f'the current epsilon is: {eps}, opt_order: {opt_order}')
                else:
                    eps, _, opt_order = compute_epsilon((global_step + 1), collaboration_args.sampling_rate,
                                                        privacy_args.noise_multiplier, privacy_args.target_delta)
                    print(f'For delta={privacy_args.target_delta}, sigma={privacy_args.noise_multiplier}, '
                          f'the current epsilon is: {eps}, opt_order: {opt_order}')
                # exit()
            aggregate_trainer.model = aggregate_trainer.model.to("cpu")

    load_model_state(args, aggregate_trainer, collaboration_args.num_global_server_steps)
    aggregate_trainer.model = aggregate_trainer.model.to(training_args.device)
    if training_args.do_eval:
        metrics = evaluate(aggregate_trainer)
        write_metrics(metrics=metrics, elapsed_time=elapsed_time)

    if training_args.do_predict:
        predict(aggregate_trainer, predict_dataset)

