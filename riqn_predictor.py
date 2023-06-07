import os

import numpy as np
import torch
from .autoregressive_pybullet import (
    AutoregressiveRecurrentIQN_v2,
    construct_batch_data,
    data_splitting,
    epsilon_decay,
    measure_as,
    ss_evaluate_model,
    ss_learn_model,
    states_min_max_finder,
)


class RIQN_Predictor(AutoregressiveRecurrentIQN_v2):
    def __init__(
        self,
        input_features,
        gru_units=64,
        quantile_embedding_dim=128,
        num_quantile_sample=64,
        device=torch.device("cuda"),
        lr=0.001,
        num_tau_sample=1,
    ):
        super().__init__(input_features, gru_units, quantile_embedding_dim, num_quantile_sample, device)
        self.num_tau_sample = num_tau_sample
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)
        self.constructor_kwargs = {
            "input_features": input_features,
            "gru_units": gru_units,
            "quantile_embedding_dim": quantile_embedding_dim,
            "num_quantile_sample": num_quantile_sample,
            "lr": lr,
            "num_tau_sample": num_tau_sample,
        }

    def fit(self, train_ep_obs, *args, **kwargs):
        checkpoint_path = os.path.join("./tmp", "riqn_chk.pt")
        X_train = self.prepare_data(observations=train_ep_obs)
        self._fit(X_train, checkpoint_path=checkpoint_path, *args, **kwargs)

    def _fit(self, X_train, checkpoint_path, epochs=1_000, batch_size=128, clip_value=10, test_interval=10):
        self.train()
        states_min, states_max = states_min_max_finder(X_train)
        train_rb, test_rb, max_len = data_splitting(X_train, batch_size, states_min, states_max, self.device)
        epsilon = 1
        all_train_losses, all_test_losses = [], []
        best_loss = float("inf")
        for i in range(epochs):
            train_loss = self.train_epoch(train_rb, max_len, epsilon, clip_value=clip_value)
            if i % test_interval == 0:
                all_train_losses.append(train_loss)
                eval_loss, best_loss = self.eval_poch(
                    test_rb, max_len, best_loss=best_loss, epsilon=epsilon, path=checkpoint_path
                )
                all_test_losses.append(eval_loss)
                print(
                    f"ep: {i}/{epochs},  train_loss: {train_loss}, eval_loss: {eval_loss}, best_loss: {best_loss}",
                    end="\r",
                    flush=True,
                )
                # plot_losses(all_train_losses, all_test_losses, env_dir, memory=True, scheduled_sampling=True)
            epsilon = epsilon_decay(epsilon, epochs, i)
        self.eval()

    def train_epoch(self, train_rb, max_len, epsilon, clip_value):
        total_loss = ss_learn_model(
            model=self,
            optimizer=self.optimizer,
            memory=train_rb,
            max_len=max_len,
            epsilon=epsilon,
            clip_value=clip_value,
            num_tau_sample=self.num_tau_sample,
            gru_size=self.gru_size,
            feature_len=self.feature_len,
            device=self.device,
            has_memory=True,
        )
        return total_loss

    def eval_poch(self, test_rb, max_len, epsilon, best_loss, path):
        eval_loss, best_loss = ss_evaluate_model(
            model=self,
            memory=test_rb,
            max_len=max_len,
            best_total_loss=best_loss,
            path=path,
            epsilon=epsilon,
            num_tau_sample=self.num_tau_sample,
            gru_size=self.gru_size,
            feature_len=self.feature_len,
            device=self.device,
            has_memory=True,
        )
        return eval_loss, best_loss

    def feed_forward(self, hx, states, batch_size, sampling_size, tree_root=False):
        states = states.reshape(states.shape[0], 1, -1)
        if tree_root:
            tau = torch.Tensor(np.random.rand(batch_size * sampling_size, 1))
            if hx is not None:
                z, hx = self.forward(states, hx, tau, sampling_size)
            else:
                z = self.forward(states, tau, sampling_size)
        else:
            tau = torch.Tensor(np.random.rand(batch_size * self.num_tau_sample, 1))
            if hx is not None:
                z, hx = self.forward(states, hx, tau, self.num_tau_sample)
            else:
                z = self.forward(states, tau, self.num_tau_sample)
        return z, hx

    def predict_episode(self, episode_obs, sampling_size=8, horizon=1, normalzie=True):
        if episode_obs.ndim != 3:
            episode_obs = np.expand_dims(episode_obs, 1)

        self.eval()
        estimated_dists = []
        anomaly_scores = []
        h_memory = torch.zeros(len(episode_obs[0]) * sampling_size, self.gru_size)
        for i in range(len(episode_obs) - horizon):
            state = episode_obs[i][:, : self.feature_len]
            state = torch.Tensor(state)
            value_return, h_memory = self.feed_forward(
                h_memory.detach().to(self.device), state, len(state), sampling_size, tree_root=True
            )
            unaffected_h_memory = h_memory
            for j in range(1, horizon):
                tmp_h_memory = []
                tmp_value_return = []
                value_return_t = value_return
                h_memory_t = h_memory
                for sample in range(sampling_size):
                    value_return, h_memory = self.feed_forward(
                        h_memory_t[sample, :].detach().reshape(1, -1),
                        value_return_t[:, :, sample],
                        len(value_return_t),
                        sampling_size,
                        tree_root=False,
                    )
                    tmp_h_memory.append(h_memory)
                    tmp_value_return.append(value_return)
                h_memory = torch.stack(tmp_h_memory).squeeze(1)
                value_return = torch.stack(tmp_value_return).squeeze(1).reshape(1, self.feature_len, -1)
            h_memory = unaffected_h_memory
            estimated_dists.append(value_return.squeeze(0).detach().cpu().numpy())
            anomaly_score = measure_as(
                value_return.squeeze(0).detach().cpu().numpy(),
                episode_obs[i + horizon][:, : self.feature_len].squeeze(0),
                self.feature_len,
            )
            anomaly_scores.append(anomaly_score)

        mean_anomaly_scores = np.array(anomaly_scores).mean(axis=1)
        return mean_anomaly_scores

    def prepare_data(self, observations, has_time_feature=False):
        tensor_observations = construct_batch_data(
            observations.shape[-1], observations, device=self.device, has_time_feature=has_time_feature
        )
        states_min, states_max = states_min_max_finder(tensor_observations)
        ep_lengths = [len(ep) for ep in observations]
        self.states_min = states_min
        self.states_max = states_max
        self.ep_length_min = min(ep_lengths)
        self.ep_length_max = max(ep_lengths)
        return tensor_observations

    def save(self, path):
        print("Saving model at: ", path)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "constructor_kwargs": self.constructor_kwargs,
                "attributes": {},
            },
            f=path,
        )

    @classmethod
    def load(cls, path, device, **model_kwargs):
        saved_variables = torch.load(path)
        model = cls(**saved_variables["constructor_kwargs"], device=device)
        model.load_state_dict(saved_variables["state_dict"])
        for k, v in saved_variables["attributes"].items():
            model.__setattr__(k, v)
        model.to(device)
        model.eval()
        return model
