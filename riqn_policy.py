import torch
import models 


class RIQN_Policy():
    def __init__(self, env, device):
        num_features = env.observation_space.shape[0]
        self.env = env
        self.model = models.IQN(num_inputs=num_features, 
                                num_outputs=env.action_space.n,
                                quantile_embedding_dim=64,
                                num_quantile_sample=32,
                                device=device,
                                env_name=env.spec.id
                                )

    def predict(self, state, *args, **kwargs):
        action, _ = models.get_action(torch.Tensor(state).unsqueeze(0), self.model, -1, self.env, 64)
        return action, _

    def do_rollout(self,env):
        raise NotImplementedError
    
    def learn():
        raise NotImplementedError

    @classmethod
    def load(cls, path, env, device):
        policy = cls(env, device)
        policy.model.load_state_dict(torch.load(path, device))
        return policy