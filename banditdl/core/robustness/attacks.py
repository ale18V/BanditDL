"""Byzantine attack implementations for robust distributed learning."""

import torch
# Import utility functions from the core math helpers
from banditdl.utils.math_utils import clip_vector, line_maximize


def signflipping(honest_vectors, **kwargs):
    return torch.stack(honest_vectors).mean(dim=0).mul(-1)


def labelflipping(flipped_vectors, **kwargs):
    return torch.stack(flipped_vectors).mean(dim=0)


def fall_of_empires(honest_vectors, attack_factor=3, negative=False, **kwargs):
    #JS: negative controls the sign of the attack_factor
    #JS: attack_factor controls the magnitude of the attack_vector to be added to avg_honest_vector
    if negative:
        attack_factor = - attack_factor
    return torch.stack(honest_vectors).mean(dim=0).mul(1 - attack_factor)


def auto_FOE(honest_vectors, aggregator, nb_byz, gradient_clip, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_FOE(factor):
        temp_vectors = [vector.clone() for vector in honest_vectors]
        #temp_vectors = copy.deepcopy(honest_vectors)
        byzantine_vector = fall_of_empires(temp_vectors, attack_factor=factor)
        if gradient_clip is not None:
            byzantine_vector = clip_vector(byzantine_vector, gradient_clip)
        byzantine_vectors = [byzantine_vector] * nb_byz
        distance = aggregator.aggregate(temp_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = line_maximize(eval_factor_FOE)
    return fall_of_empires(honest_vectors, attack_factor=best_factor)


def a_little_is_enough(honest_vectors, attack_factor=1.5, negative=False, **kwargs):
    #JS: negative controls the sign of the attack_factor
    #JS: attack_factor controls the magnitude of the attack_vector to be added to avg_honest_vector
    stacked_vectors = torch.stack(honest_vectors)
    attack_vector = stacked_vectors.var(dim=0).sqrt_()
    if negative:
        attack_factor = - attack_factor
    return stacked_vectors.mean(dim=0).add(attack_vector, alpha=attack_factor)


def auto_ALIE(honest_vectors, aggregator, nb_byz, gradient_clip, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_ALIE(factor):
        #temp_vectors = copy.deepcopy(honest_vectors)
        temp_vectors = [vector.clone() for vector in honest_vectors]
        byzantine_vector = a_little_is_enough(temp_vectors, attack_factor=factor)
        if gradient_clip is not None:
            byzantine_vector = clip_vector(byzantine_vector, gradient_clip)
        byzantine_vectors = [byzantine_vector] * nb_byz
        distance = aggregator.aggregate(temp_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = line_maximize(eval_factor_ALIE)
    return a_little_is_enough(honest_vectors, attack_factor=best_factor)


def mimic(attack, honest_vectors, current_step, **kwargs):
    mu_mimic, z_mimic = attack.mimic_state(honest_vectors)
    if attack.learning_phase_mimic is None:
        #JS: Always return the vector of the first worker
        return honest_vectors[0]

    if current_step < attack.learning_phase_mimic:
        #JS: Update mu_mimic and z_mimic only if still in learning phase
        mu_mimic, z_mimic = attack.update_mimic_heuristic(
            honest_vectors,
            current_step,
            mu_mimic=mu_mimic,
            z_mimic=z_mimic,
        )
        attack.store_mimic_state(mu_mimic, z_mimic)
        #JS: Return the vector of the first worker in learning phase
        return honest_vectors[0]

    current_max = -1
    best_worker_to_mimic = None
    for i, vector in enumerate(honest_vectors):
        dot_product = torch.dot(vector, z_mimic).norm().item()
        if dot_product > current_max:
            current_max = dot_product
            best_worker_to_mimic = i

    return honest_vectors[best_worker_to_mimic]


#JS: infinity attack
def inf(attack, **kwargs):
    honest_vectors = kwargs.get("honest_vectors", [])
    device = honest_vectors[0].device if honest_vectors else attack.device
    return torch.ones(attack.model_size, device=device) * float('inf')

#JS: Dictionary mapping every Byzantine attack to its corresponding function
byzantine_attacks = {"SF": signflipping, "LF": labelflipping, "FOE": fall_of_empires, "ALIE": a_little_is_enough, "mimic": mimic,
                    "auto_ALIE": auto_ALIE, "auto_FOE": auto_FOE, "inf": inf}

class ByzantineAttack(object):

    def __init__(self, attack_name, nb_real_byz, model_size, device, learning_phase, gradient_clip, robust_aggregator):
        self.attack_name = attack_name
        #JS: nb_real_byz is the actual number of "real" Byzantine workers
        self.nb_real_byz = nb_real_byz
        #JS: Instantiate the robust aggregator to be used (in particular to be used for auto ALIE and auto FOE)
        self.robust_aggregator = robust_aggregator
        self.model_size = model_size
        self.gradient_clip = gradient_clip
        self.device = device
        #JS: parameters for the mimic_heuristic attack
        self.z_mimic = torch.rand(model_size, device=device)
        self.mu_mimic = torch.zeros(model_size, device=device)
        self.learning_phase_mimic = learning_phase


    def generate_byzantine_vectors(self, honest_vectors, flipped_vectors, current_step):
        if self.nb_real_byz == 0:
            return list()

        #JS: Compute the Byzantine vectors
        byzantine_vector = byzantine_attacks[self.attack_name](attack=self, honest_vectors=honest_vectors, flipped_vectors=flipped_vectors, learning_phase=self.learning_phase_mimic,
                                                   current_step=current_step, aggregator=self.robust_aggregator, nb_byz = self.nb_real_byz, gradient_clip=self.gradient_clip)
        if self.gradient_clip is not None:
            byzantine_vector = clip_vector(byzantine_vector, self.gradient_clip)

        return [byzantine_vector] * self.nb_real_byz


    #JS: update the parameters of the mimic attack
    def mimic_state(self, honest_vectors):
        if not honest_vectors:
            return self.mu_mimic, self.z_mimic
        device = honest_vectors[0].device
        if self.mu_mimic.device == device:
            return self.mu_mimic, self.z_mimic
        return self.mu_mimic.to(device), self.z_mimic.to(device)

    def store_mimic_state(self, mu_mimic, z_mimic):
        self.mu_mimic = mu_mimic.to(self.device)
        self.z_mimic = z_mimic.to(self.device)

    def update_mimic_heuristic(
        self,
        honest_vectors,
        current_step,
        mu_mimic=None,
        z_mimic=None,
    ):
        if mu_mimic is None or z_mimic is None:
            mu_mimic, z_mimic = self.mimic_state(honest_vectors)
        time_factor = 1 / (current_step + 2)
        step_ratio = (current_step + 1) * time_factor
        mu_mimic.mul_(step_ratio)
        mu_mimic.add_(torch.stack(honest_vectors).mean(dim=0), alpha=time_factor)

        z_mimic.mul_(step_ratio)
        cumulative = torch.zeros(self.model_size, device=mu_mimic.device)
        for vector in honest_vectors:
            deviation = torch.sub(vector, mu_mimic)
            deviation.mul_(torch.dot(deviation, z_mimic).norm().item())
            cumulative.add_(deviation)

        z_mimic.mul_(step_ratio)
        z_mimic.add_(torch.nn.functional.normalize(cumulative, dim=0), alpha=time_factor)
        z_mimic.div_(z_mimic.norm().item())
        return mu_mimic, z_mimic
