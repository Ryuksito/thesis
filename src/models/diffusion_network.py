import jax
import jax.numpy as jnp
from typing import Callable

from tensorneat import algorithm, genome
from tensorneat.common.functions import act_jnp
from dataclasses import dataclass

@dataclass
class BaseNEAT:
    pop_size: int = 100
    species_size: int = 10
    survival_threshold: float = 0.1

@dataclass
class BaseGenome:
    max_nodes: int = 100
    max_conns: int = 500

@dataclass
class BaseMutation:
    conn_add_prob: float = 0.1
    conn_delete_prob: float = 0.05
    node_add_prob: float = 0.05
    node_delete_prob: float = 0.02

class CrystalDiffusionNetwork:
    def __init__(
        self,
        elements_dim: int = 18,
        embeddings_dim: int = 64,
        props_dim: int = 3,
        lattice_dim: int = 6,
        atom_dim: int = 4, 
        neat_config: BaseNEAT = None,
        genome_config: BaseGenome = None,
        mutation_config: BaseMutation = None,
    ):
        self.neat_config = neat_config or BaseNEAT()
        self.genome_config = genome_config or BaseGenome()
        self.mutation_config = mutation_config or BaseMutation()
        
        self.num_inputs = 1 + elements_dim + embeddings_dim + props_dim + lattice_dim + atom_dim 
        self.num_outputs = lattice_dim + atom_dim 
        
        def identity_activation(x): return x

        self.neat = algorithm.NEAT(
            pop_size=self.neat_config.pop_size,
            species_size=self.neat_config.species_size,
            survival_threshold=self.neat_config.survival_threshold,
            genome=genome.DefaultGenome(
                num_inputs=self.num_inputs,
                num_outputs=self.num_outputs,
                max_nodes=self.genome_config.max_nodes,
                max_conns=self.genome_config.max_conns,
                init_hidden_layers=(),
                output_transform=identity_activation, 
                mutation=genome.DefaultMutation(
                    conn_add=self.mutation_config.conn_add_prob,
                    conn_delete=self.mutation_config.conn_delete_prob,
                    node_add=self.mutation_config.node_add_prob,
                    node_delete=self.mutation_config.node_delete_prob,
                ),
            ),
        )

    def forward_crystal(self, state, nodes, conns, t, elements, embeddings, props, noisy_lattice, noisy_atoms):
        """
        CORRECCIÓN 1: Interfaz ajustada a 'nodes' y 'conns'.
        """
        # Transformación segura directo desde el genome
        transformed_network = self.neat.genome.transform(state, nodes, conns)
        
        t_arr = jnp.atleast_1d(t)
        global_context = jnp.concatenate([t_arr, elements, embeddings, props, noisy_lattice])
        
        def process_single_atom(atom_features):
            network_input = jnp.concatenate([global_context, atom_features])
            # Usamos el forward directo del genome
            return self.neat.genome.forward(state, transformed_network, network_input)
        
        outputs = jax.vmap(process_single_atom)(noisy_atoms)
        
        # CLIPPING GLOBAL: Protege contra activaciones lineales explosivas
        pred_noise_lattice = jnp.clip(jnp.mean(outputs[:, :6], axis=0), -5.0, 5.0) 
        pred_noise_atoms = jnp.clip(outputs[:, 6:], -5.0, 5.0) 
        
        return pred_noise_lattice, pred_noise_atoms