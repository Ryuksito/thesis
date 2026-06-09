import jax.numpy as jnp

def noise_loss_fn(pred_noise_lat, pred_noise_atoms, real_noise_lat, real_noise_atoms, mask):
    """
    Pérdida para el Gradiente Lamarckiano (Aprender a quitar el ruido).
    Aplica la máscara para ignorar el error en los átomos de padding.
    """
    # 1. Error de la Caja (Lattice)
    lat_loss = jnp.mean((pred_noise_lat - real_noise_lat)**2)
    
    # 2. Error de los Átomos
    atom_error_cuadratico = (pred_noise_atoms - real_noise_atoms)**2
    # Sumamos el error de las 4 coordenadas (Z, x, y, z)
    atom_error_sum = jnp.sum(atom_error_cuadratico, axis=-1) 
    
    # Aplicamos la máscara: Multiplicamos por 0 el error de los átomos fantasma
    atom_error_masked = atom_error_sum * mask
    
    # Promediamos solo sobre los átomos reales
    atom_loss = jnp.sum(atom_error_masked) / jnp.maximum(jnp.sum(mask), 1.0)
    
    return lat_loss + atom_loss


def crystal_loss_fn(preds, y_target, lattice_params=6, max_atoms=4, separate_results=False):
    """
    Pérdida para la Supervivencia NEAT (Filtro Geométrico Invariante).
    """
    pred_lat = preds[:, :lattice_params]
    target_lat = y_target[:, :lattice_params]
    
    pred_atoms = preds[:, lattice_params:].reshape(-1, max_atoms, 4)
    target_atoms = y_target[:, lattice_params:].reshape(-1, max_atoms, 4)
    
    pred_z = pred_atoms[:, :, 0]
    target_z = target_atoms[:, :, 0]
    
    mask = (target_z > 0).astype(jnp.float32)
    mask_2d = mask[:, :, None] * mask[:, None, :]
    
    # 1. LATTICE
    lat_loss = jnp.mean((pred_lat - target_lat)**2)

    # 2. ACSF
    def get_sorted_fingerprints(pos, m2d, eta=5.0):
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        diff = diff - jnp.round(diff)
        dist_sq = jnp.sum(diff**2, axis=-1)
        gaussians = jnp.exp(-eta * dist_sq) * m2d
        
        eye = jnp.eye(pos.shape[1])[None, :, :]
        gaussians = gaussians * (1.0 - eye)
        
        fingerprints = jnp.sum(gaussians, axis=-1)
        return jnp.sort(fingerprints, axis=-1)

    fp_pred = get_sorted_fingerprints(pred_atoms[:, :, 1:], mask_2d)
    fp_target = get_sorted_fingerprints(target_atoms[:, :, 1:], mask_2d)
    acsf_loss = jnp.sum(((fp_pred - fp_target)**2) * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # 3. Z-LOSS INVARIANTE
    sorted_pred_z = jnp.sort(pred_z * mask, axis=1)
    sorted_target_z = jnp.sort(target_z, axis=1)
    z_loss = jnp.mean((sorted_pred_z - sorted_target_z)**2)

    # 4. VARIANCE & REPULSION
    fp_var_pred = jnp.var(fp_pred * mask, axis=0)
    fp_var_target = jnp.var(fp_target * mask, axis=0)
    var_loss = jnp.mean((fp_var_pred - fp_var_target)**2)
    
    repulsion = jnp.exp(-1000.0 * (fp_pred + 1e-6)) * mask
    repulsion_loss = jnp.sum(repulsion) / jnp.maximum(jnp.sum(mask), 1.0)

    w_lat = 10.0 * lat_loss
    w_acsf = 50.0 * acsf_loss
    w_z = 20.0 * z_loss
    w_var = 10.0 * var_loss
    w_rep = 10.0 * repulsion_loss

    total_loss = w_lat + w_acsf + w_z + w_var + w_rep
    
    if separate_results:
        return total_loss, w_lat, w_acsf, w_z, w_var, w_rep
    
    return total_loss