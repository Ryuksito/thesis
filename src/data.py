import numpy as np
import jax.numpy as jnp

class JAXBatchLoader:
    def __init__(self, x_dict, y_dict, material_ids, batch_size, drop_last=True):
        """
        Cargador Multidimensional de Datos para Modelos Difusivos.
        Recibe diccionarios con las piezas segmentadas del cristal.
        """
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.material_ids = np.array(material_ids)
        self.batch_size = batch_size
        self.num_samples = len(self.material_ids)
        self.drop_last = drop_last
        self._generator = self.inf_generator()

    def inf_generator(self):
        indices_pendientes = np.array([], dtype=int)
        while True:
            if len(indices_pendientes) < self.batch_size:
                nueva_vuelta = np.random.permutation(self.num_samples)
                indices_pendientes = np.concatenate((indices_pendientes, nueva_vuelta))
            
            batch_idx = indices_pendientes[:self.batch_size]
            indices_pendientes = indices_pendientes[self.batch_size:]
            
            # Extraemos los IDs correspondientes al batch actual
            batch_ids = self.material_ids[batch_idx]
            
            batch_x = {k: jnp.array(v[batch_idx]) for k, v in self.x_dict.items()}
            batch_y = {k: jnp.array(v[batch_idx]) for k, v in self.y_dict.items()}
            
            # Entregamos batch_ids junto con los datos
            yield batch_x, batch_y, batch_ids

    def __iter__(self):
        return self._generator
    
    def __next__(self):
        return next(self._generator)

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size