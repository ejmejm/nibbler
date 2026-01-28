from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replaces the values of a tree with the provided keyword arguments."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


def configure_jax_config(
    cache_dir: str = '/tmp/jax_cache',
    device: Optional[str] = None,
):
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update('jax_persistent_cache_enable_xla_caches', 'xla_gpu_per_fusion_autotune_cache_dir')
    
    if device is not None:
        jax.config.update('jax_platform_name', device)
        print(f"JAX is using device: {jax.devices(device)[0]}")
    else:
        print(f"JAX is using the device: {jax.devices()[0]}")


def is_float_array(x: Any) -> bool:
    return isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating)