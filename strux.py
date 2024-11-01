import dataclasses

import jax
import equinox as eqx

def struct(static_fieldnames=()):
    def _struct(Class):
        """
        Wrapper that transforms a class into an immutable dataclass that is also
        registered as a JAX PyTree and uses equinox for __str__ and __repr__.
        """
        # wrap class as an immutable Python dataclass
        Dataclass = dataclasses.dataclass(Class, frozen=True)
        # decide which fields are data vs. static
        fields = [field.name for field in dataclasses.fields(Dataclass)]
        data_fields = [name for name in fields if name not in static_fieldnames]
        meta_fields = [name for name in fields if name in static_fieldnames]
        # TODO: it should be an error to have static_fieldnames not used
        # register dataclass as a JAX pytree node
        jax.tree_util.register_dataclass(
            nodetype=Dataclass,
            data_fields=data_fields,
            meta_fields=meta_fields,
        )
        # overwrite string render methods to use equinox pretty-printing
        Dataclass.__repr__ = eqx.tree_pformat
        Dataclass.__str__ = eqx.tree_pformat
        # other convenience methods
        Dataclass.replace = dataclasses.replace
        return Dataclass

    return _struct