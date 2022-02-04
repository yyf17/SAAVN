#!/usr/bin/env python3

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        from ss_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_env(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a environment to registry with key 'name'
            currently only support subclass of RLEnv.

        Args:
            name: Key with which the env will be registered.
                If None will use the name of the class.

        """

        return cls._register_impl("env", to_register, name)

    @classmethod
    def get_env(cls, name):
        return cls._get_impl("env", name)


baseline_registry = BaselineRegistry()
