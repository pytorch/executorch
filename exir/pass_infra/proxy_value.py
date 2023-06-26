# pyre-strict
import functools
from typing import Union

import torch
from executorch.exir.error import InternalError
from torch.utils import _pytree as pytree


# pyre-ignore
def unwrap_proxy_value(f):
    @functools.wraps(f)
    # pyre-ignore
    def _unwrap_proxy_value(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        return f(*args, **kwargs)

    return _unwrap_proxy_value


class ProxyValue:
    # pyre-ignore
    def __init__(self, data, proxy: Union[torch.fx.Proxy, torch.fx.Node]):
        # pyre-ignore
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> torch.fx.Node:
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        assert isinstance(self.proxy_or_node, torch.fx.Proxy)
        return self.proxy_or_node.node

    @property
    def proxy(self) -> torch.fx.Proxy:
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise InternalError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        return self.proxy_or_node

    def to_tensor(self) -> torch.Tensor:
        assert isinstance(self.data, torch.Tensor)
        return self.data

    def is_tensor(self) -> bool:
        return isinstance(self.data, torch.Tensor)

    @unwrap_proxy_value
    # pyre-ignore
    def __call__(self, *args):
        return self.data(*args)

    # pyre-ignore
    def __iter__(self):
        for data in self.data:
            yield data

    def __bool__(self) -> bool:
        return bool(self.data)
