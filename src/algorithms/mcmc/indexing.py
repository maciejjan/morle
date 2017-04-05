from typing import Any, Dict, Iterable

class Index:
    def __init__(self) -> None:
        self.items_dict = {}    # type: Dict[Any, int]
        self.last_id = 0

    def __contains__(self, key :Any) -> bool:
        return key in self.items_dict

    def __len__(self) -> int:
        return len(self.items_dict)

    def __iter__(self) -> Iterable:
        return iter(self.items_dict)

    def __getitem__(self, key :Any) -> int:
        return self.items_dict[key]

    def add(self, key :Any) -> None:
        if key not in self:
            self.items_dict[key] = self.last_id
            self.last_id += 1

    def items(self):
        return self.items_dict.items()

