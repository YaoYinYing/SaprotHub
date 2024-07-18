import yaml
from easydict import EasyDict
from pathlib import Path

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigPreset:
    current_file_path = Path(__file__).resolve()  # 获取当前脚本的绝对路径
    current_dir_path = current_file_path.parent

    def _read_config(self, config_path: str):
        with open(
            self.current_dir_path / config_path, "r", encoding="utf-8"
        ) as r:
            return EasyDict(yaml.safe_load(r))

    @property
    def Default_config(self) -> EasyDict:
        return self._read_config(config_path="default.yaml")

    @property
    def DeepLoc_cls2_config(self) -> EasyDict:
        return self._read_config(config_path="DeepLoc/cls2/saprot.yaml")

    @property
    def DeepLoc_cls10_config(self) -> EasyDict:
        return self._read_config(config_path="DeepLoc/cls10/saprot.yaml")

    @property
    def EC_config(self) -> EasyDict:
        return self._read_config(config_path="EC/saprot.yaml")

    @property
    def GO_BP_config(self) -> EasyDict:
        return self._read_config(config_path="GO/BP/saprot.yaml")

    @property
    def GO_CC_config(self) -> EasyDict:
        return self._read_config(config_path="GO/CC/saprot.yaml")

    @property
    def GO_MF_config(self) -> EasyDict:
        return self._read_config(config_path="GO/MF/saprot.yaml")

    @property
    def HumanPPI_config(self) -> EasyDict:
        return self._read_config(config_path="HumanPPI/saprot.yaml")

    @property
    def MetalIonBinding_config(self) -> EasyDict:
        return self._read_config(config_path="MetalIonBinding/saprot.yaml")

    @property
    def Thermostability_config(self) -> EasyDict:
        return self._read_config(config_path="Thermostability/saprot.yaml")

    @property
    def ClinVar_config(self) -> EasyDict:
        return self._read_config(config_path="ClinVar/saprot.yaml")

    @property
    def ProteinGym_config(self) -> EasyDict:
        return self._read_config(config_path="ProteinGym/saprot.yaml")
