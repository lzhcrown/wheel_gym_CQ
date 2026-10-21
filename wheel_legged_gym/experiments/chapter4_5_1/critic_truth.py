from .common import CriticTruthCfg as CommonCriticTruthCfg, CommonTrainCfg


class CriticTruthCfg(CommonCriticTruthCfg):
    pass


class CriticTruthTrainCfg(CommonTrainCfg):
    class runner(CommonTrainCfg.runner):
        run_name = "critic_truth"
