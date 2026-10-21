from .common import CommonCfg, CommonTrainCfg


class BaselineCfg(CommonCfg):
    pass


class BaselineTrainCfg(CommonTrainCfg):
    class runner(CommonTrainCfg.runner):
        run_name = "baseline"
