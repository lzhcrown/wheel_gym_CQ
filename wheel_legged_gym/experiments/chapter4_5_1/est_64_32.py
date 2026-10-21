from .common import EstimatorCfg, EstimatorTrainCfg


class Est6432Cfg(EstimatorCfg):
    pass


class Est6432TrainCfg(EstimatorTrainCfg):
    class policy(EstimatorTrainCfg.policy):
        estimator_hidden_dims = [64, 32]

    class runner(EstimatorTrainCfg.runner):
        run_name = "est_64_32"
