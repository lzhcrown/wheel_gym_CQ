from .common import EstimatorCfg, EstimatorTrainCfg


class Est6464Cfg(EstimatorCfg):
    pass


class Est6464TrainCfg(EstimatorTrainCfg):
    class policy(EstimatorTrainCfg.policy):
        estimator_hidden_dims = [64, 64]

    class runner(EstimatorTrainCfg.runner):
        run_name = "est_64_64"
