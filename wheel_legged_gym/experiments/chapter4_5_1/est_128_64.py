from .common import EstimatorCfg, EstimatorTrainCfg


class Est12864Cfg(EstimatorCfg):
    pass


class Est12864TrainCfg(EstimatorTrainCfg):
    class policy(EstimatorTrainCfg.policy):
        estimator_hidden_dims = [128, 64]

    class runner(EstimatorTrainCfg.runner):
        run_name = "est_128_64"
