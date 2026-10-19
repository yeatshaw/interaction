"""Recipe selector settings for adaptive EoH on constructive VRPTW."""

ADAPTIVE_RECIPE_EOH_CONFIG = {
    "enabled": False,
    "strategy": "linear_thompson_sampling",
    "prior_precision": 1.0,
    "posterior_scale": 0.5,
    "discount_factor": 0.9,
    "window_size": 50,
    "exploration": 1.0,
}
