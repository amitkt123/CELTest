"""CEL — Compute–Energy–Labor model. Build steps 1–2 implemented."""
from .params import Params, S0, SCENARIOS, scenario, load_params
from .grid import TaskGrid
from .economy import solve_period, labor_share_closed_form, Infeasible, PeriodResult
from .cohorts import DisplacedPool
from .capability import omega_train, omega_inf
