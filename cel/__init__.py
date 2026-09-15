"""CEL — Compute–Energy–Labor model. Build steps 1–2 implemented."""
from .params import Params, S0, SCENARIOS, scenario
from .grid import TaskGrid
from .economy import solve_period, labor_share_closed_form, Infeasible, PeriodResult
from .cohorts import DisplacedPool
