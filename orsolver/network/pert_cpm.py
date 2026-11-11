from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Task:
    name: str
    duration: float
    predecessors: List[str] = field(default_factory=list)


@dataclass
class CPMResult:
    project_duration: float
    critical_path: List[str]
    schedule: Dict[str, Dict[str, float]]


def compute_cpm(tasks: List[Task]) -> CPMResult:
    """
    Computes Early Start (ES), Early Finish (EF),
    Late Start (LS), Late Finish (LF), Slack,
    and the Critical Path of a project.
    """
    task_dict = {t.name: t for t in tasks}
    successors: Dict[str, List[str]] = {t.name: [] for t in tasks}
    in_degree: Dict[str, int] = {t.name: 0 for t in tasks}

    for t in tasks:
        for p in t.predecessors:
            successors[p].append(t.name)
            in_degree[t.name] += 1

    # Topological sort (Kahn's algorithm)
    ready = [t.name for t in tasks if in_degree[t.name] == 0]
    topo_order = []

    while ready:
        curr = ready.pop(0)
        topo_order.append(curr)
        for succ in successors[curr]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                ready.append(succ)

    if len(topo_order) != len(tasks):
        raise ValueError("Project graph contains a cycle (circular dependency)!")

    # Forward pass: calculate ES and EF
    es: Dict[str, float] = {t.name: 0.0 for t in tasks}
    ef: Dict[str, float] = {t.name: 0.0 for t in tasks}

    for name in topo_order:
        t = task_dict[name]
        if t.predecessors:
            es[name] = max(ef[p] for p in t.predecessors)
        else:
            es[name] = 0.0
        ef[name] = es[name] + t.duration

    project_duration = max(ef.values()) if ef else 0.0

    # Backward pass: calculate LF and LS
    lf: Dict[str, float] = {t.name: project_duration for t in tasks}
    ls: Dict[str, float] = {t.name: 0.0 for t in tasks}

    for name in reversed(topo_order):
        t = task_dict[name]
        if successors[name]:
            lf[name] = min(ls[succ] for succ in successors[name])
        else:
            lf[name] = project_duration
        ls[name] = lf[name] - t.duration

    # Slack calculation
    schedule = {}
    critical_tasks = set()
    for name in topo_order:
        slack = ls[name] - es[name]
        is_crit = abs(slack) < 1e-9
        if is_crit:
            critical_tasks.add(name)
        schedule[name] = {
            "ES": es[name],
            "EF": ef[name],
            "LS": ls[name],
            "LF": lf[name],
            "slack": slack,
            "is_critical": is_crit,
        }

    # Extract critical path sequence
    critical_path = [name for name in topo_order if name in critical_tasks]

    return CPMResult(
        project_duration=project_duration,
        critical_path=critical_path,
        schedule=schedule,
    )
