import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import dearpygui.dearpygui as dpg

# ---------------------- Data Model ----------------------

@dataclass
class Node:
    id: int
    label: str
    x: float
    y: float
    r: float = 22.0

@dataclass
class Edge:
    src: int
    dst: int
    p: float

class ChainModel:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.next_node_id: int = 1

    def add_node(self, label: Optional[str], x: float, y: float) -> int:
        if not label:
            label = f"S{self.next_node_id}"
        nid = self.next_node_id
        self.next_node_id += 1
        self.nodes[nid] = Node(id=nid, label=label, x=x, y=y)
        return nid

    def delete_node(self, nid: int) -> None:
        if nid in self.nodes:
            del self.nodes[nid]
        for key in list(self.edges.keys()):
            if key[0] == nid or key[1] == nid:
                del self.edges[key]

    def relabel_node(self, nid: int, new_label: str) -> None:
        if nid in self.nodes:
            self.nodes[nid].label = new_label

    def add_or_update_edge(self, src: int, dst: int, p: float) -> None:
        p = max(0.0, float(p))
        self.edges[(src, dst)] = Edge(src=src, dst=dst, p=p)

    def delete_edge(self, src: int, dst: int) -> None:
        self.edges.pop((src, dst), None)

    def outgoing(self, nid: int) -> List[Edge]:
        return [e for (s, _), e in self.edges.items() if s == nid]

    def normalize_outgoing(self, nid: int) -> None:
        outs = self.outgoing(nid)
        s = sum(e.p for e in outs)
        if s > 0:
            for e in outs:
                e.p = e.p / s

    def normalize_all(self) -> None:
        for nid in list(self.nodes.keys()):
            self.normalize_outgoing(nid)

    def to_json(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }

    def from_json(self, data: dict) -> None:
        self.nodes = {n["id"]: Node(**n) for n in data.get("nodes", [])}
        self.edges = {}
        for e in data.get("edges", []):
            self.edges[(e["src"], e["dst"])] = Edge(**e)
        self.next_node_id = (max(self.nodes.keys()) + 1) if self.nodes else 1

    def simulate(self, steps: int, start_node: Optional[int] = None, seed: Optional[int] = None) -> Tuple[List[int], Dict[int, int]]:
        if not self.nodes:
            raise ValueError("No nodes to simulate.")
        rng = random.Random(seed)
        if start_node is None or start_node not in self.nodes:
            start_node = next(iter(self.nodes))
        visit_order: List[int] = []
        counts: Dict[int, int] = {nid: 0 for nid in self.nodes}
        current = start_node
        counts[current] += 1
        visit_order.append(current)
        for _ in range(steps - 1):
            outs = self.outgoing(current)
            if not outs:
                nxt = current
            else:
                weights = [max(0.0, e.p) for e in outs]
                total = sum(weights)
                if total <= 0:
                    nxt = rng.choice(outs).dst
                else:
                    r = rng.random() * total
                    c = 0.0
                    nxt = outs[-1].dst
                    for e, w in zip(outs, weights):
                        c += w
                        if r <= c:
                            nxt = e.dst
                            break
            current = nxt
            counts[current] += 1
            visit_order.append(current)
        return visit_order, counts

# ---------------------- UI Helpers ----------------------

class AppState:
    def __init__(self):
        self.model = ChainModel()
        self.canvas_tag = "canvas_drawlist"
        self.sidebar_width = 350
        self.canvas_size = (1000, 700)
        self.selected_node: Optional[int] = None
        self.dragging: bool = False
        self.drag_offset: Tuple[float, float] = (0, 0)
        self.edge_mode: bool = False
        self.edge_src: Optional[int] = None
        self.file_path: str = "chain_project.json"
        self.sim_steps: int = 1000
        self.sim_seed: Optional[int] = None
        self.sim_start: Optional[int] = None
        self.last_counts: Dict[int, int] = {}
        self.last_order: List[int] = []
        self.arrow_gap: float = 12.0   # pixels, orthogonal offset when a reverse edge exists
        self.label_nudge: float = 10.0 # pixels, how far the label sits off the line
        # === MULTI RUN ADDED ===
        self.sim_runs: int = 100
        self.multi_stats: Dict[int, Tuple[float, float]] = {}  # nid -> (avg_visits, avg_freq)

state = AppState()

def dist2(a, b): return (a[0]-b[0])**2 + (a[1]-b[1])**2

def draw_arrow(parent, x1, y1, x2, y2, thickness=2.0):
    dpg.draw_line((x1, y1), (x2, y2), thickness=thickness, parent=parent)
    vx, vy = x2 - x1, y2 - y1
    L = math.hypot(vx, vy) or 1.0
    ux, uy = vx / L, vy / L
    size = 12.0
    left = (x2 - ux * size - uy * size * 0.6, y2 - uy * size + ux * size * 0.6)
    right = (x2 - ux * size + uy * size * 0.6, y2 - uy * size - ux * size * 0.6)
    dpg.draw_triangle((x2, y2), left, right, fill=(200, 200, 200, 255), parent=parent)

def redraw_canvas():
    dpg.delete_item(state.canvas_tag, children_only=True)

    for e in state.model.edges.values():
        if e.src not in state.model.nodes or e.dst not in state.model.nodes:
            continue
        n1 = state.model.nodes[e.src]
        n2 = state.model.nodes[e.dst]

        dx, dy = n2.x - n1.x, n2.y - n1.y
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux

        if n1.id == n2.id:
            r = n1.r + 18
            cx, cy = n1.x + r, n1.y
            pts = [
                (n1.x + n1.r, n1.y),
                (cx, cy - r * 0.8),
                (n1.x, n1.y - r * 0.8),
                (n1.x - n1.r * 0.2, n1.y),
            ]
            for i in range(len(pts) - 1):
                dpg.draw_line(pts[i], pts[i+1], thickness=2.0, parent=state.canvas_tag)
            draw_arrow(state.canvas_tag, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], thickness=2.0)
            label_pos = (n1.x, n1.y - r - 10)
        else:
            has_reverse = (e.dst, e.src) in state.model.edges
            # Canonical normal / side selection so reverse edges split
            if n1.id < n2.id:
                nx_c, ny_c = -uy, ux
                side = +1 if has_reverse else 0
            else:
                nx_c, ny_c = uy, -ux
                side = -1 if has_reverse else 0
            offset = state.arrow_gap * side

            ax1 = n1.x + ux * n1.r + nx_c * offset
            ay1 = n1.y + uy * n1.r + ny_c * offset
            ax2 = n2.x - ux * n2.r + nx_c * offset
            ay2 = n2.y - uy * n2.r + ny_c * offset

            draw_arrow(state.canvas_tag, ax1, ay1, ax2, ay2, thickness=2.0)
            label_pos = ((ax1 + ax2) * 0.5 + nx_c * (state.label_nudge * side),
                         (ay1 + ay2) * 0.5 + ny_c * (state.label_nudge * side))

        dpg.draw_text(label_pos, f"{e.p:.3f}", size=14, parent=state.canvas_tag)

    for nid, n in state.model.nodes.items():
        color = (40, 120, 220, 255) if nid == state.selected_node else (80, 160, 255, 255)
        dpg.draw_circle((n.x, n.y), n.r, fill=color, thickness=1.5, parent=state.canvas_tag)
        tw = len(n.label) * 6
        dpg.draw_text((n.x - tw * 0.45, n.y - 6), n.label, size=16, parent=state.canvas_tag)

    if state.edge_mode and state.edge_src is not None and state.edge_src in state.model.nodes:
        mx, my = dpg.get_mouse_pos(local=False)
        cx, cy = dpg.get_item_rect_min(state.canvas_tag)
        mx -= cx; my -= cy
        s = state.model.nodes[state.edge_src]
        draw_arrow(state.canvas_tag, s.x, s.y, mx, my, thickness=1.0)

def canvas_local_mouse():
    mx, my = dpg.get_mouse_pos(local=False)
    cx, cy = dpg.get_item_rect_min(state.canvas_tag)
    return mx - cx, my - cy

def hit_node(pos) -> Optional[int]:
    best = None
    best_d2 = 1e9
    for nid, n in state.model.nodes.items():
        if dist2((n.x, n.y), pos) <= (n.r + 3)**2 and dist2((n.x, n.y), pos) < best_d2:
            best, best_d2 = nid, dist2((n.x, n.y), pos)
    return best

# ---------------------- Callbacks ----------------------

def on_canvas_click(sender, app_data):
    btn = app_data[0] if isinstance(app_data, (list, tuple)) else app_data
    if btn != 0:
        return
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min(state.canvas_tag)
    x1, y1 = dpg.get_item_rect_max(state.canvas_tag)
    if not (x0 <= mx <= x1 and y0 <= my <= y1):
        return
    pos = (mx - x0, my - y0)
    nid = hit_node(pos)

    if state.edge_mode:
        if state.edge_src is None:
            if nid is not None:
                state.edge_src = nid
        else:
            if nid is not None:
                dpg.set_value("edge_prob_input", 1.0)
                dpg.show_item("edge_prob_modal")
                dpg.set_value("edge_modal_src", state.edge_src)
                dpg.set_value("edge_modal_dst", nid)
            else:
                state.edge_src = None
    else:
        state.selected_node = nid
        if nid is not None:
            n = state.model.nodes[nid]
            lx, ly = n.x, n.y
            mx_local, my_local = pos
            state.dragging = True
            state.drag_offset = (lx - mx_local, ly - my_local)
            dpg.set_value("node_label_input", n.label)
            dpg.set_value("node_id_text", f"Node ID: {n.id}")
            refresh_outgoing_table(nid)
        else:
            state.dragging = False
            dpg.set_value("node_id_text", "Node ID: (none)")
            dpg.set_value("node_label_input", "")
            clear_outgoing_table()

    redraw_canvas()

def on_canvas_release(sender, app_data):
    state.dragging = False

def on_canvas_drag(sender, app_data):
    if not state.dragging or state.selected_node is None:
        return
    pos = canvas_local_mouse()
    n = state.model.nodes[state.selected_node]
    n.x = pos[0] + state.drag_offset[0]
    n.y = pos[1] + state.drag_offset[1]
    redraw_canvas()

def add_node_clicked():
    label = dpg.get_value("new_node_label") or None
    cx, cy = state.canvas_size[0] * 0.5, state.canvas_size[1] * 0.5
    nid = state.model.add_node(label, cx, cy)
    state.selected_node = nid
    update_node_combos()
    redraw_canvas()

def delete_node_clicked():
    if state.selected_node is None:
        return
    state.model.delete_node(state.selected_node)
    state.selected_node = None
    update_node_combos()
    clear_outgoing_table()
    redraw_canvas()

def on_arrow_gap_change(sender, value):
    try:
        state.arrow_gap = max(0.0, float(value))
    except Exception:
        state.arrow_gap = 0.0
    redraw_canvas()

def start_edge_mode():
    state.edge_mode = True
    state.edge_src = None
    dpg.hide_item("edge_helper_off")
    dpg.show_item("edge_helper_on")
    redraw_canvas()

def stop_edge_mode():
    state.edge_mode = False
    state.edge_src = None
    dpg.hide_item("edge_helper_on")
    dpg.show_item("edge_helper_off")
    redraw_canvas()

def confirm_edge_modal():
    src_raw = dpg.get_value("edge_modal_src")
    dst_raw = dpg.get_value("edge_modal_dst")
    try:
        src = int(src_raw)
        dst = int(dst_raw)
    except Exception:
        src, dst = src_raw, dst_raw
    p = float(dpg.get_value("edge_prob_input"))
    state.model.add_or_update_edge(src, dst, p)
    dpg.hide_item("edge_prob_modal")
    state.edge_src = None
    if state.selected_node == src:
        refresh_outgoing_table(src)
    redraw_canvas()

def cancel_edge_modal():
    dpg.hide_item("edge_prob_modal")
    state.edge_src = None
    redraw_canvas()

def on_label_edit(sender, value):
    if state.selected_node is None:
        return
    state.model.relabel_node(state.selected_node, value)
    update_node_combos()
    refresh_outgoing_table(state.selected_node)
    redraw_canvas()

def normalize_clicked():
    if state.selected_node is not None:
        state.model.normalize_outgoing(state.selected_node)
        refresh_outgoing_table(state.selected_node)
    else:
        state.model.normalize_all()
        if state.selected_node is not None:
            refresh_outgoing_table(state.selected_node)
    redraw_canvas()

def save_clicked():
    path = dpg.get_value("file_path_input").strip()
    if not path:
        return
    data = state.model.to_json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    dpg.set_value("status_bar", f"Saved: {path}")

def load_clicked():
    path = dpg.get_value("file_path_input").strip()
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    state.model.from_json(data)
    state.selected_node = None
    update_node_combos()
    clear_outgoing_table()
    redraw_canvas()
    dpg.set_value("status_bar", f"Loaded: {path}")

def update_node_combos():
    labels = [f"{n.id}: {n.label}" for n in state.model.nodes.values()]
    dpg.configure_item("start_node_combo", items=labels)
    dpg.configure_item("target_node_combo", items=labels)   # === MULTI RUN ADDED ===
    if labels:
        if not dpg.get_value("start_node_combo"):
            dpg.set_value("start_node_combo", labels[0])
        if not dpg.get_value("target_node_combo"):
            dpg.set_value("target_node_combo", labels[0])

def parse_node_combo(tag: str) -> Optional[int]:
    val = dpg.get_value(tag)
    if not val:
        return None
    try:
        nid = int(val.split(":")[0])
        if nid in state.model.nodes:
            return nid
    except Exception:
        pass
    return None

def simulate_clicked():
    steps = int(max(1, dpg.get_value("sim_steps_input")))
    seed_s = dpg.get_value("sim_seed_input").strip()
    seed = None
    if seed_s:
        try:
            seed = int(seed_s)
        except Exception:
            seed = abs(hash(seed_s)) & 0xffffffff
    start_nid = parse_node_combo("start_node_combo")
    try:
        order, counts = state.model.simulate(steps=steps, start_node=start_nid, seed=seed)
    except Exception as e:
        dpg.set_value("status_bar", f"Sim error: {e}")
        return
    state.last_order = order
    state.last_counts = counts
    refresh_results_table(counts, steps)
    dpg.set_value("status_bar", f"Simulated {steps} steps.")

# === MULTI RUN ADDED ===
def simulate_many_clicked():
    runs = int(max(1, dpg.get_value("sim_runs_input")))
    steps = int(max(1, dpg.get_value("sim_steps_input")))
    start_nid = parse_node_combo("start_node_combo")
    target_nid = parse_node_combo("target_node_combo")
    base_seed_s = dpg.get_value("sim_seed_input").strip()
    base_seed = None
    if base_seed_s:
        try:
            base_seed = int(base_seed_s)
        except Exception:
            base_seed = abs(hash(base_seed_s)) & 0xffffffff

    # accumulators
    sum_visits = {nid: 0 for nid in state.model.nodes}
    target_sum = 0
    target_sumsq = 0.0
    target_hits_once = 0

    for i in range(runs):
        seed = (base_seed + i) if base_seed is not None else None
        _, counts = state.model.simulate(steps=steps, start_node=start_nid, seed=seed)
        for nid in state.model.nodes:
            sum_visits[nid] += counts.get(nid, 0)

        if target_nid is not None:
            v = counts.get(target_nid, 0)
            target_sum += v
            target_sumsq += float(v) * float(v)
            if v > 0:
                target_hits_once += 1

    # averages
    state.multi_stats = {
        nid: (sum_visits[nid] / runs, (sum_visits[nid] / runs) / steps)
        for nid in state.model.nodes
    }
    refresh_multi_table(state.multi_stats)

    # target summary
    if target_nid is not None:
        mean_v = target_sum / runs
        mean_f = mean_v / steps
        var_v = max(0.0, (target_sumsq / runs) - (mean_v ** 2))
        sd_v = var_v ** 0.5
        sd_f = sd_v / steps
        hit_prob = target_hits_once / runs
        lbl = state.model.nodes[target_nid].label
        dpg.set_value(
            "multi_summary",
            f"Target '{lbl}' (id {target_nid}) — runs={runs}, steps/run={steps}\n"
            f"Avg visits: {mean_v:.3f}   Avg freq: {mean_f:.5f}\n"
            f"StdDev visits: {sd_v:.3f}   StdDev freq: {sd_f:.5f}\n"
            f"P(visited ≥ 1): {hit_prob:.4f}"
        )
    else:
        dpg.set_value("multi_summary", "Select a target node to see summary.")
    dpg.set_value("status_bar", f"Ran {runs} runs × {steps} steps.")

def clear_outgoing_table():
    if not dpg.does_item_exist("out_table"):
        return
    try:
        rows = dpg.get_item_children("out_table", 1) or []
    except Exception:
        rows = []
    for rid in rows:
        dpg.delete_item(rid)

def refresh_outgoing_table(nid: int):
    if not dpg.does_item_exist("out_table"):
        return
    clear_outgoing_table()
    outs = sorted(state.model.outgoing(nid), key=lambda e: (e.src, e.dst))
    for e in outs:
        src_name = state.model.nodes[e.src].label if e.src in state.model.nodes else str(e.src)
        dst_name = state.model.nodes[e.dst].label if e.dst in state.model.nodes else str(e.dst)
        with dpg.table_row(parent="out_table"):
            dpg.add_text(f"{src_name} -> {dst_name}")
            dpg.add_input_float(
                default_value=float(e.p),
                width=100,
                step=0.0,
                user_data=e,                 # pass Edge safely
                callback=on_edge_prob_edit
            )
            dpg.add_button(label="Delete", user_data=e, callback=on_edge_delete)

def on_edge_prob_edit(sender, app_data, user_data):
    edge: Edge = user_data
    if edge is None:
        return
    try:
        edge.p = max(0.0, float(app_data))
    except Exception:
        return
    redraw_canvas()

def on_edge_delete(sender, app_data, user_data):
    edge: Edge = user_data
    if edge is None:
        return
    state.model.delete_edge(edge.src, edge.dst)
    if state.selected_node is not None:
        refresh_outgoing_table(state.selected_node)
    redraw_canvas()

def refresh_results_table(counts: Dict[int, int], steps: int):
    if not dpg.does_item_exist("res_table"):
        return
    try:
        rows = dpg.get_item_children("res_table", 1) or []
    except Exception:
        rows = []
    for rid in rows:
        dpg.delete_item(rid)

    total = max(1, steps)
    for nid, n in state.model.nodes.items():
        c = counts.get(nid, 0)
        with dpg.table_row(parent="res_table"):
            dpg.add_text(f"{n.label} (id {nid})")
            dpg.add_text(str(c))
            dpg.add_text(f"{c/total:.4f}")

# === MULTI RUN ADDED ===
def refresh_multi_table(stats: Dict[int, Tuple[float, float]]):
    if not dpg.does_item_exist("multi_table"):
        return
    try:
        rows = dpg.get_item_children("multi_table", 1) or []
    except Exception:
        rows = []
    for rid in rows:
        dpg.delete_item(rid)

    for nid, n in state.model.nodes.items():
        avg_v, avg_f = stats.get(nid, (0.0, 0.0))
        with dpg.table_row(parent="multi_table"):
            dpg.add_text(f"{n.label} (id {nid})")
            dpg.add_text(f"{avg_v:.3f}")
            dpg.add_text(f"{avg_f:.5f}")

# ---------------------- UI Build ----------------------

def build_ui():
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True)
    dpg.create_viewport(
        title="MCMC Graph Editor",
        width=state.sidebar_width + state.canvas_size[0] + 20,
        height=state.canvas_size[1] + 70
    )

    with dpg.window(
        tag="root_dockspace",
        label="",
        no_title_bar=True,
        no_move=True,
        no_resize=True,
        no_collapse=True,
        no_close=True
    ):
        pass
    dpg.set_primary_window("root_dockspace", True)

    with dpg.window(
        tag="controls_window",
        label="Controls",
        width=state.sidebar_width,
        height=state.canvas_size[1] + 60,
        no_close=True
    ):
        dpg.add_text("Graph")
        with dpg.group(horizontal=True):
            dpg.add_input_text(label="New node label", tag="new_node_label", width=180, hint="(optional)")
            dpg.add_button(label="Add Node", callback=add_node_clicked)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add Edge", callback=start_edge_mode)
            dpg.add_button(label="Stop Adding", tag="edge_helper_on", callback=stop_edge_mode, show=False)
            dpg.add_text("(Click source, then target)", tag="edge_helper_off")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Delete Node", callback=delete_node_clicked)
            dpg.add_button(label="Normalize", callback=normalize_clicked)

        dpg.add_input_float(label="Arrow gap (px)", default_value=state.arrow_gap,
                    step=0.0, width=120, callback=on_arrow_gap_change)

        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_text("Selected Node")
        dpg.add_text("Node ID: (none)", tag="node_id_text")
        dpg.add_input_text(label="Label", tag="node_label_input", width=260, callback=on_label_edit)

        dpg.add_spacer(height=6)
        dpg.add_text("Outgoing edges")
        with dpg.table(tag="out_table", header_row=True, resizable=True, borders_innerH=True, borders_innerV=True, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="Edge")
            dpg.add_table_column(label="p")
            dpg.add_table_column(label="")
            with dpg.table_row():
                pass

        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_text("Save / Load")
        dpg.add_input_text(label="File", tag="file_path_input", default_value=state.file_path, width=260)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save", callback=save_clicked)
            dpg.add_button(label="Load", callback=load_clicked)

        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_text("Simulation")
        dpg.add_input_int(label="Steps N", tag="sim_steps_input", default_value=state.sim_steps, min_value=1, min_clamped=True, width=120)
        dpg.add_input_text(label="Seed (optional)", tag="sim_seed_input", width=120)
        dpg.add_combo(label="Start node", tag="start_node_combo", items=[], width=260)
        dpg.add_button(label="Run Simulation", callback=simulate_clicked)

        dpg.add_spacer(height=4)
        dpg.add_text("Results (visits & freq)")
        with dpg.table(tag="res_table", header_row=True, resizable=True, borders_innerH=True, borders_innerV=True, policy=dpg.mvTable_SizingStretchProp, height=160):
            dpg.add_table_column(label="State")
            dpg.add_table_column(label="Visits")
            dpg.add_table_column(label="Freq")
            with dpg.table_row():
                pass

        # === MULTI RUN ADDED ===
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_text("Monte Carlo (multiple runs)")
        dpg.add_input_int(label="Runs M", tag="sim_runs_input", default_value=state.sim_runs, min_value=1, min_clamped=True, width=120)
        dpg.add_combo(label="Target node", tag="target_node_combo", items=[], width=260)
        dpg.add_button(label="Run Multi-Sim", callback=simulate_many_clicked)
        dpg.add_spacer(height=4)
        dpg.add_text("", tag="multi_summary")
        with dpg.table(tag="multi_table", header_row=True, resizable=True, borders_innerH=True, borders_innerV=True, policy=dpg.mvTable_SizingStretchProp, height=180):
            dpg.add_table_column(label="State")
            dpg.add_table_column(label="Avg visits")
            dpg.add_table_column(label="Avg freq")
            with dpg.table_row():
                pass

        dpg.add_spacer(height=8)
        dpg.add_text("", tag="status_bar")

    with dpg.window(
        tag="canvas_window",
        label="Canvas",
        width=state.canvas_size[0] + 10,
        height=state.canvas_size[1] + 60
    ):
        dpg.add_text("Tip: drag nodes to move. Add Edge: click source then target. Self-loops allowed.")
        dpg.add_drawlist(width=state.canvas_size[0], height=state.canvas_size[1], tag=state.canvas_tag)
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=on_canvas_click)
            dpg.add_mouse_release_handler(callback=on_canvas_release)
            dpg.add_mouse_drag_handler(callback=on_canvas_drag, button=0)

    with dpg.window(
        tag="edge_prob_modal",
        label="Edge probability",
        modal=True,
        show=False,
        no_collapse=True,
        no_resize=True
    ):
        dpg.add_text("Set probability for edge:")
        dpg.add_text("", tag="edge_modal_src", show=False)
        dpg.add_text("", tag="edge_modal_dst", show=False)
        dpg.add_input_float(label="p", tag="edge_prob_input", default_value=1.0, min_value=0.0, min_clamped=True)
        with dpg.group(horizontal=True):
            dpg.add_button(label="OK", callback=confirm_edge_modal)
            dpg.add_button(label="Cancel", callback=cancel_edge_modal)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.show_item("controls_window")
    dpg.show_item("canvas_window")

    update_node_combos()
    redraw_canvas()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    build_ui()
