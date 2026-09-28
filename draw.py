import textwrap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 設定繪圖後端
matplotlib.use("Agg")

# 設定字型與畫布
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(18, 14), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    facecolor="#FFFFFF",
    edgecolor="#3B5F8A",
    linewidth=1.5,
    radius=0.015,
    zorder=2,
    linestyle="-",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        transform=ax.transAxes,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax,
    x,
    y,
    text,
    fontsize=9,
    weight="normal",
    color="#18324A",
    ha="center",
    va="center",
    linespacing=1.2,
    zorder=5,
):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#496A87", linewidth=1.2, style="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=linewidth,
            linestyle=style,
            color=color,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=1,
    )


# 1. TOP HEADER
rounded_box(
    ax,
    0.32,
    0.935,
    0.36,
    0.05,
    facecolor="#D9EAF7",
    edgecolor="#1976D2",
    linewidth=1.8,
)
add_text(
    ax,
    0.50,
    0.965,
    "Cooperative Perception Survey Taxonomy",
    fontsize=18,
    weight="bold",
    color="#0D47A1",
)
add_text(
    ax,
    0.50,
    0.945,
    "(Refactored 6 Main Research Directions)",
    fontsize=10,
    color="#1565C0",
)

# 2. TOPIC BOXES (Level 1)
TOPICS = [
    {
        "title": "1. Comm. Efficiency",
        "examples": "Where2comm\nEffiComm\n...",
        "color": "#E8F5E9",
    },
    {
        "title": "2. Heterogeneity",
        "examples": "PolyInter\nPHCP\nModel-Agnostic CP",
        "color": "#EEE5FA",
    },
    {
        "title": "3. Latency & Asynchrony",
        "examples": "SyncNet, CoBEVFlow\nTraF-Align, CoAnchor",
        "color": "#DCEBFA",
        "highlight": True,
    },
    {
        "title": "4. Robustness & Security",
        "examples": "RCP-Bench\nAdversarial Robustness\n...",
        "color": "#FCE3E3",
    },
    {
        "title": "5. V2I / Infrastructure",
        "examples": "STDS\nInfrastructure-assisted\n...",
        "color": "#FFF0D9",
    },
    {
        "title": "6. Real-World Deployment",
        "examples": "Safety\nSim-to-Real\nDeployment Constraints",
        "color": "#DFF4F4",
    },
]

topic_y = 0.81
topic_h = 0.08
topic_w = 0.145
gap = 0.015
start_x = 0.02
topic_centers = []

for i, topic in enumerate(TOPICS):
    x = start_x + i * (topic_w + gap)
    edge = "#1976D2" if topic.get("highlight") else "#78909C"
    lw = 2.2 if topic.get("highlight") else 1.2
    rounded_box(
        ax,
        x,
        topic_y,
        topic_w,
        topic_h,
        facecolor=topic["color"],
        edgecolor=edge,
        linewidth=lw,
    )
    add_text(
        ax,
        x + topic_w / 2,
        topic_y + 0.058,
        topic["title"],
        fontsize=9.5,
        weight="bold",
    )
    add_text(
        ax,
        x + topic_w / 2,
        topic_y + 0.022,
        topic["examples"],
        fontsize=7.5,
        color="#424242",
    )
    topic_centers.append((x + topic_w / 2, topic_y))

# 連接頂層架構
ax.plot(
    [0.50, 0.50],
    [0.935, 0.905],
    transform=ax.transAxes,
    color="#496A87",
    linewidth=1.2,
)
ax.plot(
    [topic_centers[0][0], topic_centers[-1][0]],
    [0.905, 0.905],
    transform=ax.transAxes,
    color="#496A87",
    linewidth=1.2,
)

for cx, cy in topic_centers:
    draw_arrow(ax, cx, 0.905, cx, cy + topic_h)

# 指向 Latency & Asynchrony 展開
latency_x, latency_y = topic_centers[2]
draw_arrow(
    ax, latency_x, latency_y, 0.50, 0.77, color="#1976D2", linewidth=1.5, style="--"
)

# 3. DEEPER LOOK: LATENCY & ASYNCHRONY (Level 2 外框)
zoom_box = FancyBboxPatch(
    (0.015, 0.015),
    0.97,
    0.74,
    transform=ax.transAxes,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    linewidth=1.5,
    linestyle="--",
    edgecolor="#1E88E5",
    facecolor="#F4F8FB",
    zorder=0,
)
ax.add_patch(zoom_box)

rounded_box(
    ax,
    0.30,
    0.72,
    0.40,
    0.045,
    facecolor="#E3F2FD",
    edgecolor="#1565C0",
    linewidth=1.6,
)
add_text(
    ax,
    0.50,
    0.742,
    "Detailed Taxonomy: 3. Latency & Asynchrony",
    fontsize=14,
    weight="bold",
    color="#0D47A1",
)

# 4. THREE SUB-TAXONOMIES (Level 2)
SUB_TAXONOMIES = [
    {
        "title": "3.1 Temporal Synchronization",
        "desc": "Focuses on global/local time sync and latency estimation.",
        "works": "• SyncNet\n• Latency-aware feature estimation",
        "color": "#E1F5FE",
        "edge": "#0288D1",
    },
    {
        "title": "3.2 Feature / Motion Alignment",
        "desc": "Warping BEV feature maps or motion vectors to compensate delay.",
        "works": "• FFNet\n• CoBEVFlow\n• TraF-Align",
        "color": "#E8F5E9",
        "edge": "#2E7D32",
    },
    {
        "title": "3.3 Object-Level Alignment",
        "desc": "Propagating object states/anchors forward in time.",
        "works": "• Track / Motion Propagation\n• Object State Prediction\n• CoAnchor (Spatio-Temporal)",
        "color": "#F3E5F5",
        "edge": "#7B1FA2",
    },
]

sub_y = 0.12
sub_h = 0.54
sub_w = 0.305
sub_xs = [0.03, 0.347, 0.665]

for idx, sub in enumerate(SUB_TAXONOMIES):
    x = sub_xs[idx]
    rounded_box(
        ax,
        x,
        sub_y,
        sub_w,
        sub_h,
        facecolor=sub["color"],
        edgecolor=sub["edge"],
        linewidth=1.5,
    )
    add_text(
        ax,
        x + sub_w / 2,
        sub_y + sub_h - 0.04,
        sub["title"],
        fontsize=12,
        weight="bold",
        color="#0D47A1" if idx == 0 else "#1B5E20" if idx == 1 else "#4A148C",
    )
    add_text(
        ax,
        x + sub_w / 2,
        sub_y + sub_h - 0.09,
        textwrap.fill(sub["desc"], width=32),
        fontsize=8.5,
        color="#37474F",
    )

    ax.plot(
        [x + 0.015, x + sub_w - 0.015],
        [sub_y + sub_h - 0.13, sub_y + sub_h - 0.13],
        transform=ax.transAxes,
        color="#B0BEC5",
        linewidth=0.8,
    )

    rounded_box(
        ax,
        x + 0.015,
        sub_y + 0.03,
        sub_w - 0.03,
        sub_h - 0.18,
        facecolor="#FFFFFF",
        edgecolor="#CFD8DC",
        linewidth=0.8,
    )
    add_text(
        ax,
        x + 0.03,
        sub_y + sub_h - 0.16,
        "Key Methods & Literature:",
        fontsize=9,
        weight="bold",
        color="#263238",
        ha="left",
    )
    ax.text(
        x + 0.03,
        sub_y + sub_h - 0.20,
        sub["works"],
        transform=ax.transAxes,
        fontsize=8.5,
        color="#0D47A1",
        ha="left",
        va="top",
        linespacing=1.4,
    )

ax.plot(
    [0.50, 0.50],
    [0.72, 0.69],
    transform=ax.transAxes,
    color="#1565C0",
    linewidth=1.2,
)
ax.plot(
    [sub_xs[0] + sub_w / 2, sub_xs[-1] + sub_w / 2],
    [0.69, 0.69],
    transform=ax.transAxes,
    color="#1565C0",
    linewidth=1.2,
)

for x_center in [
    sub_xs[0] + sub_w / 2,
    sub_xs[1] + sub_w / 2,
    sub_xs[2] + sub_w / 2,
]:
    draw_arrow(ax, x_center, 0.69, x_center, sub_y + sub_h)

# 儲存圖片並關閉畫布
plt.savefig(
    "cooperative_perception_refactored_tree.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()