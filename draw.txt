import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# ============================================================
# Basic Settings
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(20, 15), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ============================================================
# Helper Functions
# ============================================================

def rounded_box(
    x, y, w, h,
    facecolor="#FFFFFF",
    edgecolor="#78909C",
    linewidth=1.5,
    radius=0.012,
    linestyle="-",
    zorder=2
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def text(
    x, y, s,
    fontsize=10,
    weight="normal",
    color="#263238",
    ha="center",
    va="center",
    zorder=5
):
    ax.text(
        x, y, s,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def arrow(
    x1, y1, x2, y2,
    color="#607D8B",
    linewidth=1.4,
    linestyle="-"
):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=1,
    )


def line(x1, y1, x2, y2, color="#607D8B", linewidth=1.2, linestyle="-"):
    ax.plot(
        [x1, x2],
        [y1, y2],
        transform=ax.transAxes,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=1,
    )


# ============================================================
# TITLE
# ============================================================

rounded_box(
    0.30, 0.935, 0.40, 0.05,
    facecolor="#D9EAF7",
    edgecolor="#1565C0",
    linewidth=2.0,
)

text(
    0.50, 0.965,
    "Cooperative Perception Survey Taxonomy",
    fontsize=19,
    weight="bold",
    color="#0D47A1",
)

text(
    0.50, 0.945,
    "From Broad Research Directions to Object-Level Asynchronous Alignment",
    fontsize=10,
    color="#1565C0",
)


# ============================================================
# LEVEL 1
# Six Main Research Directions
# ============================================================

topics = [
    ("1. Communication\nEfficiency",
     "Where2comm\nEffiComm",
     "#E8F5E9", "#388E3C"),

    ("2. Heterogeneity",
     "PolyInter\nPHCP\nModel-Agnostic CP",
     "#EEE5FA", "#7E57C2"),

    ("3. Latency &\nAsynchrony",
     "SyncNet\nCoBEVFlow\nTraF-Align\nCoAnchor",
     "#DCEBFA", "#1976D2"),

    ("4. Robustness &\nSecurity",
     "RCP-Bench\nAdversarial Robustness",
     "#FCE4EC", "#C62828"),

    ("5. V2I /\nInfrastructure",
     "STDS\nInfrastructure-assisted CP",
     "#FFF3E0", "#EF6C00"),

    ("6. Real-World\nDeployment",
     "Safety\nSim-to-Real\nDeployment",
     "#E0F2F1", "#00897B"),
]

topic_y = 0.805
topic_w = 0.145
topic_h = 0.085
gap = 0.018
start_x = 0.015

topic_centers = []

for i, (title, examples, fc, ec) in enumerate(topics):

    x = start_x + i * (topic_w + gap)

    # Highlight Latency
    highlight = (i == 2)

    rounded_box(
        x, topic_y,
        topic_w, topic_h,
        facecolor=fc,
        edgecolor="#1565C0" if highlight else ec,
        linewidth=2.5 if highlight else 1.3,
    )

    text(
        x + topic_w / 2,
        topic_y + 0.060,
        title,
        fontsize=9.5,
        weight="bold",
        color="#0D47A1" if highlight else "#263238",
    )

    text(
        x + topic_w / 2,
        topic_y + 0.022,
        examples,
        fontsize=7.2,
        color="#455A64",
    )

    topic_centers.append(x + topic_w / 2)


# Top hierarchy connector

line(
    0.50, 0.935,
    0.50, 0.895,
    color="#455A64",
)

line(
    topic_centers[0],
    0.895,
    topic_centers[-1],
    0.895,
    color="#455A64",
)

for cx in topic_centers:
    arrow(
        cx, 0.895,
        cx, topic_y + topic_h,
        color="#607D8B",
        linewidth=1.0,
    )


# ============================================================
# LEVEL 2
# Focus on Latency & Asynchrony
# ============================================================

# Large dashed focus area
rounded_box(
    0.02, 0.425,
    0.96, 0.335,
    facecolor="#F5F9FD",
    edgecolor="#1976D2",
    linewidth=1.5,
    linestyle="--",
    radius=0.015,
    zorder=0,
)

# Focus title

rounded_box(
    0.32, 0.715,
    0.36, 0.045,
    facecolor="#E3F2FD",
    edgecolor="#1565C0",
    linewidth=1.8,
)

text(
    0.50, 0.738,
    "3. Latency & Asynchrony",
    fontsize=15,
    weight="bold",
    color="#0D47A1",
)

text(
    0.50, 0.690,
    "Communication delay, processing delay, and moving objects cause temporal misalignment",
    fontsize=9,
    color="#455A64",
)


# Connector from Level 1 Latency

arrow(
    topic_centers[2],
    topic_y,
    0.50,
    0.715,
    color="#1976D2",
    linewidth=2.0,
    linestyle="--",
)


# ============================================================
# LEVEL 2 SUB-TAXONOMY
# Three directions inside Latency & Asynchrony
# ============================================================

sub_y = 0.475
sub_w = 0.285
sub_h = 0.17

sub_xs = [0.035, 0.3575, 0.68]

subcategories = [

    {
        "title": "3.1 Temporal\nSynchronization",
        "desc": "Synchronize timestamps and\nestimate communication latency.",
        "methods": "SyncNet\nLatency-aware feature estimation",
        "fc": "#E3F2FD",
        "ec": "#0288D1",
    },

    {
        "title": "3.2 Asynchronous\nFeature Alignment",
        "desc": "Align stale features to the\ncurrent timestamp.",
        "methods": "Feature Prediction\nMotion Compensation\nTrajectory Alignment",
        "fc": "#E8F5E9",
        "ec": "#2E7D32",
    },

    {
        "title": "3.3 Communication-efficient\nLatency Reduction",
        "desc": "Reduce transmitted information\nto reduce communication delay.",
        "methods": "Where2comm\nEffiComm\nSTDS\nSlimComm",
        "fc": "#FFF3E0",
        "ec": "#EF6C00",
    },
]


for i, sub in enumerate(subcategories):

    x = sub_xs[i]

    rounded_box(
        x, sub_y,
        sub_w, sub_h,
        facecolor=sub["fc"],
        edgecolor=sub["ec"],
        linewidth=1.5,
    )

    text(
        x + sub_w / 2,
        sub_y + 0.132,
        sub["title"],
        fontsize=11,
        weight="bold",
        color=sub["ec"],
    )

    text(
        x + sub_w / 2,
        sub_y + 0.085,
        sub["desc"],
        fontsize=8,
        color="#455A64",
    )

    line(
        x + 0.02,
        sub_y + 0.058,
        x + sub_w - 0.02,
        sub_y + 0.058,
        color="#B0BEC5",
        linewidth=0.8,
    )

    text(
        x + sub_w / 2,
        sub_y + 0.025,
        sub["methods"],
        fontsize=7.4,
        color="#37474F",
    )


# Highlight Asynchronous Feature Alignment

rounded_box(
    sub_xs[1] - 0.006,
    sub_y - 0.006,
    sub_w + 0.012,
    sub_h + 0.012,
    facecolor="none",
    edgecolor="#2E7D32",
    linewidth=2.4,
    linestyle="--",
    zorder=4,
)


# Connector from Latency title to three branches

line(
    0.50, 0.715,
    0.50, 0.665,
    color="#1565C0",
)

line(
    sub_xs[0] + sub_w / 2,
    0.665,
    sub_xs[2] + sub_w / 2,
    0.665,
    color="#1565C0",
)

for x in sub_xs:

    arrow(
        x + sub_w / 2,
        0.665,
        x + sub_w / 2,
        sub_y + sub_h,
        color="#1565C0",
        linewidth=1.2,
    )


# ============================================================
# LEVEL 3
# Asynchronous Feature Alignment
# ============================================================

# Large focus box

rounded_box(
    0.02, 0.075,
    0.96, 0.325,
    facecolor="#F6FBF7",
    edgecolor="#2E7D32",
    linewidth=1.5,
    linestyle="--",
    radius=0.015,
    zorder=0,
)


# Title

rounded_box(
    0.30, 0.365,
    0.40, 0.045,
    facecolor="#E8F5E9",
    edgecolor="#2E7D32",
    linewidth=1.8,
)

text(
    0.50, 0.388,
    "3.2 Asynchronous Feature Alignment",
    fontsize=14,
    weight="bold",
    color="#1B5E20",
)

text(
    0.50, 0.340,
    "How can stale cooperative information be aligned to the current timestamp?",
    fontsize=9,
    color="#455A64",
)


# Connector from 3.2

arrow(
    sub_xs[1] + sub_w / 2,
    sub_y,
    0.50,
    0.365,
    color="#2E7D32",
    linewidth=2.0,
    linestyle="--",
)


# ============================================================
# Four Alignment Strategies
# ============================================================

alignment_y = 0.105
alignment_w = 0.215
alignment_h = 0.195
alignment_xs = [0.035, 0.265, 0.495, 0.725]

alignment = [

    {
        "title": "3.2.1 Feature\nPrediction",
        "question": "Predict what the\nfeature looks like now.",
        "papers": "SyncNet",
        "detail": "Historical feature\n→ current feature",
        "fc": "#E3F2FD",
        "ec": "#1976D2",
    },

    {
        "title": "3.2.2 Motion\nCompensation",
        "question": "Compensate object motion\ncaused by latency.",
        "papers": "FFNet\nCoBEVFlow",
        "detail": "Motion / BEV Flow\n→ feature warping",
        "fc": "#E8F5E9",
        "ec": "#388E3C",
    },

    {
        "title": "3.2.3 Trajectory\nAlignment",
        "question": "Use temporal trajectories\nto find correspondence.",
        "papers": "TraF-Align",
        "detail": "Trajectory field\n→ sampling / attention",
        "fc": "#FFF3E0",
        "ec": "#EF6C00",
    },

    {
        "title": "3.2.4 Object-Level\nAlignment",
        "question": "Use object states / anchors\nas alignment references.",
        "papers": "CoAnchor\nObject-level CP",
        "detail": "Object correspondence\n→ temporal alignment",
        "fc": "#F3E5F5",
        "ec": "#7B1FA2",
    },
]


for i, item in enumerate(alignment):

    x = alignment_xs[i]

    # Stronger highlight for Object-Level Alignment
    is_object = (i == 3)

    rounded_box(
        x, alignment_y,
        alignment_w, alignment_h,
        facecolor=item["fc"],
        edgecolor=item["ec"],
        linewidth=2.2 if is_object else 1.3,
    )

    text(
        x + alignment_w / 2,
        alignment_y + 0.158,
        item["title"],
        fontsize=10,
        weight="bold",
        color=item["ec"],
    )

    text(
        x + alignment_w / 2,
        alignment_y + 0.115,
        item["question"],
        fontsize=7.5,
        color="#455A64",
    )

    line(
        x + 0.015,
        alignment_y + 0.085,
        x + alignment_w - 0.015,
        alignment_y + 0.085,
        color="#B0BEC5",
        linewidth=0.8,
    )

    text(
        x + alignment_w / 2,
        alignment_y + 0.058,
        item["papers"],
        fontsize=8,
        weight="bold",
        color="#0D47A1",
    )

    text(
        x + alignment_w / 2,
        alignment_y + 0.025,
        item["detail"],
        fontsize=7,
        color="#455A64",
    )


# ============================================================
# LEVEL 4
# Object-Level Alignment Deep Dive
# ============================================================

# New separate bottom/side conceptual box

rounded_box(
    0.055, 0.015,
    0.89, 0.045,
    facecolor="#F3E5F5",
    edgecolor="#7B1FA2",
    linewidth=1.8,
)

text(
    0.50, 0.037,
    "→ Object-Level Alignment: Object / Track / Anchor as the Temporal Correspondence Unit",
    fontsize=11,
    weight="bold",
    color="#6A1B9A",
)


# ============================================================
# Strong Visual Emphasis on the Main Research Path
# ============================================================

# Highlight path:
# 6 directions → Latency → Async Feature Alignment → Object Level

# Vertical path from Latency box to focus
arrow(
    topic_centers[2],
    topic_y,
    topic_centers[2],
    0.715,
    color="#1565C0",
    linewidth=2.5,
    linestyle="--",
)

# Vertical path from Async Feature Alignment to Object Level
arrow(
    alignment_xs[3] + alignment_w / 2,
    alignment_y,
    alignment_xs[3] + alignment_w / 2,
    0.060,
    color="#7B1FA2",
    linewidth=2.2,
    linestyle="--",
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "cooperative_perception_latency_object_alignment_tree.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print("Saved:")
print("cooperative_perception_latency_object_alignment_tree.png")