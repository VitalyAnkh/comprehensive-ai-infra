"""Selected figure 12-4 panels from exact shared-media public event results."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from .paths import PROJECT
from .sources import is_lfs_pointer, sha256_or_lfs_oid

DIRECTORY = PROJECT / "figures/shared-media"
NAMES = (
    "hol-connection",
    "hol-per_stream",
    "schedule-fifo",
    "schedule-priority",
    "playback-reliable",
    "playback-slots",
    "mixed-priority",
)
ARTIFACTS = ("figure.png", "figure.svg", "data.json")


def read_results():
    return {
        name: json.loads((PROJECT / f"results/shared-media-{name}.json").read_text())
        for name in NAMES
    }


def input_paths():
    paths = [
        Path(__file__),
        PROJECT / "src/infra_calc/topics/shared_media_transport.py",
        PROJECT / "scenarios/book.json",
        PROJECT / "configs/sources.lock.json",
    ]
    for name, result in read_results().items():
        paths.append(PROJECT / f"results/shared-media-{name}.json")
        paths.extend(
            PROJECT / row["file"] for row in result.get("reference_sources", [])
        )
    return sorted(set(paths))


def hashes(paths):
    return [
        dict(
            file=str(path.relative_to(PROJECT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    ]


def verify():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    lfs_unpulled = any(
        is_lfs_pointer(PROJECT / f"results/shared-media-{name}.json") for name in NAMES
    )
    expected = dict(
        inputs=None if lfs_unpulled else {str(p.relative_to(PROJECT)) for p in input_paths()},
        artifacts={str((DIRECTORY / name).relative_to(PROJECT)) for name in ARTIFACTS},
    )
    for key, names in expected.items():
        rows = manifest.get(key)
        if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
            raise ValueError("Malformed shared-media figure manifest")
        actual = [r.get("file") for r in rows]
        if names is not None and (len(actual) != len(names) or set(actual) != names):
            raise ValueError("Incomplete or duplicate shared-media figure manifest")
        for row in rows:
            digest = sha256_or_lfs_oid(PROJECT / row["file"])
            if digest != row["sha256"]:
                raise ValueError(
                    "Stale shared-media figure input/artifact: " + row["file"]
                )
    return {"verified_figures": 2}


def render():
    from .reproduce import verify_results

    verify_results(include_figures=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    results = read_results()
    number = lambda value: float(Fraction(str(value)))
    # Panel A uses one immutable receipt trace, not two different simulations.
    hol = results["hol-connection"]
    if hol["transmissions"] != results["hol-per_stream"]["transmissions"]:
        raise ValueError("HOL comparison requires identical wire trace")
    for name in ("schedule", "playback"):
        variants = ("fifo", "priority") if name == "schedule" else ("reliable", "slots")
        left, right = [results[name + "-" + v] for v in variants]
        signature = lambda r: [
            (p["id"], p["payload_bytes"]) for p in r["inputs"]["packets"]
        ]
        if signature(left) != signature(right):
            raise ValueError(
                "Comparisons must retain packet identities and payload sizes"
            )
    if (
        results["playback-reliable"]["transmissions"]
        != results["playback-slots"]["transmissions"]
    ):
        raise ValueError("Playback comparison requires identical wire trace")
    mixed = results["mixed-priority"]
    data = dict(
        figure="12-4-selected-panels",
        metadata_kind="declared_teaching",
        scope="Finite packet-ID transport. Selected panels only; not complete figure 12-4 scope or measured TCP/QUIC.",
        hol=dict(
            fixed_transmissions=hol["transmissions"],
            delivery=hol["delivery_only_replay"],
        ),
        scheduling={
            v: dict(
                transmissions=results["schedule-" + v]["transmissions"],
                businesses=results["schedule-" + v]["businesses"],
            )
            for v in ("fifo", "priority")
        },
        playback={
            v: results["playback-" + v]["businesses"] for v in ("reliable", "slots")
        },
        mixed=dict(
            visible_until_seconds_exact="1",
            transmissions=[
                r for r in mixed["transmissions"] if Fraction(r["start"]) < 1
            ],
            work=[r for r in mixed["work"] if Fraction(r["start"]) < 1],
            businesses=mixed["businesses"],
        ),
    )
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "shared-media-figure12-4-v1",
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.09,
        right=0.97,
        bottom=0.13,
        top=0.87,
        hspace=0.64,
        wspace=0.29,
        height_ratios=[1, 1, 1.35],
    )
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[0, 1])
    c = fig.add_subplot(grid[1, :])
    d = fig.add_subplot(grid[2, :])
    blue, orange, green = "#2368a0", "#d07924", "#27856b"
    fig.text(
        0.09,
        0.96,
        "Figure 12-4 | Shared media: selected quantitative panels",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.09,
        0.925,
        "Declared finite teaching transport; no measured TCP / QUIC performance. Exact trace data accompany this figure.",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.09,
        0.9,
        "A–C use small integer teaching traces. D uses a separate 30 MB image + PCM + screenshot workload.",
        fontsize=10,
        color="#444444",
    )
    packets = ["B0", "A0", "B1"]
    for i, mode in enumerate(("connection", "per_stream")):
        values = [number(data["hol"]["delivery"][mode][p]) for p in packets]
        a.barh(
            [j + (i - 0.5) * 0.32 for j in range(3)],
            values,
            height=0.29,
            color=blue if i == 0 else orange,
            label=mode.replace("_", " "),
        )
        for j, value in enumerate(values):
            a.text(
                value + 0.08,
                j + (i - 0.5) * 0.32,
                f"{value:g}",
                va="center",
                fontsize=9,
            )
    a.set_yticks(range(3), ["B0 (lost first)", "A0 audio", "B1 image"])
    a.invert_yaxis()
    a.set_xlim(0, 10.5)
    a.set_xlabel("Application delivery (s)")
    a.set_title(
        "A. HOL: same receipt trace, different delivery order",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    a.legend(frameon=False, fontsize=9, loc="lower right")
    for i, strategy in enumerate(("fifo", "priority")):
        rows = {r["id"]: r for r in results["schedule-" + strategy]["businesses"]}
        values = [number(rows[k]["complete"]) for k in ("image", "audio")]
        b.barh(
            [j + (i - 0.5) * 0.32 for j in range(2)],
            values,
            height=0.29,
            color=blue if i == 0 else orange,
            label=strategy,
        )
        for j, value in enumerate(values):
            b.text(
                value + 0.08,
                j + (i - 0.5) * 0.32,
                f"{value:g}",
                va="center",
                fontsize=9,
            )
    b.set_yticks([0, 1], ["Image complete", "Audio complete"])
    b.invert_yaxis()
    b.set_xlim(0, 8.7)
    b.set_xlabel("Complete application delivery (s)")
    b.set_title(
        "B. Same packets: FIFO vs priority scheduling",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    b.legend(frameon=False, fontsize=9, loc="lower right")
    for y, policy in enumerate(("reliable", "slots")):
        business = results["playback-" + policy]["businesses"][0]
        for i, block in enumerate(business["blocks"]):
            start, end = number(block["start"]), number(block["end"])
            c.barh(
                y,
                end - start,
                left=start,
                height=0.45,
                color=green if block["played"] else "#f8e7df",
                edgecolor="white" if block["played"] else "#b3261e",
                hatch=None if block["played"] else "///",
            )
            c.text(
                (start + end) / 2,
                y,
                f"block {i}" if block["played"] else "missing",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if block["played"] else "#8b241c",
            )
        c.text(
            10.2,
            y,
            f"stall {business['stall_seconds']} s; missing {business['missing_seconds']} s",
            va="center",
            fontsize=9,
        )
    c.set_yticks([0, 1], ["Reliable", "Fixed slots"])
    c.invert_yaxis()
    c.set_xlim(2.5, 13)
    c.set_xticks(range(3, 11))
    c.set_xlabel("Playback time (s); all three packets arrive in both policies")
    c.set_title(
        "C. Identical arrivals: 1 s stall vs 2 s missing audio",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    colors = {
        "image": blue,
        "audio": green,
        "screen": orange,
        "ack": "#777777",
        "work": "#9874a7",
    }

    def color(row):
        if row["kind"] == "ack":
            return colors["ack"]
        stream = row["stream"]
        return (
            colors["image"]
            if "image" in stream
            else colors["screen"] if "screen" in stream else colors["audio"]
        )

    for row in data["mixed"]["transmissions"]:
        y = 0 if row["direction"] == "c2s" else 1
        start, end = number(row["start"]), min(1, number(row["end"]))
        d.broken_barh([(start, end - start)], (y - 0.22, 0.44), facecolors=color(row))
    for row in data["mixed"]["work"]:
        start, end = number(row["start"]), min(1, number(row["end"]))
        d.broken_barh(
            [(start, end - start)], (2 - 0.22, 0.44), facecolors=colors["work"]
        )
    for row in mixed["businesses"]:
        field = "first_play" if row["kind"] == "tts" else "complete"
        if row.get(field) is not None and Fraction(row[field]) <= 1:
            at = number(row[field])
            label = row["id"] + (" (stale)" if row.get("usable") is False else "")
            d.plot(
                at,
                3,
                "o",
                color="#b3261e" if row.get("usable") is False else "#222222",
                markersize=5,
            )
            d.annotate(
                label,
                (at, 3),
                xytext=(4, -14 if row["kind"] == "asr" else 9),
                textcoords="offset points",
                fontsize=8,
            )
    d.set_yticks(
        range(4),
        [
            "c2s wire",
            "s2c wire",
            "Server work",
            "Delivery / play",
        ],
    )
    d.set_ylim(3.6, -0.6)
    d.set_xlim(0, 1)
    d.set_xlabel("Absolute time (s); serialization bars exclude propagation")
    d.set_title(
        "D. Mixed priority workload: first 1 s only (image continues beyond view)",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    d.legend(
        handles=[Patch(color=value, label=key) for key, value in colors.items()],
        ncol=5,
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    for ax in (a, b, c, d):
        ax.grid(axis="x", alpha=0.14)
        ax.set_axisbelow(True)
    final_image = next(r["complete"] for r in mixed["businesses"] if r["id"] == "image")
    fig.text(
        0.09,
        0.075,
        f"D complete image: {number(final_image):.6f} s. A stale screenshot is delivered but unusable; receipt is not playback.",
        fontsize=10,
    )
    fig.text(
        0.09,
        0.048,
        "No claim of full figure 12-4 completion: actual protocol handshakes, congestion control and measured media execution remain separate.",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(
        DIRECTORY / "figure.svg",
        metadata={"Date": None, "Creator": "shared_media_plot"},
    )
    fig.savefig(
        DIRECTORY / "figure.png", dpi=160, metadata={"Software": "shared_media_plot"}
    )
    plt.close(fig)
    manifest = dict(
        inputs=hashes(input_paths()),
        artifacts=hashes([DIRECTORY / name for name in ARTIFACTS]),
        scope=data["scope"],
    )
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"directory": str(DIRECTORY.relative_to(PROJECT)), **verify()}
