import csv
import os
import matplotlib.pyplot as plt

RUNS_DIR = r"C:\Dirk\Neethling\iems\master\PythonLBCourse\param_study_phase2_theta_runs"
RESULTS_CSV = r"C:\Dirk\Neethling\iems\master\PythonLBCourse\param_study_phase2_theta_runs\param_study_phase2_theta_results.csv"

def load_profile(run_no):
    path = os.path.join(RUNS_DIR, run_no, "interface_profile.csv")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        xs = []
        ys = []
        for row in reader:
            x_pos = int(row["x"])        
            xs.append(x_pos)
            y_pos = float(row["interface_y"])
            ys.append(y_pos)

    return xs, ys


def load_results():
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []

        for row in reader:
            run_no = row["label"]
            rows.append(row)

        return rows


if __name__ == "__main__":
    rows = load_results()
    rows.sort(key=lambda r: float(r["vf_theta"]))  # so the color order makes sense

    fig, ax = plt.subplots()

    for row in rows:
        xs, ys = load_profile(row["label"])
        ax.plot(xs, ys, label=f"theta={row['vf_theta']}")

    ax.set_xlabel("x-index")
    ax.set_ylabel("interface y-index (phi=0.5 crossing)")
    ax.set_title("Meniscus profiles vs vf_theta")
    ax.legend(title="vf_theta", loc="lower center", ncol=2)
    ax.grid(True, color="lightgrey", linestyle=":")
    fig.savefig("zhang_meniscus_profiles.png", dpi=200, bbox_inches="tight")
    plt.show()