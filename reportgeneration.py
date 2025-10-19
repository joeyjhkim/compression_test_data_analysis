import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

base_dir = os.getcwd()
output_pdf = os.path.join(base_dir, "CompressionTestingReport.pdf")

sections = {
    "Highly Squishy Material": [
        "highSquish_side1.is_comp_Exports",
        "highSquish_side2.is_comp_Exports",
        "highSquish_side3.is_comp_Exports"
    ],
    "Moderately Squishy Material": [
        "lowSquish_side1.is_comp_Exports",
        "lowSquish_side2.is_comp_Exports",
        "lowSquish_side3.is_comp_Exports"
    ]
}

# === HELPER: process one curve ===
def process_curve(disp, force, smooth = True):
    """
    Cleans, averages, and returns a smooth spline-fitted curve.
    Returns (x_smooth, y_smooth)
    """
    df = pd.DataFrame({"disp": disp, "force": force})

    # Round displacement to 0.01 mm bins to remove actuator chatter
    df["disp"] = df["disp"].round(2)

    # Average out repeated displacements
    df = df.groupby("disp", as_index = False).mean().dropna()
    df = df.sort_values("disp")

    x = df["disp"].to_numpy()
    y = df["force"].to_numpy()

    # Remove non-monotonic points that cause backtracking
    mask = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[mask], y[mask]

    # Require sufficient data points
    if len(x) < 6:
        return x, y

    # Dense spline interpolation — handle increasing or decreasing data safely
    if np.all(np.diff(x) > 0):
        # Increasing sequence (normal)
        x_dense = np.linspace(x.min(), x.max(), 1000)
        spline = make_interp_spline(x, y, k = 3)
        y_dense = spline(x_dense)
    elif np.all(np.diff(x) < 0):
        # Decreasing sequence (unloading phase)
        x_dense = np.linspace(x.max(), x.min(), 1000)
        spline = make_interp_spline(x[::-1], y[::-1], k = 3)
        y_dense = spline(x_dense)
    else:
        # Mixed sequence — sort to enforce monotonic behavior
        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
        spline = make_interp_spline(x_sorted, y_sorted, k = 3)
        y_dense = spline(x_dense)

    # Optional Savitzky–Golay smoothing
    if smooth and len(y_dense) > 31:
        window = min(101, len(y_dense) // 30 * 2 + 1)
        y_dense = savgol_filter(y_dense, window_length = window, polyorder = 3)
    
    return x_dense, y_dense

# === HELPER: find initial elastic peak ===
def find_initial_peak(x, y):
    """
    Finds the first major local maximum (initial elastic peak)
    in the loading curve and returns its (x, y) coordinates.
    """
    # Find index of global maximum within the first 40% of the curve
    # (avoids noise at the end or densification peaks)
    limit = int(len(y) * 0.4)
    idx_peak = np.argmax(y[:limit])
    return x[idx_peak], y[idx_peak]


# === MAIN REPORT GENERATION ===
with PdfPages(output_pdf) as pdf:
    for section_name, sides in sections.items():
        for side in sides:
            folder_path = os.path.join(base_dir, side)
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue

            csv_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")],
            key = lambda name: int(''.join(filter(str.isdigit, name)) or 0)
)
            if not csv_files:
                print(f"No CSVs found in {folder_path}")
                continue

            all_loads, all_unloads = [], []

            # === PROCESS EACH TRIAL ===
            fig, ax = plt.subplots(figsize = (8, 6))
            for i, file in enumerate(csv_files, start = 1):
                df = pd.read_csv(os.path.join(folder_path, file))

                disp_col = next((c for c in df.columns if "disp" in c.lower()), None)
                force_col = next((c for c in df.columns if "force" in c.lower()), None)
                if not disp_col or not force_col:
                    print(f"Skipping {file} — missing columns.")
                    continue

                # Clean + convert
                df[disp_col] = pd.to_numeric(df[disp_col], errors = "coerce")
                df[force_col] = pd.to_numeric(df[force_col], errors = "coerce")
                df = df.dropna(subset = [disp_col, force_col])

                df[disp_col] -= df[disp_col].iloc[0]  # zero start
                df[force_col] *= 1000  # N → mN

                disp = df[disp_col].to_numpy()
                force = df[force_col].to_numpy()

                # Split at max displacement
                peak_idx = np.argmax(disp)
                disp_load, force_load = disp[:peak_idx + 1], force[:peak_idx + 1]
                disp_unload, force_unload = disp[peak_idx:], force[peak_idx:]

                # Process both halves
                x_load, y_load = process_curve(disp_load, force_load)
                x_unload, y_unload = process_curve(disp_unload, force_unload)

                # Store processed data
                all_loads.append((x_load, y_load))
                all_unloads.append((x_unload, y_unload))

                color = ax._get_lines.get_next_color()
                ax.plot(x_load, y_load, color = color, linewidth = 1.0, label = f"Trial {i}")
                ax.plot(x_unload, y_unload, color = color, linewidth = 1.0)
                

            # === FORMAT ===
            ax.set_title(f"{section_name} – {side.replace('_', ' ').title()} (All Trials)")
            ax.set_xlabel("Displacement (mm)")
            ax.set_ylabel("Force (mN)")
            ax.grid(True)
            ax.legend(fontsize = 7)
            pdf.savefig(fig)
            plt.close(fig)

            # === AVERAGE ACROSS TRIALS ===
            if not all_loads:
                continue

            # Interpolate all to shared grid
            common_x = np.linspace(
                min(min(x) for x, _ in all_loads),
                max(max(x) for x, _ in all_loads),
                400
            )

            load_interp = []
            for x, y in all_loads:
                spline = make_interp_spline(x, y, k = 3)
                load_interp.append(spline(common_x))
            avg_load = np.mean(np.column_stack(load_interp), axis = 1)

            unload_interp = []
            for x, y in all_unloads:
                spline = make_interp_spline(x, y, k = 3)
                unload_interp.append(spline(common_x))
            avg_unload = np.mean(np.column_stack(unload_interp), axis = 1)

            # === PLOT AVERAGE CURVE ===
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(common_x, avg_load, color = "black", linewidth = 1.5, label = "Average Load (Smoothed)")
            ax.plot(common_x, avg_unload, color = "black", linewidth = 1.5)
            
            # === Detect and annotate initial elastic peak ===
            try:
                x_peak, y_peak = find_initial_peak(common_x, avg_load)
                ax.axvline(x = x_peak, color = "red", linestyle = ":", linewidth = 0.8, alpha = 0.8)
                ax.text(
                    x_peak, 0, f"{y_peak:.1f} mN",
                    rotation = 90, va = "bottom", ha = "right",
                    fontsize = 8, color = "red",
                    fontweight = "bold"
                )
            except Exception:
                pass
            
            ax.set_title(f"{section_name} – {side.replace('_', ' ').title()} (Average Curve)")
            ax.set_xlabel("Displacement (mm)")
            ax.set_ylabel("Force (mN)")
            ax.grid(True)
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
            

    print(f"Report saved to: {output_pdf}")