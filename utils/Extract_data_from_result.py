from pathlib import Path
import pandas as pd


def safe_read_csv(path, header_row=1):
    """Helper function to safely read CSV files with error handling."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path, header=header_row)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()


def extract_ThicknessReduction(fem_model):
    """Extract maximum thickness reduction state from simulation results."""
    source = Path(fem_model) / "%ThicknessReduction.csv"
    df = safe_read_csv(source)
    
    if df.empty or "A1" not in df.columns:
        return 0, pd.DataFrame()

    # Clean and process data
    df["A1"] = pd.to_numeric(df["A1"], errors="coerce")
    valid_data = df[df["A1"].notna() & (df["A1"] < 30)]
    
    if valid_data.empty:
        return 0, df
    
    max_index = valid_data["A1"].idxmax()
    return max_index + 1, df


def set_state_from_y_dis(fem_model):
    """Determine state from Y-displacement data."""
    source = Path(fem_model) / "Y-displacement.csv"
    df = safe_read_csv(source)
    
    if df.empty or len(df.columns) < 4:
        return 0
    
    try:
        y_series = pd.to_numeric(df.iloc[:, 3], errors="coerce").dropna()
        return y_series.idxmax() + 1 if not y_series.empty else 0
    except Exception as e:
        print(f"Y-displacement processing error: {e}")
        return 0


def extract_y_displacement(fem_model, max_index):
    """Extract maximum Y-displacement at specified state."""
    thicc_df = safe_read_csv(Path(fem_model) / "%ThicknessReduction.csv")
    y_df = safe_read_csv(Path(fem_model) / "Y-displacement.csv", header_row=0)
    
    if thicc_df.empty or y_df.empty:
        return 0.0

    try:
        max_time = pd.to_numeric(thicc_df["Time"], errors="coerce").iloc[max_index]
        time_series = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")
        y_disp = pd.to_numeric(y_df.iloc[:, 3], errors="coerce")
        
        valid_idx = time_series[time_series <= max_time].last_valid_index()
        return y_disp.iloc[valid_idx] if valid_idx else 0.0
    except IndexError:
        print(f"Index {max_index} out of bounds")
        return 0.0
    except Exception as e:
        print(f"Y-displacement extraction error: {e}")
        return 0.0


def extract_angle_node(fem_model):
    """Extract nodal information from message file using improved parsing."""
    msg_file = Path(fem_model) / "lspost.msg"
    nodes = {"center": "", "tri": "", "angles": []}
    
    try:
        with msg_file.open() as fp:
            for line in fp:
                if line.startswith("NODE"):
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                        
                    if not nodes["center"]:
                        nodes["center"] = parts[2]
                    elif not nodes["tri"]:
                        nodes["tri"] = parts[2]
                    else:
                        nodes["angles"].append(parts[2])
                        if len(nodes["angles"]) >= 3:
                            break
    except FileNotFoundError:
        print(f"Message file not found: {msg_file}")
        return "", "", ""
    
    cut_line = f"splane dep1 {nodes['center']} 0.000 1.000 0.000"
    angle_cmd = "measure angle3 N{}/0 N{}/0 N{}/0 ;".format(*nodes["angles"])
    return cut_line, angle_cmd, nodes["tri"]
