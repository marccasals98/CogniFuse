import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATASET_PATH = Path(
    "/home/usuaris/veussd/marc.casals/datasets/WAB_samples/labels.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the duration distribution of the labeled audio files."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Audio directory (default: <dataset-path parent>/audios).",
    )
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Plot path (default: <dataset-path parent>/audio_duration_histogram.png).",
    )
    parser.add_argument("--show", action="store_true", help="Also display the plot.")
    return parser.parse_args()


def get_audio_durations(dataset_path, audio_dir):
    df = pd.read_csv(dataset_path)
    filename_column = df.columns[0]
    durations = []
    missing_files = []
    unreadable_files = []

    for filename in df[filename_column].dropna().astype(str):
        audio_path = audio_dir / filename
        if not audio_path.is_file():
            missing_files.append(audio_path)
            continue

        try:
            durations.append(librosa.get_duration(path=audio_path))
        except Exception as error:
            unreadable_files.append((audio_path, error))

    return df, durations, missing_files, unreadable_files


def plot_duration_histogram(durations, bins, output_path, show):
    duration_series = pd.Series(durations, name="duration_seconds")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(duration_series, bins=bins, color="steelblue", edgecolor="black")
    ax.axvline(
        duration_series.mean(),
        color="darkorange",
        linestyle="--",
        label=f"Mean: {duration_series.mean():.1f} s",
    )
    ax.axvline(
        duration_series.median(),
        color="crimson",
        linestyle=":",
        label=f"Median: {duration_series.median():.1f} s",
    )
    ax.set_title("Audio duration distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Number of audio files")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return duration_series


def main():
    args = parse_args()
    audio_dir = args.audio_dir or args.dataset_path.parent / "audios"
    output_path = (
        args.output_path
        or args.dataset_path.parent / "audio_duration_histogram.png"
    )

    df, durations, missing_files, unreadable_files = get_audio_durations(
        args.dataset_path, audio_dir
    )
    if not durations:
        raise RuntimeError(f"No readable audio files found in {audio_dir}")

    duration_series = plot_duration_histogram(
        durations, args.bins, output_path, args.show
    )

    print(df.head())
    print(df["DX_Pilar"].unique())
    print(f"Audio files plotted: {len(duration_series)}")
    print(f"Missing audio files: {len(missing_files)}")
    print(f"Unreadable audio files: {len(unreadable_files)}")
    print(duration_series.describe().to_string())
    print(f"Histogram saved to: {output_path}")

    for audio_path in missing_files:
        print(f"Missing: {audio_path}")
    for audio_path, error in unreadable_files:
        print(f"Unreadable: {audio_path} ({error})")


if __name__ == "__main__":
    main()

