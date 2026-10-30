
import csv
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter

OUTPUT_FILE = Path(__file__).resolve().parent / "ML" / "toctou_training_data.csv"

FIELDS = [
    "check_duration_ms",
    "check_use_delay_ms",
    "use_duration_ms",
    "concurrent_requests",
    "duplicate_request_count",
    "resource_version_changed",
    "label",
]


def run_test(is_toctou):
    checked = Event()
    changed = Event()

    results = {}

    with TemporaryDirectory() as folder:
        resource = Path(folder) / "resource.txt"
        resource.write_text("SAFE")

        def application():
            start = perf_counter()

            # Time of check
            checked_content = resource.read_text()
            check_end = perf_counter()

            checked.set()
            changed.wait()

            # Time of use
            use_start = perf_counter()
            used_content = resource.read_text()
            use_end = perf_counter()

            results["check_duration_ms"] = (
                check_end - start
            ) * 1000

            results["check_use_delay_ms"] = (
                use_start - check_end
            ) * 1000

            results["use_duration_ms"] = (
                use_end - use_start
            ) * 1000

            results["resource_version_changed"] = int(
                checked_content != used_content
            )

        def controlled_change():
            checked.wait()

            if is_toctou:
                resource.write_text("UNSAFE")

            changed.set()

        app_thread = Thread(target=application)
        change_thread = Thread(target=controlled_change)

        app_thread.start()
        change_thread.start()

        app_thread.join()
        change_thread.join()

    results["concurrent_requests"] = 2 if is_toctou else 1
    results["duplicate_request_count"] = 1 if is_toctou else 0
    results["label"] = 1 if is_toctou else 0

    return results


def main():
    random.seed(42)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # 50 normal controlled tests
    for _ in range(50):
        rows.append(run_test(False))

    # 50 controlled TOCTOU tests
    for _ in range(50):
        rows.append(run_test(True))

    random.shuffle(rows)

    with OUTPUT_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("=== TOCTOU DATASET CREATED ===")
    print("Total records:", len(rows))
    print("Normal records:", sum(r["label"] == 0 for r in rows))
    print("TOCTOU records:", sum(r["label"] == 1 for r in rows))
    print("Dataset saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
